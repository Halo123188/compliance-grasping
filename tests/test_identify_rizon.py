"""Tests for the Rizon system-identification sweep's pure logic.

Everything here runs without a robot or the C++ bridge.  The parts worth pinning
down are the ones whose bugs are silent: a park target that leaves a joint with no
room (or worse, outside its limits), an excitation sized past a safety bound, and
a fit that returns confident numbers from data that never moved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjlab.tasks.compliance.deploy.identify_rizon import (
  DQ_CAP_FLOOR,
  DQ_CAP_FRAC,
  EFFORT_LIMIT,
  Q_LIMIT_MARGIN,
  Q_WINDOW,
  VSWEEP_LEVELS,
  design_ramps,
  design_sines,
  design_vsweeps,
  excitation_torque,
  parse_joints,
  plan_park,
)

# Rizon 4S joint limits and rated speeds, as the bridge publishes them.
Q_MIN = np.radians(np.array([-170.0, -140.0, -170.0, -60.0, -170.0, -115.0, -170.0]))
Q_MAX = np.radians(np.array([170.0, 140.0, 170.0, 159.0, 170.0, 115.0, 170.0]))
DQ_MAX = np.array([2.09, 2.09, 2.44, 2.44, 4.89, 4.89, 4.89])
# The pose the tracking policy resets to, where j4 sits 5 deg under its stop.
Q_TRACKING_HOME = np.array(
  [-0.225705, 0.003626, 0.507476, 2.438249, 0.002759, 0.864276, -1.290931]
)


def test_parse_joints_forms():
  assert parse_joints("1-7") == list(range(7))
  assert parse_joints("2") == [1]
  assert parse_joints("1,3,5") == [0, 2, 4]
  assert parse_joints("3,3,1") == [2, 0]  # deduped, order kept


@pytest.mark.parametrize("spec", ["0", "8", "5-2", "", "1,9"])
def test_parse_joints_rejects_bad_specs(spec):
  with pytest.raises(SystemExit):
    parse_joints(spec)


def test_plan_park_leaves_room_on_both_sides():
  """Every parked joint must have a full window inside the limit margin."""
  target = plan_park(Q_TRACKING_HOME, Q_MIN, Q_MAX)
  room_up = (Q_MAX - Q_LIMIT_MARGIN) - target
  room_down = target - (Q_MIN + Q_LIMIT_MARGIN)
  assert np.all(room_up >= Q_WINDOW - 1e-9), f"no room above: {room_up}"
  assert np.all(room_down >= Q_WINDOW - 1e-9), f"no room below: {room_down}"


def test_plan_park_only_moves_the_joint_that_needs_it():
  """j4 is the one joint short of room at the tracking home; the rest must not move."""
  target = plan_park(Q_TRACKING_HOME, Q_MIN, Q_MAX)
  moved = np.flatnonzero(np.abs(target - Q_TRACKING_HOME) > 1e-9)
  assert moved.tolist() == [3]
  # And it moves the minimum: exactly onto the edge of the legal band.
  assert target[3] == pytest.approx(Q_MAX[3] - Q_LIMIT_MARGIN - Q_WINDOW)
  assert target[3] < Q_TRACKING_HOME[3]  # away from the stop, not toward it


def test_plan_park_mid_centres_every_joint():
  target = plan_park(Q_TRACKING_HOME, Q_MIN, Q_MAX, mid=True)
  assert np.allclose(target, 0.5 * (Q_MIN + Q_MAX))


def test_plan_park_never_exceeds_limits_even_from_an_illegal_pose():
  """A start angle past a stop must not produce a target past it too."""
  q = Q_MAX + 0.5
  target = plan_park(q, Q_MIN, Q_MAX)
  assert np.all(target <= Q_MAX - Q_LIMIT_MARGIN + 1e-9)
  assert np.all(target >= Q_MIN + Q_LIMIT_MARGIN - 1e-9)


def test_plan_park_handles_a_joint_with_no_legal_band():
  """A joint whose travel is shorter than 2*(margin + window) parks dead centre."""
  q_min = np.zeros(7)
  q_max = np.full(7, 0.2)  # far too short for a 0.30 rad window
  target = plan_park(np.full(7, 0.19), q_min, q_max)
  assert np.allclose(target, 0.1)


def test_design_ramps_skips_a_direction_without_room():
  """A ramp needs only enough travel to start moving, not a full window.

  ``Q_WINDOW_MIN``, not ``Q_WINDOW``, is the threshold: breakaway is detected the
  moment the joint clears 0.03 rad/s, which takes a few hundredths of a radian.  So
  j4 at the tracking home, with 0.087 rad above it, still gets both directions --
  and both is what yields the gravity-comp residual the sines need.
  """
  assert len(design_ramps(10.0, room_pos=0.30, room_neg=0.30)) == 2
  assert len(design_ramps(10.0, room_pos=0.087, room_neg=0.30)) == 2
  one = design_ramps(10.0, room_pos=0.01, room_neg=0.30)
  assert len(one) == 1 and one[0].sign < 0
  assert design_ramps(10.0, room_pos=0.0, room_neg=0.0) == []


@pytest.mark.parametrize("joint", range(7))
def test_sines_respect_both_the_position_window_and_the_speed_cap(joint):
  """The sizing bound that the first hardware run violated on all seven joints.

  A sine of amplitude A at omega on inertia I has position amplitude A/(I omega^2)
  and velocity amplitude A/(I omega).  Fixing the position and letting the velocity
  fall out gave 0.94 rad/s at 1 Hz against a 0.80 rad/s cap, so every block aborted.
  """
  inertia = np.array([3.80, 4.42, 2.10, 2.48, 0.19, 0.18, 0.13])[joint]
  tau_cap = 0.25 * EFFORT_LIMIT[joint]
  dq_cap = max(DQ_CAP_FLOOR, DQ_CAP_FRAC * DQ_MAX[joint])
  x_max = 0.5 * Q_WINDOW
  for exc in design_sines(inertia, 2.0, tau_cap, x_max, dq_cap):
    omega = 2.0 * math.pi * exc.freq
    assert exc.tau_amp <= tau_cap + 1e-9
    assert exc.tau_amp / (inertia * omega**2) <= x_max + 1e-9
    assert exc.tau_amp / (inertia * omega) <= dq_cap + 1e-9


def test_sine_fades_in_over_its_first_cycle():
  """Switching a sine on at full amplitude doubles the first swing's velocity."""
  (exc, _) = design_sines(1.0, 1.0, 10.0, 0.15, 0.5)
  quarter = 0.25 / exc.freq
  # A quarter period in, the envelope is at 1/4 while the sine itself peaks.
  assert excitation_torque(exc, quarter) == pytest.approx(0.25 * exc.tau_amp, rel=1e-6)
  # A full period and beyond, the envelope is done and the shape is a pure sine.
  assert excitation_torque(exc, 1.25 / exc.freq) == pytest.approx(exc.tau_amp, rel=1e-6)


def test_sine_offset_cancels_the_gravity_comp_residual():
  """Without this the joint ratchets out of the window instead of oscillating."""
  (exc, _) = design_sines(1.0, 1.0, 10.0, 0.15, 0.5, offset=-0.3)
  # Averaged over a whole cycle past the envelope, the command is the offset.
  t0 = 1.0 / exc.freq
  ts = t0 + np.linspace(0.0, 1.0 / exc.freq, 2001)
  mean = np.mean([excitation_torque(exc, float(t)) for t in ts])
  assert mean == pytest.approx(-0.3, abs=1e-3)


def test_sine_amplitude_leaves_headroom_for_its_offset():
  """Amplitude plus offset must still fit under the torque cap."""
  for exc in design_sines(1.0, 20.0, 10.0, 0.15, 0.5, offset=-3.0):
    assert exc.tau_amp + abs(exc.offset) <= 10.0 + 1e-9


def test_vsweeps_span_a_range_of_speeds_under_the_cap():
  """Damping is only separable from Coulomb friction across distinct speeds."""
  sweeps = design_vsweeps(2.0, 2.0, 16.0, x_rev=0.12, v_max=0.5)
  assert len(sweeps) == len(VSWEEP_LEVELS)
  vels = [e.vel for e in sweeps]
  assert vels == sorted(vels)
  assert max(vels) <= 0.5 + 1e-9
  assert max(vels) / min(vels) >= 2.0  # enough spread to resolve damping


def test_vsweep_torque_is_a_velocity_loop_that_reverses():
  (exc,) = design_vsweeps(2.0, 1.0, 16.0, x_rev=0.12, v_max=0.5)[:1]
  # At rest, drive toward the target; the two directions are mirror images.
  fwd = excitation_torque(exc, 0.0, dq=0.0, direction=+1.0)
  rev = excitation_torque(exc, 0.0, dq=0.0, direction=-1.0)
  assert fwd > 0.0 and rev == pytest.approx(-fwd)
  # Overspeed pulls back.
  assert excitation_torque(exc, 0.0, dq=2.0 * exc.vel, direction=+1.0) < fwd
