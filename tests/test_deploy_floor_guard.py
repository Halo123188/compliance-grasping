"""The floor guard, replayed against the run it was written for.

On 2026-08-07 the arm was driven into the foam: the pads dropped 213 mm in
0.52 s and `--max-jump`, which watches joint tracking error, did not trip until
two steps after they were already under the surface. This replays the exact
online rule `deploy/run.py` uses against the recorded joint trajectories and
requires that it would have stopped that run with real clearance left.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from deploy import calib
from deploy.kinematics import gripper_height

REPO = Path(__file__).resolve().parents[1]
PERIOD = 1.0 / calib.CONTROL_HZ
FOAM = calib.WORK_SURFACE_Z - calib.ARM_BASE_Z
LOOKAHEAD = 0.15  # deploy/run.py --floor-lookahead default
# Measured over 64 sim envs x 300 steps of this checkpoint: the policy never
# takes the pads below this, i.e. 22 mm of clearance over the foam.
SIM_MIN_PAD_Z = 0.057


def replay(joint_pos: np.ndarray, floor: float = FOAM, look: float = LOOKAHEAD):
  """-> (step it fires on or None, pad height there, descent rate there)."""
  h_prev = None
  for i, q in enumerate(joint_pos):
    h = gripper_height(q)
    descent = 0.0 if h_prev is None else max(0.0, (h_prev - h) / PERIOD)
    h_prev = h
    if h - descent * look < floor:
      return i, h, descent
  return None, None, None


def _load(name: str) -> np.ndarray:
  path = REPO / name
  if not path.exists():
    pytest.skip(f"{name} not present")
  return np.load(path)["joint_pos"]


def test_the_default_floor_is_below_anything_sim_does():
  """A floor the policy's own trained behaviour would trip is a broken floor."""
  assert FOAM < SIM_MIN_PAD_Z
  assert (SIM_MIN_PAD_Z - FOAM) > 0.020  # 22 mm of headroom


def test_it_stops_the_crash_with_clearance_left():
  q = _load("run3.npz")
  step, h, descent = replay(q)
  # `replay` returns (None, None, None) when the guard never fires, so h and
  # descent are only numbers once step is. Asserting all three keeps the later
  # arithmetic honest instead of raising a TypeError that reads like a bug in
  # the guard.
  assert step is not None, "the guard must fire on the run that hit the foam"
  assert h is not None and descent is not None
  assert h - FOAM > 0.050, f"only {1000 * (h - FOAM):.0f} mm of clearance"
  assert descent > 0.3, "it should fire while genuinely diving"
  # And well before the breach, which was at step 124 of 127.
  first_breach = int(np.argmax([gripper_height(x) < FOAM for x in q]))
  assert step < first_breach - 5


def test_a_purely_reactive_floor_would_have_been_too_late():
  """Why --floor-lookahead is not 0: height alone only fires after the fact."""
  q = _load("run3.npz")
  late, h_late, _ = replay(q, look=0.0)
  early, _, _ = replay(q)
  assert late is not None and early is not None
  assert early < late
  assert h_late is not None and h_late < FOAM  # already under the surface


def test_the_slow_run_is_not_aborted_early():
  """run1 never went below +4 mm; the guard must not cut it short."""
  q = _load("run1.npz")
  step, _, _ = replay(q)
  # It does fire eventually -- run1 got to 4 mm above the foam, which is
  # itself closer than sim ever goes -- but only in the last few percent.
  assert step is None or step > 0.9 * len(q)


def test_disabling_is_explicit():
  q = _load("run3.npz")
  step, _, _ = replay(q, floor=-1.0)
  assert step is None
