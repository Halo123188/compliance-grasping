"""How much torque the policy is allowed, which is not a deployment choice.

`max_offset_rad` caps |commanded - measured|, so it caps the impedance
controller's torque at K_q * offset. The TRAINING environment was torque-limited
too -- the wide-claw model's `actfrcrange` is [123, 123, 64, 64, 39, 39, 39] Nm,
exactly the Rizon 4S's own `tau_max` -- so the offset that reproduces training
is tau_max / K_q per joint, and anything smaller is a rate limit inside the
policy's closed loop.
"""

from __future__ import annotations

import numpy as np
import pytest
from deploy import calib
from deploy.arm import ReplayArm, resolve_max_offset

# What the robot reported on Rizon4s-063501, and what the sim's actfrcrange is.
TAU_MAX = np.array([123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0])
K_Q = np.asarray(calib.JOINT_STIFFNESS, float)[calib.ARM_SLICE]


def test_default_is_the_training_environments_own_saturation():
  offset, full = resolve_max_offset(None, TAU_MAX, K_Q)
  assert np.allclose(offset, full)
  assert np.allclose(offset.round(3), [0.426, 0.183, 0.286, 0.172, 0.165, 0.168, 0.210])
  # By construction the commanded torque is then exactly tau_max, no more.
  assert np.allclose(K_Q * offset, TAU_MAX)


def test_a_scalar_broadcasts_and_is_a_fraction_of_that():
  offset, full = resolve_max_offset(0.08, TAU_MAX, K_Q)
  assert offset.shape == (7,)
  assert np.all(offset == 0.08)
  frac = offset / full
  # The number that made the arm oscillate: 19% of joint1's authority, and no
  # joint above half. joint1 is the worst and is also the one that was swinging.
  assert (100 * frac).round().tolist() == [19.0, 44.0, 28.0, 47.0, 49.0, 48.0, 38.0]


def test_a_per_joint_vector_is_taken_as_given():
  want = [0.426, 0.183, 0.286, 0.172, 0.165, 0.168, 0.210]
  offset, full = resolve_max_offset(want, TAU_MAX, K_Q)
  assert np.allclose(offset, want)
  assert np.allclose(offset, full, atol=5e-4)


def test_a_wrong_length_vector_raises_rather_than_broadcasting():
  with pytest.raises(ValueError):
    resolve_max_offset([0.1, 0.2], TAU_MAX, K_Q)


def test_replay_arm_satisfies_the_interface():
  home = np.asarray(calib.DEFAULT_JOINT_POS)
  arm = ReplayArm(home)
  q, dq = arm.read()
  assert q.shape == (7,) and dq.shape == (7,)
  assert np.allclose(q, home[calib.ARM_SLICE])
  assert arm.tcp_pose() is None
  arm.set_speed_scale(0.5)
  arm.set_targets(q + 0.01)
  assert len(arm.commanded) == 1
  arm.stop()
