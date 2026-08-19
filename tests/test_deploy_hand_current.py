"""The finger current the recording carries, without a gripper on the desk.

A finger that stops short of its goal is either being held by the object or is
stuck, and the joint trace looks identical either way -- the current is what
separates them. So the parts worth pinning down are the ones whose bugs are
silent: a unit slip between the protocol's milliamps and the cap's amperes, a
sign dropped from a signed field, and a motor order that does not match the
joint order the rest of the recording is written in.
"""

from __future__ import annotations

import numpy as np
import pytest
from deploy import calib
from deploy.hand import Hand


def obs(*cur_ma: int) -> dict:
  """An OBS carrying one current per motor, in the protocol's own units."""
  return {"motors": [{"pos": 0, "vel": 0, "cur_ma": c, "temp": 30} for c in cur_ma]}


def test_currents_come_back_in_amperes():
  """The protocol packs milliamps; `--grip-cap` and `cap_a` are amperes."""
  assert Hand.currents_a(obs(800, 0, -800, 12)) == pytest.approx(
    [0.8, 0.0, -0.8, 0.012]
  )


def test_the_sign_survives():
  """int16, and the sign is which way the motor is pushing.

  Taking the magnitude here instead would erase the one thing that tells a
  finger leaning into a grip from one being driven back out of it.
  """
  assert Hand.currents_a(obs(0, 0, -650, 0))[2] < 0


def test_motor_order_is_the_joint_order_the_recording_uses():
  """Four distinct values, read back through `calib.JOINT_NAMES[7:11]`.

  The recording's `finger_current_a` column is indexed by this order and read
  back alongside `joint_pos[:, 7:11]`. A transposed pair would attribute a
  stalled left finger's current to the right one, which is exactly the
  asymmetry the column exists to measure.
  """
  a = Hand.currents_a(obs(100, 200, 300, 400))
  names = calib.JOINT_NAMES[calib.HAND_SLICE]
  assert names == ("left_1", "left_2", "right_1", "right_2")
  assert dict(zip(names, a, strict=True))["right_1"] == pytest.approx(0.3)


def test_a_motor_that_omitted_the_field_reads_zero_rather_than_raising():
  """Reporting only: it must never be able to take down a running loop."""
  o = obs(500, 500, 500, 500)
  del o["motors"][1]["cur_ma"]
  assert Hand.currents_a(o)[1] == 0.0


def test_the_shape_is_what_the_recorder_preallocated():
  assert Hand.currents_a(obs(1, 2, 3, 4)).shape == (4,)
  assert Hand.currents_a(obs(1, 2, 3, 4)).dtype == np.float32
