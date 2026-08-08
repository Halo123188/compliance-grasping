"""The transcribed chain against MuJoCo's own, on real recorded poses.

`deploy/kinematics.py` exists so the loop can refuse a target that would drive
the gripper into the bench, which means a wrong transcription would disable the
guard silently. So it is checked against `mujoco.mj_kinematics` on the compiled
model -- including on the joint trajectories recorded from hardware, which is
where a chain that is right at the home pose and wrong elsewhere shows up.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from deploy import calib
from deploy.kinematics import forward, gripper_height

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[1]
# Bodies carry the asset's `_tf` suffix in the model; deploy/ drops it.
PAIRS = {
  "link1": "link1",
  "link4": "link4",
  "link7": "link7",
  "hand_base": "hand_base_tf",
  "left_pad": "left_pad_tf",
  "right_pad": "right_pad_tf",
}


@pytest.fixture(scope="module")
def reference():
  """MuJoCo FK on the same model, as a callable q -> {deploy name: position}."""
  mujoco = pytest.importorskip("mujoco", reason="mjlab venv only")
  from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (  # noqa: PLC0415
    twofinger_arm_spec,
  )

  # The spec is compiled WITHOUT lifting `base` to ARM_BASE_Z, so the model's
  # world frame already is deploy's base frame and no offset is subtracted.
  m = twofinger_arm_spec().compile()
  d = mujoco.MjData(m)
  bid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in PAIRS.items()}
  adr = [
    m.jnt_qposadr[
      mujoco.mj_name2id(
        m,
        mujoco.mjtObj.mjOBJ_JOINT,
        n if n.startswith("joint") else f"{n}_tf",
      )
    ]
    for n in calib.JOINT_NAMES
  ]

  def fk(q):
    d.qpos[adr] = np.asarray(q, float)
    mujoco.mj_kinematics(m, d)
    return {k: d.xpos[i] for k, i in bid.items()}

  return fk


def _random_poses(n: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  home = np.asarray(calib.DEFAULT_JOINT_POS)
  return home + rng.uniform(-1.2, 1.2, size=(n, 11))


def test_home_pose_matches_mujoco(reference):
  q = np.asarray(calib.DEFAULT_JOINT_POS)
  mine, theirs = forward(q), reference(q)
  for name in PAIRS:
    assert np.allclose(mine[name], theirs[name], atol=1e-6), name


def test_random_poses_match_mujoco(reference):
  worst = 0.0
  for q in _random_poses(64):
    mine, theirs = forward(q), reference(q)
    for name in PAIRS:
      worst = max(worst, float(np.abs(mine[name] - theirs[name]).max()))
  assert worst < 1e-6, f"worst disagreement {worst:.2e} m"


@pytest.mark.parametrize("run", ["run1.npz", "run2.npz", "run3.npz"])
def test_recorded_hardware_trajectories_match_mujoco(reference, run):
  """The poses the arm actually visited, which is what the guard will see."""
  path = REPO / run
  if not path.exists():
    pytest.skip(f"{run} not present")
  q = np.load(path)["joint_pos"]
  worst = 0.0
  for row in q[:: max(1, len(q) // 40)]:
    mine, theirs = forward(row), reference(row)
    for name in PAIRS:
      worst = max(worst, float(np.abs(mine[name] - theirs[name]).max()))
  assert worst < 1e-6, f"worst disagreement {worst:.2e} m"


def test_gripper_height_is_the_lower_pad():
  q = np.asarray(calib.DEFAULT_JOINT_POS)
  p = forward(q)
  assert gripper_height(q) == pytest.approx(min(p["left_pad"][2], p["right_pad"][2]))


def test_the_run3_dive_is_detected_below_the_foam():
  """The incident: pads 17 mm under the foam while joint space looked fine."""
  path = REPO / "run3.npz"
  if not path.exists():
    pytest.skip("run3.npz not present")
  q = np.load(path)["joint_pos"]
  foam = calib.WORK_SURFACE_Z - calib.ARM_BASE_Z
  h = np.array([gripper_height(row) for row in q])
  assert h.min() < foam, "the dive must read as below the foam"
  # And it must be catchable EARLIER than the joint-space guard managed: the
  # first breach was step 124 of 127, two steps before --max-jump tripped.
  assert int(np.argmax(h < foam)) < len(q) - 1
