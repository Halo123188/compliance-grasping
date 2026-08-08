"""Where the gripper is, from joint angles alone. Numpy only.

`deploy/` has no mujoco and no URDF parser, but the one guard the loop was
missing needs Cartesian space: `--max-jump` measures JOINT TRACKING ERROR, which
can only grow after the arm has already failed to follow, and by then the arm is
where it was going. On 2026-08-07 that cost a run -- the pads descended 213 mm
in 0.52 s (0.41 m/s) and were 1 mm under the foam by the time the joint-space
guard reached its threshold two steps later.

So the chain is transcribed here from the compiled model and evaluated in
closed form. It is a claim about the same geometry `arm_cfg.py` builds, checked
against `mujoco.mj_kinematics` in `tests/test_deploy_kinematics.py` to 1e-6 m
over the recorded hardware runs.

Frame is the ROBOT BASE (see deploy/check_camera.py): origin at the top of the
arm's mounting plate, +x forward over the bench, +y to the arm's left, +z up.
"""

from __future__ import annotations

import numpy as np

# (body, parent, body_pos, body_quat wxyz, (joint, axis, anchor) or None), read
# off the compiled wide-claw model. Regenerate by walking `body_parentid` from
# left_pad_tf/right_pad_tf up to `base` and printing body_pos/body_quat/jnt_*.
LINKS: tuple = (
  ("link1", None, (0.0, 0.0, 0.155), (0.0, 0.0, 0.0, 1.0), ("joint1", (0.0, 0.0, 1.0))),
  (
    "link2",
    "link1",
    (0.0, 0.03, 0.21),
    (1.0, 0.0, 0.0, 0.0),
    ("joint2", (0.0, 1.0, 0.0)),
  ),
  (
    "link3",
    "link2",
    (0.0, 0.035, 0.205),
    (1.0, 0.0, 0.0, 0.0),
    ("joint3", (0.0, 0.0, 1.0)),
  ),
  (
    "link4",
    "link3",
    (-0.02, -0.03, 0.19),
    (0.0, 0.0, 0.0, 1.0),
    ("joint4", (0.0, 1.0, 0.0)),
  ),
  (
    "link5",
    "link4",
    (-0.02, 0.025, 0.195),
    (0.0, 0.0, 0.0, 1.0),
    ("joint5", (0.0, 0.0, 1.0)),
  ),
  (
    "link6",
    "link5",
    (0.0, 0.03, 0.19),
    (1.0, 0.0, 0.0, 0.0),
    ("joint6", (0.0, 1.0, 0.0)),
  ),
  (
    "link7",
    "link6",
    (-0.015, 0.073, 0.11),
    (0.707106781, 0.0, -0.707106781, 0.0),
    ("joint7", (0.0, 0.0, 1.0)),
  ),
  (
    "hand_base",
    "link7",
    (3.02e-07, -2.032e-05, 0.147999996),
    (0.0, -0.923879754, -0.382682898, 0.0),
    None,
  ),
  (
    "left_1",
    "hand_base",
    (-0.035, -0.0145, -0.0345),
    (1.0, 0.0, 0.0, 0.0),
    ("left_1", (0.0, 1.0, 0.0)),
  ),
  (
    "left_2",
    "left_1",
    (0.0, 0.0, -0.0445),
    (1.0, 0.0, 0.0, 0.0),
    ("left_2", (0.0, 1.0, 0.0)),
  ),
  (
    "left_pad",
    "left_2",
    (0.010000001, 0.01149749, -0.022326534),
    (1.0, 0.0, 0.0, 0.0),
    None,
  ),
  (
    "right_1",
    "hand_base",
    (0.035, -0.0145, -0.0345),
    (1.0, 0.0, 0.0, 0.0),
    ("right_1", (0.0, 1.0, 0.0)),
  ),
  (
    "right_2",
    "right_1",
    (0.0, 0.0, -0.0445),
    (1.0, 0.0, 0.0, 0.0),
    ("right_2", (0.0, 1.0, 0.0)),
  ),
  (
    "right_pad",
    "right_2",
    (-0.010000001, 0.01149749, -0.022326534),
    (1.0, 0.0, 0.0, 0.0),
    None,
  ),
)

# Every joint anchor in this chain is at its body's origin, so the hinge is a
# plain rotation of the body frame with no translate-rotate-translate. Asserted
# by the parity test rather than assumed.
_JOINT_ORDER = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
  "left_1",
  "left_2",
  "right_1",
  "right_2",
)


def _quat_to_mat(q) -> np.ndarray:
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _axis_angle(axis, angle: float) -> np.ndarray:
  a = np.asarray(axis, float)
  c, s = np.cos(angle), np.sin(angle)
  k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
  return np.eye(3) + s * k + (1 - c) * (k @ k)


def forward(q: np.ndarray) -> dict[str, np.ndarray]:
  """joint_pos[11] in JOINT_NAMES order -> {body: position[3]} in the base frame."""
  q = np.asarray(q, dtype=float).reshape(11)
  angles = dict(zip(_JOINT_ORDER, q, strict=True))
  pos: dict[str, np.ndarray] = {}
  rot: dict[str, np.ndarray] = {}
  for name, parent, bpos, bquat, joint in LINKS:
    p0 = np.zeros(3) if parent is None else pos[parent]
    r0 = np.eye(3) if parent is None else rot[parent]
    r = r0 @ _quat_to_mat(bquat)
    p = p0 + r0 @ np.asarray(bpos, float)
    if joint is not None:
      jname, axis = joint
      r = r @ _axis_angle(axis, angles[jname])
    pos[name], rot[name] = p, r
  return pos


def gripper_height(q: np.ndarray) -> float:
  """Lowest of the two pad origins, in metres above the arm's base plate.

  The pad ORIGIN, not the lowest point of the pad geom -- so this is optimistic
  by however far the pad extends below its frame, and the floor it is compared
  against has to carry that margin.
  """
  p = forward(q)
  return float(min(p["left_pad"][2], p["right_pad"][2]))
