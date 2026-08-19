"""Option B v2: identify the true 4-bar input joint and extract mimic curves.

Revolute 1/2/3 are NOT in the loop (they swing the whole finger). The 4-bar's
single internal DOF is one of the loop joints. We pick the prompt's 2nd driven
joint (nRevolute 5/8/11) as the INPUT, sweep it, solve the remaining 3 loop joints
to keep the coupler copies coincident, and extract passive curves + poly fits +
fingertip travel.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "src/my_mjlab_project/assets/robots/mygripper_H100_R/robot.urdf"
MESH_ORIGINS = {
  "s_link2_step": ((0.0005, 0.0005, 0.001), (0, 0, 0)),
  "s_link2_step_2": ((-0.0334378, 0.016261, -0.008), (0, 0, -0.780037)),
  "finger3_step": ((0.0, 0.0, 0.0009), (-3.14159, 0, 0)),
  "finger3_step_2": ((-0.0359102, -0.0134137, -0.0079), (-3.14159, 0, 0.652269)),
  "finger4_step": ((0.0, 0.0, 0.0009), (0, 0, 0)),
  "finger4_step_2": ((-0.0377587, 0.00661434, -0.008), (0, 0, -0.468199)),
}
FINGERS = [
  {
    "name": "F1",
    "b1": "s_link2_step",
    "b2": "s_link2_step_2",
    "input": "nRevolute 5",
    "tip": "s_link2_step_2",
    "loop": ["Revolute 4", "nRevolute 5", "nCylindrical 1", "nRevolute 6"],
  },
  {
    "name": "F2",
    "b1": "finger3_step",
    "b2": "finger3_step_2",
    "input": "nRevolute 8",
    "tip": "finger3_step_2",
    "loop": ["Revolute 7", "nRevolute 8", "nCylindrical 2", "nRevolute 9"],
  },
  {
    "name": "F3",
    "b1": "finger4_step",
    "b2": "finger4_step_2",
    "input": "nRevolute 11",
    "tip": "finger4_step_2",
    "loop": ["Revolute 10", "nRevolute 11", "nCylindrical 3", "nRevolute 12"],
  },
]


def mesh_T(n):
  xyz, rpy = MESH_ORIGINS[n]
  T = np.eye(4)
  T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
  T[:3, 3] = xyz
  return T


def relpose(b1, b2):
  rel = mesh_T(b1) @ np.linalg.inv(mesh_T(b2))
  q = R.from_matrix(rel[:3, :3]).as_quat()
  return rel[:3, 3].copy(), np.array([q[3], q[0], q[1], q[2]])


spec = mujoco.MjSpec.from_file(str(HAND_URDF))
m = spec.compile()
d = mujoco.MjData(m)


def bid(n):
  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def jadr(n):
  return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])


def body_rel(b1, b2):
  p1, p2 = d.xpos[bid(b1)], d.xpos[bid(b2)]
  R1 = d.xmat[bid(b1)].reshape(3, 3)
  R2 = d.xmat[bid(b2)].reshape(3, 3)
  q = R.from_matrix(R1.T @ R2).as_quat()
  return R1.T @ (p2 - p1), np.array([q[3], q[0], q[1], q[2]])


ROT_W = 0.05

for f in FINGERS:
  tpos, tquat = relpose(f["b1"], f["b2"])
  inp = f["input"]
  inp_adr = jadr(inp)
  passive = [j for j in f["loop"] if j != inp]
  passive_adrs = [jadr(j) for j in passive]
  grid = np.linspace(-0.6, 1.2, 25)
  cols = {j: [] for j in passive}
  valid = []
  tips = []
  x_prev = np.zeros(len(passive))
  for iv in grid:

    def resid(x):
      mujoco.mj_resetData(m, d)
      d.qpos[inp_adr] = iv
      for a, v in zip(passive_adrs, x):
        d.qpos[a] = v
      mujoco.mj_forward(m, d)
      rp, rq = body_rel(f["b1"], f["b2"])
      rqi = np.array([rq[0], -rq[1], -rq[2], -rq[3]])
      qd = np.zeros(4)
      mujoco.mju_mulQuat(qd, rqi, tquat)
      if qd[0] < 0:
        qd = -qd
      return np.concatenate([rp - tpos, ROT_W * 2.0 * qd[1:]])

    sol = least_squares(resid, x_prev, method="lm", max_nfev=500)
    if np.linalg.norm(resid(sol.x)) < 1e-4:
      x_prev = sol.x
      valid.append(iv)
      for j, v in zip(passive, sol.x):
        cols[j].append(v)
      mujoco.mj_resetData(m, d)
      d.qpos[inp_adr] = iv
      for a, v in zip(passive_adrs, sol.x):
        d.qpos[a] = v
      mujoco.mj_forward(m, d)
      tips.append(d.xpos[bid(f["tip"])].copy())
  valid = np.array(valid)
  tips = np.array(tips)
  travel = np.linalg.norm(tips[-1] - tips[0]) * 1000 if len(tips) > 1 else 0
  print(
    f"\n{f['name']} input={inp}: {len(valid)} closeable over [{valid.min():.2f},{valid.max():.2f}]  TIP travel={travel:.1f} mm",
    flush=True,
  )
  for j in passive:
    y = np.array(cols[j])
    c3 = np.polyfit(valid, y, 3)
    res3 = np.degrees(np.max(np.abs(np.polyval(c3, valid) - y)))
    print(
      f"   {j:16s} range[{y.min():+.3f},{y.max():+.3f}] cubic={np.round(c3, 5).tolist()} maxres={res3:.2f}deg",
      flush=True,
    )
