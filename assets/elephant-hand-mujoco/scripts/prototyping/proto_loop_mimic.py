"""Option B: joint-mimic loop closure.

Treat each finger as a 1-DOF mechanism driven by its BASE joint (Revolute 1/2/3).
For a grid of base-joint values, solve the loop (the 3 other loop joints) so the
duplicate coupler bodies coincide (target relpose). This yields passive-vs-driven
curves. Fit them (polynomial) -> these become mjEQ_JOINT polycoef couplings, OR
we confirm a near-linear ratio and use a simple coupling.

This avoids the workspace-weld redundancy/conditioning blowup entirely: the loop
is enforced in JOINT space.

Run: /home/alok/.conda/envs/g313/bin/python scripts/proto_loop_mimic.py
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
    "base": "Revolute 1",
    "loop": ["Revolute 4", "nRevolute 5", "nCylindrical 1", "nRevolute 6"],
  },
  {
    "name": "F2",
    "b1": "finger3_step",
    "b2": "finger3_step_2",
    "base": "Revolute 2",
    "loop": ["Revolute 7", "nRevolute 8", "nCylindrical 2", "nRevolute 9"],
  },
  {
    "name": "F3",
    "b1": "finger4_step",
    "b2": "finger4_step_2",
    "base": "Revolute 3",
    "loop": ["Revolute 10", "nRevolute 11", "nCylindrical 3", "nRevolute 12"],
  },
]


def mesh_T(name):
  xyz, rpy = MESH_ORIGINS[name]
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
  rel_R = R1.T @ R2
  rel_p = R1.T @ (p2 - p1)
  q = R.from_matrix(rel_R).as_quat()
  return rel_p, np.array([q[3], q[0], q[1], q[2]])


ROT_W = 0.05
print("=== sweep base joint, solve loop, extract passive curves ===", flush=True)
results = {}
for f in FINGERS:
  tpos, tquat = relpose(f["b1"], f["b2"])
  base_adr = jadr(f["base"])
  # passive joints = loop joints (the base joint is upstream, NOT in loop list)
  passive = list(f["loop"])
  passive_adrs = [jadr(j) for j in passive]

  base_grid = np.linspace(-0.4, 1.4, 19)  # range we expect to drive
  curves = {j: [] for j in passive}
  valid_base = []
  x_prev = np.zeros(len(passive))
  for bv in base_grid:

    def resid(x):
      mujoco.mj_resetData(m, d)
      d.qpos[base_adr] = bv
      for adr, v in zip(passive_adrs, x):
        d.qpos[adr] = v
      mujoco.mj_forward(m, d)
      rp, rq = body_rel(f["b1"], f["b2"])
      rq_inv = np.array([rq[0], -rq[1], -rq[2], -rq[3]])
      qd = np.zeros(4)
      mujoco.mju_mulQuat(qd, rq_inv, tquat)
      if qd[0] < 0:
        qd = -qd
      return np.concatenate([rp - tpos, ROT_W * 2.0 * qd[1:]])

    sol = least_squares(resid, x_prev, method="lm", max_nfev=500)
    cost = np.linalg.norm(resid(sol.x))
    if cost < 1e-4:
      x_prev = sol.x
      valid_base.append(bv)
      for j, v in zip(passive, sol.x):
        curves[j].append(v)
  valid_base = np.array(valid_base)
  results[f["name"]] = (valid_base, curves, passive)
  print(
    f"\n{f['name']} (base={f['base']}): {len(valid_base)} valid configs "
    f"over base in [{valid_base.min():.2f},{valid_base.max():.2f}]",
    flush=True,
  )
  for j in passive:
    y = np.array(curves[j])
    # linear fit slope (ratio) + quadratic to gauge nonlinearity
    c1 = np.polyfit(valid_base, y, 1)
    c2 = np.polyfit(valid_base, y, 2)
    lin_res = np.max(np.abs(np.polyval(c1, valid_base) - y))
    quad_res = np.max(np.abs(np.polyval(c2, valid_base) - y))
    print(
      f"   {j:16s} range[{y.min():+.3f},{y.max():+.3f}] "
      f"linfit slope={c1[0]:+.4f} off={c1[1]:+.4f} "
      f"lin_maxres={np.degrees(lin_res):.2f}deg quad_maxres={np.degrees(quad_res):.2f}deg",
      flush=True,
    )
