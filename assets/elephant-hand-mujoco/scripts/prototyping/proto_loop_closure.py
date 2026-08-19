"""Prototype four-bar loop closure for mygripper_H100_R (hand-alone).

Step 1: compute the weld target relpose M1*inv(M2) per finger (from URDF mesh origins).
Step 2: solve an assembly config (loop joint angles minimizing body2-in-body1 pose
        error vs target) via scipy least_squares -> set as qpos0.
Step 3: add 3 welds via MjSpec.
Step 4: validate headless (drive driven joints, check finite + residual + travel).

Run: /home/alok/.conda/envs/g313/bin/python scripts/proto_loop_closure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "src/my_mjlab_project/assets/robots/mygripper_H100_R/robot.urdf"

# Mesh-placement origins of the two coupler copies (from URDF <visual>/<collision>).
# xyz, rpy (XYZ-euler radians). M = T(xyz) * R(rpy).
MESH_ORIGINS = {
  "s_link2_step": ((0.0005, 0.0005, 0.001), (0, 0, 0)),
  "s_link2_step_2": ((-0.0334378, 0.016261, -0.008), (0, 0, -0.780037)),
  "finger3_step": ((0.0, 0.0, 0.0009), (-3.14159, 0, 0)),
  "finger3_step_2": ((-0.0359102, -0.0134137, -0.0079), (-3.14159, 0, 0.652269)),
  "finger4_step": ((0.0, 0.0, 0.0009), (0, 0, 0)),
  "finger4_step_2": ((-0.0377587, 0.00661434, -0.008), (0, 0, -0.468199)),
}

# (body1, body2, driven joints, loop joints, passive loop joints to solve)
# passive = loop joints that are NOT driven (we solve these to close the loop).
FINGERS = [
  {
    "name": "F1",
    "b1": "s_link2_step",
    "b2": "s_link2_step_2",
    "driven": ["Revolute 1", "nRevolute 5"],
    "loop": ["Revolute 4", "nRevolute 5", "nCylindrical 1", "nRevolute 6"],
  },
  {
    "name": "F2",
    "b1": "finger3_step",
    "b2": "finger3_step_2",
    "driven": ["Revolute 2", "nRevolute 8"],
    "loop": ["Revolute 7", "nRevolute 8", "nCylindrical 2", "nRevolute 9"],
  },
  {
    "name": "F3",
    "b1": "finger4_step",
    "b2": "finger4_step_2",
    "driven": ["Revolute 3", "nRevolute 11"],
    "loop": ["Revolute 10", "nRevolute 11", "nCylindrical 3", "nRevolute 12"],
  },
]


def mesh_T(name: str) -> np.ndarray:
  """4x4 mesh-placement transform from URDF origin (xyz, rpy XYZ-euler)."""
  xyz, rpy = MESH_ORIGINS[name]
  T = np.eye(4)
  T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
  T[:3, 3] = xyz
  return T


def relpose(b1: str, b2: str) -> tuple[np.ndarray, np.ndarray]:
  """Target relpose of body2 in body1 frame so the two mesh copies coincide:
  X_b1 * M1 = X_b2 * M2  =>  X_b2^-1 X_b1 ... we want body2 in body1:
  rel = M1 * inv(M2). Returns (pos[3], quat[wxyz] 4)."""
  M1, M2 = mesh_T(b1), mesh_T(b2)
  rel = M1 @ np.linalg.inv(M2)
  pos = rel[:3, 3].copy()
  quat_xyzw = R.from_matrix(rel[:3, :3]).as_quat()  # x,y,z,w
  quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
  return pos, quat_wxyz


# ---------------------------------------------------------------------------
spec = mujoco.MjSpec.from_file(str(HAND_URDF))
m = spec.compile()
d = mujoco.MjData(m)


def bid(n):
  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def jadr(n):
  j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
  return int(m.jnt_qposadr[j])


def body_rel(b1: str, b2: str, d) -> tuple[np.ndarray, np.ndarray]:
  """Actual relpose of body2 in body1 frame from current mjData (pos, quat wxyz)."""
  p1, p2 = d.xpos[bid(b1)], d.xpos[bid(b2)]
  R1 = d.xmat[bid(b1)].reshape(3, 3)
  R2 = d.xmat[bid(b2)].reshape(3, 3)
  rel_R = R1.T @ R2
  rel_p = R1.T @ (p2 - p1)
  q_xyzw = R.from_matrix(rel_R).as_quat()
  q_wxyz = np.array([q_xyzw[3], *q_xyzw[:3]])
  return rel_p, q_wxyz


def quat_geodesic(qa: np.ndarray, qb: np.ndarray) -> float:
  """Angle (rad) between two wxyz quats."""
  dot = abs(float(np.dot(qa, qb)))
  dot = min(1.0, dot)
  return 2.0 * np.arccos(dot)


print("=== Step 1: target weld relpose (body2 in body1) ===")
targets = {}
for f in FINGERS:
  pos, quat = relpose(f["b1"], f["b2"])
  targets[f["name"]] = (pos, quat)
  print(f"{f['name']}: {f['b1']} <- {f['b2']}", flush=True)
  print(f"   pos  = {np.round(pos, 6).tolist()}", flush=True)
  print(f"   quat = {np.round(quat, 6).tolist()}  (wxyz)", flush=True)


print("\n=== Step 2: solve assembly config (all 4 loop joints per finger) ===")
ROT_W = 0.05  # weight rotation error (rad) relative to position error (m)
assembly = {}
for f in FINGERS:
  tpos, tquat = targets[f["name"]]
  loop_adrs = [jadr(j) for j in f["loop"]]

  def resid(x, loop_adrs=loop_adrs, tpos=tpos, tquat=tquat, f=f):
    mujoco.mj_resetData(m, d)
    for adr, v in zip(loop_adrs, x, strict=False):
      d.qpos[adr] = v
    mujoco.mj_forward(m, d)
    rp, rq = body_rel(f["b1"], f["b2"], d)
    perr = rp - tpos
    # quaternion log-error (vector part of relative quat), scaled
    qd = np.zeros(4)
    # rq^-1 * tquat
    rq_inv = np.array([rq[0], -rq[1], -rq[2], -rq[3]])
    mujoco.mju_mulQuat(qd, rq_inv, tquat)
    if qd[0] < 0:
      qd = -qd
    rerr = 2.0 * qd[1:]  # small-angle approx of rotation error vector
    return np.concatenate([perr, ROT_W * rerr])

  best = None
  rng = np.random.default_rng(0)
  seeds = [np.zeros(4)]
  # add random seeds for robustness (multi-start, basin escape)
  for _ in range(12):
    seeds.append(rng.uniform(-np.pi, np.pi, 4))
  for s in seeds:
    sol = least_squares(resid, s, method="lm", max_nfev=400)
    cost = np.linalg.norm(resid(sol.x))
    if best is None or cost < best[0]:
      best = (cost, sol.x)
    if best[0] < 1e-5:  # already essentially closed; stop early
      break

  x = best[1]
  # report final residual decomposed
  mujoco.mj_resetData(m, d)
  for adr, v in zip(loop_adrs, x, strict=False):
    d.qpos[adr] = v
  mujoco.mj_forward(m, d)
  rp, rq = body_rel(f["b1"], f["b2"], d)
  pos_err_mm = np.linalg.norm(rp - tpos) * 1000
  rot_err_deg = np.degrees(quat_geodesic(rq, tquat))
  assembly[f["name"]] = {j: float(v) for j, v in zip(f["loop"], x, strict=False)}
  print(
    f"{f['name']}: loop angles (rad) = "
    + ", ".join(f"{j}={v:.4f}" for j, v in zip(f["loop"], x, strict=False)),
    flush=True,
  )
  print(
    f"   residual: pos_err={pos_err_mm:.3f} mm   rot_err={rot_err_deg:.3f} deg",
    flush=True,
  )

print("\nassembly dict:")
print(json.dumps(assembly, indent=2))
