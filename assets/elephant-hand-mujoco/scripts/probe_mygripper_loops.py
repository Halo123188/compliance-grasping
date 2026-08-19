"""Probe the four-bar loop-closure geometry of mygripper_H100_R.
For each duplicate-link pair, report the world gap between body origins at the
rest pose, and the gap minimized over a coarse sweep of that finger's two driven
joints (-> the approximate assembly config where the loop physically closes)."""

import itertools
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "src/my_mjlab_project/assets/robots/mygripper_H100_R/robot.urdf"

# (coupler-on-input-side, coupler-on-distal-side, driven joints feeding this finger)
FINGERS = [
  ("s_link2_step", "s_link2_step_2", ["Revolute 1", "nRevolute 5"]),
  ("finger3_step", "finger3_step_2", ["Revolute 2", "nRevolute 8"]),
  ("finger4_step", "finger4_step_2", ["Revolute 3", "nRevolute 11"]),
]

spec = mujoco.MjSpec.from_file(str(HAND_URDF))
m = spec.compile()
d = mujoco.MjData(m)


def bid(n):
  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def jadr(n):
  j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
  return m.jnt_qposadr[j]


mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
print("=== gap at rest (qpos0) ===")
for a, b, _ in FINGERS:
  pa, pb = d.xpos[bid(a)].copy(), d.xpos[bid(b)].copy()
  print(
    f"{a:16s} {np.round(pa, 4)}  <->  {b:16s} {np.round(pb, 4)}  gap={np.linalg.norm(pa - pb) * 1000:.1f} mm"
  )

print("\n=== min gap over coarse driven-joint sweep (approx assembly config) ===")
grid = np.linspace(-np.pi, np.pi, 25)
for a, b, drv in FINGERS:
  best = (1e9, None)
  for combo in itertools.product(grid, repeat=len(drv)):
    mujoco.mj_resetData(m, d)
    for jn, v in zip(drv, combo, strict=False):
      d.qpos[jadr(jn)] = v
    mujoco.mj_forward(m, d)
    g = np.linalg.norm(d.xpos[bid(a)] - d.xpos[bid(b)])
    if g < best[0]:
      best = (g, combo)
  print(f"{a} <-> {b}: min gap={best[0] * 1000:.1f} mm at {drv}={np.round(best[1], 3)}")
