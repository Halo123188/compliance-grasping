"""What joint angles make the CUBE bear on the tapered fingertip, not the pad?

  uv run python scripts/diag_tip_grip.py

The pad site was deliberately put on the taper at link z = -44. At the jaw angles
the task currently uses, that point sits 7.2 mm clear of the cube on each side --
because the taper is 8.6 mm behind the flat gripping face, so the face touches
first and stops the finger. Moving the site is therefore not the fix; closing
FURTHER is, which is a joint-angle question, not a geometry one.

Sweeps the distal curl (which is what swings the tip inward) against the
proximal, closes onto a real 50 mm cube each time, and reports where along the
finger the contact actually lands.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  _GRASP_SITE_POS,
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

CUBE = 0.050
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}
SITE_Z = -44.0  # where the pad site now is
PAD_LO = -31.5  # bottom edge of the flat gripping face

spec = mujoco.MjSpec.from_file(str(HAND_XML))
b = spec.worldbody.add_body(name="cube", pos=list(_GRASP_SITE_POS))
g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3)
g.name = "cube_geom"
for a, b_ in (
  ("base_link", "left_1"),
  ("base_link", "right_1"),
  ("left_1", "left_2"),
  ("right_1", "right_2"),
):
  e = spec.add_exclude()
  e.bodyname1, e.bodyname2 = a, b_

model = spec.compile()
for n in HOVER:
  a = model.actuator(n)
  model.actuator_gainprm[a.id, 0] = FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 1] = -FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 2] = -FINGER_DAMPING
  model.actuator_forcerange[a.id] = (-FINGER_EFFORT_LIMIT, FINGER_EFFORT_LIMIT)
cube_gid = model.geom("cube_geom").id
data = mujoco.MjData(model)


def close_on_cube(p_cmd: float, d_cmd: float):
  mujoco.mj_resetData(model, data)
  for n, v in HOVER.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  for n, v in (
    ("left_1", p_cmd),
    ("left_2", d_cmd),
    ("right_1", -p_cmd),
    ("right_2", -d_cmd),
  ):
    data.ctrl[model.actuator(n).id] = v
  for _ in range(4000):
    mujoco.mj_step(model, data)

  zs, fn = [], 0.0
  for i in range(data.ncon):
    c = data.contact[i]
    gs = {c.geom1, c.geom2}
    if cube_gid not in gs:
      continue
    other = (gs - {cube_gid}).pop()
    bid = model.geom_bodyid[other]
    if model.body(bid).name not in ("left_2", "right_2"):
      continue
    zs.append((data.xmat[bid].reshape(3, 3).T @ (c.pos - data.xpos[bid]))[2] * 1000)
    f = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f)
    fn += abs(f[0])
  q = [data.qpos[model.joint(n).qposadr[0]] for n in ("left_1", "left_2")]
  return q, zs, fn


print(f"pad site is at link z = {SITE_Z:+.1f} mm; flat face ends at {PAD_LO:+.1f} mm")
print("contact z below that line means the cube is bearing on the TAPER.\n")
print(
  f"{'cmd prox':>9} {'cmd dist':>9} {'settled p/d':>16} {'contact z (mm)':>18} {'Fn (N)':>8}"
)
for p_cmd in (0.10, 0.00, -0.10, -0.20):
  for d_cmd in (-0.40, -0.60, -0.80):
    q, zs, fn = close_on_cube(p_cmd, d_cmd)
    if not zs:
      print(
        f"{p_cmd:+9.2f} {d_cmd:+9.2f} {q[0]:+7.3f}/{q[1]:+7.3f}   {'(no contact)':>18}"
      )
      continue
    lo, hi = min(zs), max(zs)
    tag = "  <- ON THE TAPER" if hi < PAD_LO else ""
    print(
      f"{p_cmd:+9.2f} {d_cmd:+9.2f} {q[0]:+7.3f}/{q[1]:+7.3f}   "
      f"{lo:+7.1f}..{hi:+7.1f}  {fn:7.2f}{tag}"
    )
