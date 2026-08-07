"""Sweep hand HEIGHT as well as joint angle, and render where the cube bears.

  uv run python scripts/diag_tip_grip_render.py OUT.png

An earlier sweep (diag_tip_grip.py) varied only the joint angles, at one hand
height, and concluded the cube can never bear on the tapered fingertip. That
conclusion was wrong because height was held fixed: the flat pad's lower edge is
the most protruding point of the finger, so at a given height it always touches
first -- but raising the hand slides the whole finger up past the cube, and once
the pad edge clears the cube's top face only the taper is left alongside it.

Each panel closes onto a real 50 mm cube from the pre-curled hover and is drawn
at the settled pose, so what is shown is what the physics actually produced.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  _GRASP_SITE_POS,
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/tip_grip.png"
CUBE = 0.050
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}
PAD_LO = -31.5  # bottom edge of the flat gripping face, in link mm
W, H = 620, 700

# (label, hand raise in mm, proximal cmd, distal cmd). Raising the hand is
# modelled by lowering the cube, which is the same relative motion and keeps the
# camera framing identical between panels.
CASES = (
  ("d=-1.20", 0.0, -0.10, -1.20),
  ("d=-1.30", 0.0, -0.13, -1.30),
  ("d=-1.40", 0.0, -0.17, -1.40),
  ("d=-1.50", 0.0, -0.20, -1.50),
)

font = None
panels, rows = [], []
for label, raise_mm, p_cmd, d_cmd in CASES:
  spec = mujoco.MjSpec.from_file(str(HAND_XML))
  spec.visual.global_.offwidth = W
  spec.visual.global_.offheight = H
  for pos, dirv in (
    ((0.0, 0.0, -0.30), (0.0, 0.0, 1.0)),
    ((0.25, -0.20, 0.10), (-0.6, 0.6, -0.5)),
  ):
    lt = spec.worldbody.add_light()
    lt.pos, lt.dir = list(pos), list(dirv)
    lt.diffuse = [0.5, 0.5, 0.5]

  cpos = list(_GRASP_SITE_POS)
  cpos[2] -= raise_mm / 1000.0  # cube down == hand up
  b = spec.worldbody.add_body(name="cube", pos=cpos)
  g = b.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3, rgba=[0.9, 0.9, 0.9, 0.55]
  )
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
  data = mujoco.MjData(model)
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

  cube_gid = model.geom("cube_geom").id
  zs, per = [], {"left_2": 0.0, "right_2": 0.0}
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
    # Normal force on THIS finger. Summing both fingers double-counts: an
    # antipodal pinch applies equal and opposite forces, so the grip force is
    # one side's value, not the pair's total.
    per[model.body(bid).name] += abs(f[0])
  q = [data.qpos[model.joint(n).qposadr[0]] for n in ("left_1", "left_2")]
  where = "none" if not zs else f"{min(zs):+.1f}..{max(zs):+.1f}"
  ontap = bool(zs) and max(zs) < PAD_LO
  fn = max(per.values())
  rows.append((label, raise_mm, p_cmd, d_cmd, q, where, fn, ontap, per))

  renderer = mujoco.Renderer(model, height=H, width=W)
  opt = mujoco.MjvOption()
  opt.geomgroup[1] = 1
  opt.geomgroup[3] = 0
  opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
  model.vis.scale.contactwidth = 0.012
  model.vis.scale.contactheight = 0.012
  model.vis.scale.forcewidth = 0.006
  model.vis.map.force = 0.004
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = cpos
  cam.distance, cam.azimuth, cam.elevation = 0.20, 90.0, -14.0
  renderer.update_scene(data, camera=cam, scene_option=opt)
  panels.append(renderer.render())
  renderer.close()

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 20)
  small = ImageFont.truetype("DejaVuSans.ttf", 17)
except OSError:
  font = small = ImageFont.load_default()
for i, (label, _raise_mm, p, d, q, where, _fn, ontap, per) in enumerate(rows):
  draw.text((i * W + 12, 10), label, fill=(255, 255, 255), font=font)
  draw.text(
    (i * W + 12, 36), f"cmd p={p:+.2f} d={d:+.2f}", fill=(200, 200, 200), font=small
  )
  draw.text(
    (i * W + 12, 58),
    f"settled {q[0]:+.3f}/{q[1]:+.3f}",
    fill=(200, 200, 200),
    font=small,
  )
  col = (120, 255, 140) if ontap else (255, 190, 120)
  draw.text((i * W + 12, 80), f"contact z {where} mm", fill=col, font=small)
  draw.text(
    (i * W + 12, 102),
    f"{'ON THE TAPER' if ontap else 'on the pad edge'}"
    f"  L={per['left_2']:.1f} R={per['right_2']:.1f} N",
    fill=col,
    font=small,
  )
sheet.save(OUT)

print(f"flat pad lower edge is link z = {PAD_LO:+.1f} mm; below that is the taper\n")
for label, _raise_mm, p, d, q, where, _fn, ontap, per in rows:
  print(
    f"  {label:20s} raise {raise_mm:5.1f} mm  cmd {p:+.2f}/{d:+.2f}  "
    f"settled {q[0]:+.3f}/{q[1]:+.3f}  contact z {where:>16} mm  "
    f"Fn left {per['left_2']:6.2f} N  right {per['right_2']:6.2f} N"
    f"{'   <- TAPER' if ontap else ''}"
  )
print(f"wrote {OUT}")
