"""Show where each joint limit actually puts the hand.

  uv run python scripts/diag_limits_render.py [OUT_PREFIX]

Writes two sheets:

  <prefix>_xml.png    the XML's +-1.6 on each joint, swept one joint at a time
  <prefix>_oneway.png the four corners of the one-way box used by TouchOneWay
                      (proximal outward only, distal inward only)

Each panel is labelled with the separation between the two tuned fingertip
sites, which is the number that has to reach 50 mm for the cube to be pinched.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "videos/limits"
W, H = 560, 660
HOVER_P, HOVER_D = 0.50, -0.40
CUBE = 0.050

spec = mujoco.MjSpec.from_file(str(HAND_XML))
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H
for pos, dirv in (
  ((0.0, 0.0, -0.30), (0.0, 0.0, 1.0)),
  ((0.30, -0.25, 0.05), (-0.7, 0.7, -0.3)),
  ((-0.30, -0.25, 0.05), (0.7, 0.7, -0.3)),
):
  lt = spec.worldbody.add_light()
  lt.pos, lt.dir = list(pos), list(dirv)
  lt.diffuse = [0.45, 0.45, 0.45]
# A translucent 50 mm cube parked where the pinch pose closes, purely as a ruler.
body = spec.worldbody.add_body(name="ruler", pos=[0.0, 0.0151, -0.0999])
gg = body.add_geom(
  type=mujoco.mjtGeom.mjGEOM_BOX,
  size=[CUBE / 2] * 3,
  contype=0,
  conaffinity=0,
  rgba=[0.85, 0.85, 0.9, 0.32],
)
gg.name = "ruler_geom"
model = spec.compile()
data = mujoco.MjData(model)
sL, sR = model.site("left_pad").id, model.site("right_pad").id


def pose(p, d):
  data.qpos[:] = 0
  for n, v in (("left_1", p), ("left_2", d), ("right_1", -p), ("right_2", -d)):
    data.qpos[model.joint(n).qposadr[0]] = v
  mujoco.mj_forward(model, data)
  return abs(data.site_xpos[sR][0] - data.site_xpos[sL][0]) * 1000


def render(p, d):
  pose(p, d)
  r = mujoco.Renderer(model, height=H, width=W)
  opt = mujoco.MjvOption()
  opt.geomgroup[1] = 1
  opt.geomgroup[3] = 0
  opt.sitegroup[:] = 1
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = [0.0, 0.010, -0.080]
  cam.distance, cam.azimuth, cam.elevation = 0.30, 90.0, -8.0
  r.update_scene(data, camera=cam, scene_option=opt)
  img = r.render()
  r.close()
  return img


def sheet(cases, out, note):
  panels = [(render(p, d), lab, p, d, pose(p, d)) for lab, p, d in cases]
  im = Image.fromarray(np.hstack([c[0] for c in panels]))
  dr = ImageDraw.Draw(im)
  try:
    f1 = ImageFont.truetype("DejaVuSans.ttf", 23)
    f2 = ImageFont.truetype("DejaVuSans.ttf", 18)
  except OSError:
    f1 = f2 = ImageFont.load_default()
  for i, (_, lab, p, d, sep) in enumerate(panels):
    x = i * W + 12
    dr.text((x, 10), lab, fill=(255, 255, 255), font=f1)
    dr.text((x, 40), f"prox {p:+.2f}   dist {d:+.2f}", fill=(205, 205, 205), font=f2)
    col = (120, 255, 140) if abs(sep - 50) < 5 else (255, 175, 115)
    dr.text((x, 64), f"tip sites {sep:.1f} mm apart", fill=col, font=f2)
    dr.text((x, 88), "(cube is 50 mm)", fill=(150, 150, 160), font=f2)
  dr.text((12, H - 34), note, fill=(150, 210, 255), font=f2)
  im.save(out)
  print(f"wrote {out}")
  for _, lab, p, d, sep in panels:
    print(f"  {lab:34s} prox {p:+.2f} dist {d:+.2f}  tip sep {sep:6.1f} mm")


print("=== XML limits: +-1.6 on every joint, one joint at a time ===")
sheet(
  (
    ("proximal at -1.6", -1.60, HOVER_D),
    ("proximal at +1.6", 1.60, HOVER_D),
    ("distal at -1.6", HOVER_P, -1.60),
    ("distal at +1.6", HOVER_P, 1.60),
  ),
  f"{PREFIX}_xml.png",
  "XML +-1.6, the other joint held at the hover value",
)

print("\n=== one-way box: proximal [+0.50,+1.60], distal [-1.60,-0.40] ===")
sheet(
  (
    ("corner: hover (both at limit)", 0.50, -0.40),
    ("proximal fully OUT", 1.60, -0.40),
    ("distal fully IN", 0.50, -1.60),
    ("both at far limit", 1.60, -1.60),
  ),
  f"{PREFIX}_oneway.png",
  "one-way box corners; the grasp (+0.50, -1.05) sits inside it",
)
