"""What does the finger actually look like, and which way does each motor move it?

  uv run python scripts/diag_finger_shape.py [OUT.png]

Every geometric claim I made about "the pad" was derived from convex-hull facets
rather than from looking at the finger, so this starts over from the two things
that are not open to interpretation:

  - a big, face-on render of the hand with the tuned pad sites drawn on it, at
    several (proximal, distal) poses, so the shape is visible;
  - the measured world position of each pad site as a function of each joint, so
    the sign convention comes from the model rather than from my guess.

No cube, no contact, no theory -- just kinematics and pictures.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/finger_shape.png"
W, H = 620, 720
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}

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
model = spec.compile()
data = mujoco.MjData(model)

sidL = model.site("left_pad").id
sidR = model.site("right_pad").id


def pose(p_l, d_l, p_r=None, d_r=None):
  data.qpos[:] = 0
  p_r = -p_l if p_r is None else p_r
  d_r = -d_l if d_r is None else d_r
  for n, v in (("left_1", p_l), ("left_2", d_l), ("right_1", p_r), ("right_2", d_r)):
    data.qpos[model.joint(n).qposadr[0]] = v
  mujoco.mj_forward(model, data)
  return data.site_xpos[sidL].copy(), data.site_xpos[sidR].copy()


# --- which way does each motor move the tuned pad site? ----------------------
print("Pad-site world position vs each joint, other joint held at the hover value.")
print("The jaw closes along x; the two sites must end up 50 mm apart to pinch the")
print("cube, one on each face.\n")
for joint, other, othval in (
  ("left_1", "left_2", HOVER["left_2"]),
  ("left_2", "left_1", HOVER["left_1"]),
):
  print(f"--- sweeping {joint} (with {other} = {othval:+.2f})")
  print(
    f"{joint:>9} {'site x':>9} {'site y':>9} {'site z':>9} {'L-R sep':>9}  direction"
  )
  prev = None
  for v in np.arange(-1.6, 1.61, 0.20):
    if joint == "left_1":
      sL, sR = pose(v, othval)
    else:
      sL, sR = pose(othval, v)
    sep = abs(sR[0] - sL[0]) * 1000
    arrow = ""
    if prev is not None:
      arrow = "closing" if sep < prev else "opening"
    prev = sep
    print(
      f"{v:+9.2f} {sL[0] * 1000:+9.1f} {sL[1] * 1000:+9.1f} {sL[2] * 1000:+9.1f}"
      f" {sep:9.1f}  {arrow}"
    )
  print()

# --- render the shape --------------------------------------------------------
CASES = (
  ("hover (current start)", HOVER["left_1"], HOVER["left_2"]),
  ("proximal OUT, distal IN", 1.00, -1.00),
  ("proximal OUT more", 1.40, -1.20),
  ("both straight (zero)", 0.0, 0.0),
)
VIEWS = (("face-on", 90.0, -8.0), ("from below", 90.0, 55.0))

grid = []
for vname, az, el in VIEWS:
  rowimgs = []
  for label, p, d in CASES:
    pose(p, d)
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 1  # visual meshes
    opt.geomgroup[3] = 0  # hide collision hulls
    opt.sitegroup[:] = 1
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.008, -0.085]
    cam.distance, cam.azimuth, cam.elevation = 0.24, az, el
    renderer.update_scene(data, camera=cam, scene_option=opt)
    img = renderer.render()
    renderer.close()
    sL, sR = pose(p, d)
    rowimgs.append((img, vname, label, p, d, abs(sR[0] - sL[0]) * 1000))
  grid.append(rowimgs)

sheet = Image.fromarray(np.vstack([np.hstack([c[0] for c in row]) for row in grid]))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 22)
  small = ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
  font = small = ImageFont.load_default()
for r, row in enumerate(grid):
  for c, (_, vname, label, p, d, sep) in enumerate(row):
    x, y = c * W + 12, r * H + 10
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    draw.text((x, y + 28), f"[{vname}]", fill=(150, 200, 255), font=small)
    draw.text(
      (x, y + 52), f"prox {p:+.2f}  dist {d:+.2f}", fill=(205, 205, 205), font=small
    )
    col = (120, 255, 140) if abs(sep - 50) < 8 else (255, 175, 115)
    draw.text((x, y + 76), f"pad sites {sep:.1f} mm apart", fill=col, font=small)
sheet.save(OUT)
print(f"wrote {OUT}")
