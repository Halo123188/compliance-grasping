"""Draw labelled candidate pad-site positions on the open hand, to pick from.

  uv run python scripts/diag_pad_candidates.py OUT.png

Candidates sit at a fixed z near the fingertip and step across the finger in y,
with x taken from the finger's real inner surface at that height rather than
held at the flat pad's x -- below the pad the finger tapers, so a fixed x floats
off the front of it.

Rendered on the FULLY OPEN hand so nothing occludes the finger, from below and
from the side. The side view (looking down the closing axis) is the one that
actually shows y, which is what is being chosen.

Note what a site at this height is and is not: the flat gripping pad ends at
z = -31.5, and by z = -44 the inner surface has pulled back to x = +1.3, so the
two fingers there are ~2.7 mm apart. A 50 mm cube cannot reach that point -- it
bears on the pad above it. A site here is a fingertip reference, not the contact
point for this cube.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/pad_candidates.png"
Z = -44.0  # the height picked from the previous sheet
OPEN = {"left_1": 1.40, "left_2": 0.0, "right_1": -1.40, "right_2": 0.0}
W, H = 760, 820
VIEWS = (
  # Looking almost straight up, so the finger's cross-section faces the camera
  # and y runs across the image -- at elevation 58 the finger was edge-on and y
  # was the depth axis, which is the one thing that had to be readable.
  ("from below (looking up the finger)", 90.0, 80.0),
  ("from the side (down the closing axis)", 0.0, 0.0),
)

# label, y (mm), colour
CANDS = (
  ("A", 8.85, (0.20, 0.55, 1.00)),  # today's y, unchanged
  ("B", 12.70, (0.15, 0.85, 0.45)),  # centre of the inner surface at this z
  ("C", 16.00, (1.00, 0.85, 0.10)),
  ("D", 18.85, (1.00, 0.50, 0.05)),  # halfway from A to the far edge (+28.8)
  ("E", 22.00, (1.00, 0.25, 0.55)),
)

spec = mujoco.MjSpec.from_file(str(HAND_XML))
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H

# The hand XML ships no lights, so MuJoCo's single default headlight leaves the
# underside of the fingers almost black -- which is the side being inspected.
for pos, d in (
  ((0.0, 0.0, -0.30), (0.0, 0.0, 1.0)),  # from below, pointing up
  ((0.20, -0.20, 0.10), (-0.6, 0.6, -0.5)),
  ((-0.20, 0.25, 0.05), (0.6, -0.7, -0.3)),
):
  lt = spec.worldbody.add_light()
  lt.pos, lt.dir = list(pos), list(d)
  lt.diffuse = [0.55, 0.55, 0.55]

probe = spec.compile()
gid = probe.geom("left_2_col").id
mid = probe.geom_dataid[gid]
v0, nv = probe.mesh_vertadr[mid], probe.mesh_vertnum[mid]
verts = probe.mesh_vert[v0 : v0 + nv].astype(np.float64)
R = np.zeros(9)
mujoco.mju_quat2Mat(R, probe.geom_quat[gid])
verts = (verts @ R.reshape(3, 3).T + probe.geom_pos[gid]) * 1000.0
sl = verts[np.abs(verts[:, 2] - Z) < 1.5]
inner_x = float(sl[:, 0].max())
near = sl[sl[:, 0] > inner_x - 1.5]
y_lo, y_hi = float(near[:, 1].min()), float(near[:, 1].max())

# Candidates go on the LEFT finger only: mirroring them doubles the clutter
# without adding information, since the hand is symmetric by construction.
left = spec.body("left_2")
for label, y, rgb in CANDS:
  s = left.add_site()
  s.name = f"cand_{label}"
  s.pos = [inner_x / 1000.0, y / 1000.0, Z / 1000.0]
  s.size = [0.0022, 0.0, 0.0]
  s.rgba = [*rgb, 1.0]

model = spec.compile()
data = mujoco.MjData(model)
for n, v in OPEN.items():
  data.qpos[model.joint(n).qposadr[0]] = v
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=H, width=W)
opt = mujoco.MjvOption()
opt.geomgroup[1] = 1
opt.geomgroup[3] = 0
lookat = data.site_xpos[model.site("cand_C").id].copy()

panels = []
for _, az, el in VIEWS:
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = lookat
  cam.distance, cam.azimuth, cam.elevation = 0.085, az, el
  renderer.update_scene(data, camera=cam, scene_option=opt)
  panels.append(renderer.render())

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 22)
  small = ImageFont.truetype("DejaVuSans.ttf", 19)
except OSError:
  font = small = ImageFont.load_default()
for i, (label, _, _) in enumerate(VIEWS):
  draw.text((i * W + 14, 12), label, fill=(255, 255, 255), font=font)

y0 = H - 26 * len(CANDS) - 56
draw.text(
  (16, y0 - 30),
  f"all at z={Z:+.0f} mm, x={inner_x:+.2f} mm (on the surface)",
  fill=(230, 230, 230),
  font=small,
)
for j, (label, y, rgb) in enumerate(CANDS):
  col = tuple(int(255 * c) for c in rgb)
  draw.ellipse((16, y0 + 26 * j + 4, 30, y0 + 26 * j + 18), fill=col)
  note = ""
  if label == "A":
    note = "  (today's y)"
  elif label == "B":
    note = f"  (centre of surface {y_lo:+.1f}..{y_hi:+.1f})"
  elif label == "D":
    note = "  (halfway to the far edge)"
  draw.text(
    (40, y0 + 26 * j),
    f"{label}   y = {y:+.2f} mm{note}",
    fill=(235, 235, 235),
    font=small,
  )
sheet.save(OUT)

print(f"z = {Z:+.1f} mm   inner surface x = {inner_x:+.2f} mm")
print(f"inner surface y at this height: {y_lo:+.1f} .. {y_hi:+.1f} mm")
for label, y, _ in CANDS:
  flag = "" if y_lo <= y <= y_hi else "   <- OFF the surface at this height"
  print(f"  {label}  y = {y:+6.2f} mm{flag}")
print(f"wrote {OUT}")
