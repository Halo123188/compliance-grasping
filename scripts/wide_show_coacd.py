"""The three collision representations of the gripping finger, side by side.

  uv run python scripts/wide_show_coacd.py OUT_DIR

Turntable video of the distal link (``right_2`` -- the one that actually touches
the cube) in four representations at once:

  visual mesh    the true CAD surface, which is what the renderer and the depth
                 camera draw and what the real finger is
  fitted boxes   what the physics uses today: a 4-band axis-aligned staircase
                 (`fit_collision_boxes`)
  coacd_t0.2     3 hulls, from assets/hands/wide/collision/
  coacd_t0.4     2 hulls, the coarsest of the three shipped sets

Each COACD hull gets its own colour, so the decomposition is visible as a
decomposition rather than as one blob. Nothing here is wired into the sim -- the
task still runs on the boxes; this only makes the alternative inspectable.

The hulls are stored in each PART'S OWN FRAME, matching the STL they came from
(see the collision README), so all four bodies can be placed on a common origin
and differ only by the display offset applied here.
"""

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.twofinger_wide.hand_urdf import (
  MESH_DIR,
  fit_collision_boxes,
  load_hand_cad,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "videos/coacd")
OUT.mkdir(parents=True, exist_ok=True)
COL = Path(
  "src/mjlab/asset_zoo/robots/twofinger_wide/assets/hands/wide/collision"
).resolve()
LINK = "right_2"
W, H, FRAMES = 1280, 560, 180
SPACING = 0.075  # metres between the four copies

# Distinct per-hull colours so a 3-hull decomposition reads as three pieces.
PALETTE = [
  (0.91, 0.30, 0.24, 1.0),
  (0.20, 0.60, 0.86, 1.0),
  (0.95, 0.77, 0.06, 1.0),
  (0.18, 0.80, 0.44, 1.0),
  (0.61, 0.35, 0.71, 1.0),
  (0.90, 0.49, 0.13, 1.0),
]

cad = load_hand_cad()
stl = (MESH_DIR / f"{cad.links[LINK].mesh}.stl").resolve()

spec = mujoco.MjSpec()
spec.compiler.meshdir = ""
# The offscreen framebuffer defaults to 640x480 and the Renderer refuses any
# size above it, so this has to be set on the MODEL before compiling -- not on
# the renderer.
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H
spec.add_mesh(name="visual", file=str(stl))


def _slot(idx: int) -> mujoco.MjsBody:
  b = spec.worldbody.add_body(name=f"slot{idx}")
  b.pos = [0.0, (idx - 1.5) * SPACING, 0.0]
  return b


# 0: the true surface.
g = _slot(0).add_geom(
  name="visual_g", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visual"
)
g.rgba = (0.25, 0.42, 0.75, 1.0)
g.contype, g.conaffinity = 0, 0

# 1: the fitted boxes the task runs on today.
body = _slot(1)
for i, (half, pos) in enumerate(fit_collision_boxes(cad)[LINK]):
  gb = body.add_geom(name=f"box{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos)
  gb.rgba = PALETTE[i % len(PALETTE)]
  gb.contype, gb.conaffinity = 0, 0

# 2, 3: the shipped decompositions.
for slot, tag in ((2, "coacd_t0.2"), (3, "coacd_t0.4")):
  man = json.load(open(COL / tag / "manifest.json"))[LINK]
  body = _slot(slot)
  for i, rel in enumerate(man["files"]):
    name = f"{tag}_{i}".replace(".", "_")
    spec.add_mesh(name=name, file=str(COL / tag / rel))
    gh = body.add_geom(name=f"g_{name}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
    gh.rgba = PALETTE[i % len(PALETTE)]
    gh.contype, gh.conaffinity = 0, 0
  print(f"{tag}: {man['n_hulls']} hulls")

m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

cam = mujoco.MjvCamera()
cam.lookat[:] = (0.0, 0.0, -0.024)  # the link spans z -55..+8 mm
cam.distance, cam.elevation = 0.21, -12.0

LABELS = [
  "visual mesh (the real finger)",
  "fitted boxes  x4  (used today)",
  "coacd_t0.2  x3",
  "coacd_t0.4  x2",
]


def _label(img):
  """Name each column, because four grey-on-black shapes are unreadable alone.

  The bodies are laid out along +y and the camera spins, so which column is
  which SWAPS every half turn -- the labels are therefore drawn in the order
  the projection actually puts them on screen, not in slot order.
  """
  pil = Image.fromarray(img)
  draw = ImageDraw.Draw(pil)
  try:
    font = ImageFont.truetype(
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17
    )
  except OSError:
    font = ImageFont.load_default()
  order = sorted(range(4), key=lambda i: _screen_x(i))
  for rank, slot in enumerate(order):
    x = W * (rank + 0.5) / 4.0
    t = LABELS[slot]
    w = draw.textbbox((0, 0), t, font=font)[2]
    draw.text((x - w / 2, H - 30), t, fill=(235, 235, 235), font=font)
  return np.asarray(pil)


def _screen_x(slot: int) -> float:
  """Horizontal screen position of a slot, for the current camera azimuth."""
  y = (slot - 1.5) * SPACING
  return -y * np.cos(np.radians(cam.azimuth))


renderer = mujoco.Renderer(m, height=H, width=W)
opt = mujoco.MjvOption()
frames = []
for k in range(FRAMES):
  cam.azimuth = 360.0 * k / FRAMES
  renderer.update_scene(d, camera=cam, scene_option=opt)
  frames.append(_label(renderer.render()))

path = OUT / "right_2_collision_representations.mp4"
imageio.mimwrite(path, frames, fps=30, quality=8)
print(f"wrote {path}  ({FRAMES} frames)")
print("left to right: visual mesh | fitted boxes (4) | coacd_t0.2 (3) | coacd_t0.4 (2)")
