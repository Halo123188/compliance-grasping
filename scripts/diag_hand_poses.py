"""Render the hand in three poses side by side, square-on to the jaw.

  uv run python scripts/diag_hand_poses.py OUT.png

The joint numbers alone do not show whether a pose is a grasp or nonsense. This
puts the pose the trained policy settles in next to the pose the pull-out test
found (proximal closed + distal curled INWARD, measured at 5.2-6.1 N holding
force on this cube), with a 50 mm cube drawn at the jaw so the comparison is to
scale.

The camera looks along the hinge axis (the hand's local Y). Any other angle
foreshortens one finger into the other and the jaw opening cannot be read.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/hand_poses.png"
CUBE = 0.050
W, H = 640, 720

POSES = (
  (
    "OLD hover: distals straight",
    {"left_1": 0.70, "left_2": 0.0, "right_1": -0.70, "right_2": 0.0},
  ),
  (
    "NEW hover: pre-curled",
    {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40},
  ),
  (
    "the pull-out-test grasp",
    {"left_1": 0.10, "left_2": -0.40, "right_1": -0.10, "right_2": 0.40},
  ),
  (
    "what run 45530 learned",
    {"left_1": -0.745, "left_2": 1.054, "right_1": 0.080, "right_2": -0.471},
  ),
)

spec = mujoco.MjSpec.from_file(str(HAND_XML))
# The hand XML has no <visual> block, so the offscreen framebuffer defaults to
# 640x480 and anything larger fails to render.
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H

# The cube sits where a real grasp would hold it: the pad midpoint of the third
# pose. It is drawn in every panel at that same fixed spot, so the panels are
# directly comparable -- the hand moves, the target does not.
probe = spec.compile()
pd = mujoco.MjData(probe)
for n, v in POSES[2][1].items():
  pd.qpos[probe.joint(n).qposadr[0]] = v
mujoco.mj_forward(probe, pd)
centre = (
  pd.site_xpos[probe.site("left_pad").id] + pd.site_xpos[probe.site("right_pad").id]
) / 2

body = spec.worldbody.add_body(name="cube_ghost", pos=centre.tolist())
body.add_geom(
  type=mujoco.mjtGeom.mjGEOM_BOX,
  size=[CUBE / 2] * 3,
  rgba=[0.95, 0.95, 0.95, 0.45],
  contype=0,
  conaffinity=0,
)

model = spec.compile()
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.lookat[:] = centre
cam.distance = 0.26
cam.elevation = 0.0
cam.azimuth = 90.0  # look along the hinge axis, so the jaw opens left-to-right

opt = mujoco.MjvOption()
opt.geomgroup[1] = 1  # visual meshes
opt.geomgroup[3] = 0  # hide the collision meshes; they clutter the silhouette

panels = []
for label, q in POSES:
  data.qpos[:] = 0.0
  for n, v in q.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  mujoco.mj_forward(model, data)
  renderer.update_scene(data, camera=cam, scene_option=opt)
  panels.append(renderer.render())

  gap = np.linalg.norm(
    data.site_xpos[model.site("left_pad").id]
    - data.site_xpos[model.site("right_pad").id]
  )
  print(f"{label:26s} jaw gap = {gap * 1000:6.1f} mm   (cube is {CUBE * 1000:.0f} mm)")

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 22)
except OSError:
  font = ImageFont.load_default()
for i, (label, q) in enumerate(POSES):
  draw.text((i * W + 14, 14), label, fill=(255, 255, 255), font=font)
  draw.text(
    (i * W + 14, 44),
    f"L1={q['left_1']:+.2f} L2={q['left_2']:+.2f} "
    f"R1={q['right_1']:+.2f} R2={q['right_2']:+.2f}",
    fill=(190, 190, 190),
    font=font,
  )
sheet.save(OUT)
print(f"wrote {OUT}")
