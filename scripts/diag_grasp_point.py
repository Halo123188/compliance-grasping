"""Render the grasp pose from the front and from below, with the pads marked.

  uv run python scripts/diag_grasp_point.py OUT.png [SITE_Z_MM ...]

The face-on view alone cannot show how far along the finger the pads sit -- the
cube hides them. Looking up from below shows where they are relative to the tip,
which is the thing being tuned.

Extra SITE_Z_MM values are drawn as ghost markers at that z, with x taken from
the finger's actual inner surface at that height (scripts/diag_finger_profile.py)
rather than held at the pad's x. Below the flat pad the finger tapers hard --
inner x is +10.0 at z=-32 but +4.6 by z=-39 -- so a marker that keeps the pad's x
while sliding down z hangs in free space in front of the finger, which is what
makes a naive "move it toward the tip" wrong.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import _GRASP_SITE_POS, HAND_XML

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/grasp_point.png"
GHOSTS = [float(a) for a in sys.argv[2:]]
CUBE = 0.050
GRIP = {"left_1": 0.10, "left_2": -0.40, "right_1": -0.10, "right_2": 0.40}
W, H = 700, 760
VIEWS = (("front (along the hinge axis)", 90.0, -12.0), ("from below", 90.0, 52.0))

spec = mujoco.MjSpec.from_file(str(HAND_XML))
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H

# Cube at grasp_site: that is where the arm puts it, so the pads' position
# relative to THIS is what the grip actually is.
body = spec.worldbody.add_body(name="cube_ghost", pos=list(_GRASP_SITE_POS))
body.add_geom(
  type=mujoco.mjtGeom.mjGEOM_BOX,
  size=[CUBE / 2] * 3,
  rgba=[0.95, 0.95, 0.95, 0.35],
  contype=0,
  conaffinity=0,
)


def inner_x(model: mujoco.MjModel, z_mm: float) -> float:
  """Inner-surface x of the left distal at this height, in mm."""
  gid = model.geom("left_2_col").id
  mid = model.geom_dataid[gid]
  v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
  v = model.mesh_vert[v0 : v0 + nv].astype(np.float64)
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, model.geom_quat[gid])
  v = (v @ R.reshape(3, 3).T + model.geom_pos[gid]) * 1000.0
  sl = v[np.abs(v[:, 2] - z_mm) < 2.0]
  return float(sl[:, 0].max()) if len(sl) else float("nan")


probe = spec.compile()
for z in GHOSTS:
  x = inner_x(probe, z)
  for side, sgn in (("left", 1.0), ("right", -1.0)):
    b = spec.body(f"{side}_2")
    s = b.add_site()
    s.name = f"{side}_ghost_{abs(int(z))}"
    s.pos = [sgn * x / 1000.0, 0.00885, z / 1000.0]
    s.size = [0.0035, 0.0, 0.0]
    s.rgba = [1.0, 0.85, 0.1, 0.95]

model = spec.compile()
data = mujoco.MjData(model)
for n, v in GRIP.items():
  data.qpos[model.joint(n).qposadr[0]] = v
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=H, width=W)
opt = mujoco.MjvOption()
opt.geomgroup[1] = 1
opt.geomgroup[3] = 0
centre = 0.5 * (
  data.site_xpos[model.site("left_pad").id] + data.site_xpos[model.site("right_pad").id]
)

panels = []
for _, az, el in VIEWS:
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = centre
  cam.distance, cam.azimuth, cam.elevation = 0.24, az, el
  renderer.update_scene(data, camera=cam, scene_option=opt)
  panels.append(renderer.render())

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 21)
except OSError:
  font = ImageFont.load_default()
sz = model.site("left_pad").pos[2] * 1000
for i, (label, _, _) in enumerate(VIEWS):
  draw.text((i * W + 14, 12), label, fill=(255, 255, 255), font=font)
  draw.text(
    (i * W + 14, 40),
    f"blue/red = pad site, z={sz:+.1f} mm"
    + (f"   yellow = {', '.join(f'{g:+.0f}' for g in GHOSTS)} mm" if GHOSTS else ""),
    fill=(200, 200, 200),
    font=font,
  )
sheet.save(OUT)

print(f"pad site z = {sz:+.2f} mm ; finger tip at z = -55.0 mm")
for z in GHOSTS:
  print(f"  ghost z={z:+6.1f} mm -> inner surface x = {inner_x(model, z):+6.2f} mm")
print(f"wrote {OUT}")
