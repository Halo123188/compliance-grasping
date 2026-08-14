"""What the claw has to do to grip a 15 mm cube, next to what it does today.

  uv run python scripts/wide_show_small_cube.py OUT_DIR

Two side-by-side side views of the same hand:

  left    the 50 mm cube it grasps today, fingers at the settled grip angle
  right   a 15 mm cube, fingers closed as far as they would have to go

Both panels draw the WORK SURFACE as a slab, placed from the geometry rather
than assumed, so the thing to look at is where the fingertips end up relative to
it. That is the whole question: the claw's inner gripping face spans 12.5-27.5 mm
ABOVE the fingertip (measured off the mesh), so contacting a small cube on its
side means the tip has to go somewhere, and on this bench that somewhere is
inside the foam.

The geometry, all measured rather than assumed:

  sep(q) = 50.1 + 134.0*q mm   gripping-face separation vs proximal angle
  50 mm cube -> q = -0.0023    (the settled grip, wide_calibrate section 2)
  15 mm cube -> q = -0.262     which is well past the -0.0754 the finger range
                               currently stops at, i.e. outside the action space

Placement rule, per panel: the cube's centre sits at the height of the lowest
point of the gripping face, and the surface sits a cube half-edge below that.
For 50 mm that puts the tip near the surface; for 15 mm it puts the tip under it.
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import TWOFINGER_ARM_HOME
from mjlab.asset_zoo.robots.twofinger_wide.hand_urdf import (
  MESH_DIR,
  load_hand_cad,
  pad_marker_pos,
  stl_vertices,
  tip_marker_pos,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "videos/smallcube")
OUT.mkdir(parents=True, exist_ok=True)
W, H = 820, 760

# sep(q) = 50.1 + 134.0 q  (mm), inverted per cube.
# HALF-edges. Writing the full edge here made the "50 mm" panel a 100 mm cube.
CASES = [
  (0.025, -0.0023, "50 mm cube  (today)"),
  (0.0075, (15.0 - 50.1) / 134.0, "15 mm cube  (q = -0.262)"),
]


def _face_span_above_tip() -> tuple[float, float]:
  """Lowest and highest point of the inner gripping face, above the fingertip."""
  cad = load_hand_cad()
  v = stl_vertices(MESH_DIR / f"{cad.links['right_2'].mesh}.stl")
  inner = v[v[:, 0] < v[:, 0].min() + 0.0015]
  tip_z = tip_marker_pos("right")[2]
  return float(inner[:, 2].min() - tip_z), float(inner[:, 2].max() - tip_z)


LOW, HIGH = _face_span_above_tip()


def render(half: float, q: float, label: str) -> np.ndarray:
  from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import twofinger_arm_spec

  spec = twofinger_arm_spec()
  spec.visual.global_.offwidth = W
  spec.visual.global_.offheight = H

  cube = spec.worldbody.add_body(name="viz_cube")
  gc = cube.add_geom(
    name="viz_cube",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half,) * 3,
    rgba=(0.88, 0.88, 0.90, 1.0),
  )
  gc.contype, gc.conaffinity, gc.group = 0, 0, 0
  slab = spec.worldbody.add_body(name="viz_surface")
  gs = slab.add_geom(
    name="viz_surface",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.045, 0.045, 0.004),
    rgba=(0.55, 0.42, 0.30, 1.0),
  )
  gs.contype, gs.conaffinity, gs.group = 0, 0, 0

  m = spec.compile()
  d = mujoco.MjData(m)
  # THE ARM'S HOVER POSE, not its zero configuration. At zero the hand points in
  # an arbitrary direction, so "down the finger" is not world-down, a horizontal
  # work surface is meaningless against it, and the render comes out tilted with
  # the slab edge-on across the frame. At the home pose the claw hangs over the
  # bench the way it does in the task and world-up is image-up.
  for name, val in (TWOFINGER_ARM_HOME.joint_pos or {}).items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
      d.qpos[m.jnt_qposadr[jid]] = val
  for name, val in (
    ("left_1_tf", q),
    ("right_1_tf", -q),
    ("left_2_tf", 0.0),
    ("right_2_tf", 0.0),
  ):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qpos[m.jnt_qposadr[jid]] = val
  mujoco.mj_forward(m, d)

  sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
  lp, rp = d.site_xpos[sid("left_pad_tf")], d.site_xpos[sid("right_pad_tf")]
  mid = 0.5 * (lp + rp)

  def _world(body: str, local) -> np.ndarray:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array(local)

  # The GRIPPING FACES, not the pad sites. The sites are at the tips and sit
  # ~17 mm outboard of the faces at the working grip -- and that offset is not
  # even constant, because the fingers rotate as they close. Reporting the site
  # separation as "face separation" made the 50 mm panel read 66.9 mm.
  lf = _world("left_2_tf", pad_marker_pos("left"))
  rf = _world("right_2_tf", pad_marker_pos("right"))
  face_sep = float(np.linalg.norm(lf - rf))

  # The fingertip in world, from the distal body frame plus the measured tip
  # offset -- there is no tip SITE on the model, only the pad site.
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_2_tf")
  tip_w = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.array(tip_marker_pos("right"))

  # Contact on the cube's side at the LOWEST point of the gripping face, and the
  # surface one half-edge below that. This is the most favourable placement --
  # any higher contact needs the tip even deeper.
  contact_z = tip_w[2] + LOW
  m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "viz_cube")] = (
    mid[0],
    mid[1],
    contact_z,
  )
  m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "viz_surface")] = (
    mid[0],
    mid[1],
    contact_z - half - 0.004,
  )
  mujoco.mj_forward(m, d)
  # Tip height ABOVE the surface: negative means the tip is inside the foam.
  # The sign was inverted here, which reported the working 50 mm grasp as
  # burying the tip 37 mm underground.
  sink = tip_w[2] - (contact_z - half)
  print(
    f"{label:28s} face sep {face_sep * 1e3:6.1f} mm"
    f"   tip vs surface {sink * 1e3:+6.1f} mm"
    f"   {'(tip INSIDE the foam)' if sink < 0 else ''}"
  )

  # Hide the arm; keep the two distal links, the two proximals, cube and slab.
  # DISTALS ONLY. Keeping the proximals as well filled the frame and pushed one
  # finger out of it; the distal pair plus the cube is the whole question here.
  keep = {
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    for n in ("left_2_tf", "right_2_tf", "viz_cube", "viz_surface")
  } - {-1}
  for i in range(m.ngeom):
    if m.geom_bodyid[i] not in keep:
      m.geom_rgba[i] = (0, 0, 0, 0)

  # A LEVEL side view. With the arm at its hover pose the interesting plane is
  # vertical, so the camera stays horizontal (elevation ~0, world-up is image-up)
  # and only its azimuth is derived -- perpendicular to the closing axis, so both
  # fingers straddle the cube across the frame instead of hiding each other.
  close_ax = rp - lp
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (mid[0], mid[1], contact_z)
  cam.azimuth = float(np.degrees(np.arctan2(close_ax[1], close_ax[0]))) + 90.0
  cam.elevation = -6.0
  cam.distance = 0.13

  r = mujoco.Renderer(m, height=H, width=W)
  r.update_scene(d, camera=cam, scene_option=mujoco.MjvOption())
  img = r.render()
  r.close()

  pil = Image.fromarray(img)
  draw = ImageDraw.Draw(pil)
  try:
    font = ImageFont.truetype(
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
    )
  except OSError:
    font = ImageFont.load_default()
  draw.text((14, 12), label, fill=(240, 240, 240), font=font)
  draw.text(
    (14, 38),
    f"tip vs surface {sink * 1e3:+.1f} mm",
    fill=(255, 170, 120) if sink < 0 else (170, 230, 170),
    font=font,
  )
  return np.asarray(pil)


print(f"inner gripping face spans {LOW * 1e3:.1f}..{HIGH * 1e3:.1f} mm above the tip\n")
panels = [render(h, q, lab) for h, q, lab in CASES]
path = OUT / "grip_50mm_vs_15mm.png"
imageio.imwrite(path, np.concatenate(panels, axis=1))
print(f"\nwrote {path}")
