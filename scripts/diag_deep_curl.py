"""Does curling the fingers deeper turn point contact into line contact?

  uv run python scripts/diag_deep_curl.py [OUT.png]

Two experiments.

(1) Sweep the distal curl from the current grip depth to the joint's mechanical
    end, with a FREE cube, and count how many contact points MuJoCo generates per
    finger, where they land along the finger, and what pull-out force they hold.

(2) Hold the curl fixed and instead rotate the CUBE about the hand's hinge axis.
    The fingers rotate about y, so pad tilt lives in the x-z plane and cube pitch
    is the one knob that can bring the two flat faces parallel. This separates
    "the pad is in the wrong place" from "the pad is at the wrong angle".

Note on the solver: MuJoCo 3.8 exposes multiccd as a DISABLE bit (mjDSBL_MULTICCD),
i.e. multiple contact points for near-parallel convex faces are already ON by
default. So a single contact point here is a statement about the geometry, not
about a missing flag. Experiment (1) verifies that by running both ways.
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

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/deep_curl.png"
CUBE, MASS = 0.050, 0.05
WEIGHT = MASS * 9.81
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}
PAD_LO, PAD_HI = -31.5, -16.9  # flat gripping face span, in link mm
SLIP_MM = 5.0
W, H = 560, 660

# (proximal cmd, distal cmd). The proximal gives way as the distal curls in,
# otherwise the finger jams on the cube before the tip can wrap.
CASES = (
  (0.10, -0.40),
  (0.00, -0.60),
  (-0.10, -0.85),
  (-0.10, -1.00),
  (-0.10, -1.20),
  (-0.15, -1.35),
  (-0.20, -1.50),
)
PITCHES = np.arange(-0.35, 0.351, 0.05)  # cube rotation about the hinge axis
PITCH_AT = (0.10, -0.40)  # the depth the pull-out test uses


def build(alpha_cube: float = 1.0, multiccd: bool = True):
  spec = mujoco.MjSpec.from_file(str(HAND_XML))
  spec.visual.global_.offwidth = W
  spec.visual.global_.offheight = H
  if not multiccd:
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
  for pos, dirv in (
    ((0.0, 0.0, -0.30), (0.0, 0.0, 1.0)),
    ((0.25, -0.20, 0.10), (-0.6, 0.6, -0.5)),
  ):
    lt = spec.worldbody.add_light()
    lt.pos, lt.dir = list(pos), list(dirv)
    lt.diffuse = [0.5, 0.5, 0.5]
  b = spec.worldbody.add_body(name="cube", pos=list(_GRASP_SITE_POS))
  b.add_freejoint(name="cube_joint")
  g = b.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[CUBE / 2] * 3,
    mass=MASS,
    rgba=[0.9, 0.9, 0.9, alpha_cube],
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
  # The pull is applied explicitly; weight would be an uncontrolled offset on it.
  model.opt.gravity[:] = 0.0
  return model, mujoco.MjData(model)


def settle(model, data, p_cmd: float, d_cmd: float, pitch: float = 0.0):
  mujoco.mj_resetData(model, data)
  for n, v in HOVER.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  adr = model.jnt_qposadr[model.joint("cube_joint").id]
  data.qpos[adr : adr + 3] = _GRASP_SITE_POS
  data.qpos[adr + 3 : adr + 7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
  for n, v in (
    ("left_1", p_cmd),
    ("left_2", d_cmd),
    ("right_1", -p_cmd),
    ("right_2", -d_cmd),
  ):
    data.ctrl[model.actuator(n).id] = v
  for _ in range(4000):
    mujoco.mj_step(model, data)


def contacts(model, data):
  """Per distal link: contact z along the finger (mm), and total normal force."""
  cube_gid = model.geom("cube_geom").id
  out = {"left_2": ([], 0.0), "right_2": ([], 0.0)}
  for i in range(data.ncon):
    c = data.contact[i]
    gs = {c.geom1, c.geom2}
    if cube_gid not in gs:
      continue
    bid = model.geom_bodyid[(gs - {cube_gid}).pop()]
    name = model.body(bid).name
    if name not in out:
      continue
    z = (data.xmat[bid].reshape(3, 3).T @ (c.pos - data.xpos[bid]))[2] * 1000
    f = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f)
    zs, fn = out[name]
    out[name] = (zs + [z], fn + abs(f[0]))
  return out


def pullout(model, data):
  cube_bid = model.body("cube").id
  settled = data.xpos[cube_bid].copy()
  for _ in range(400):  # control: drift under NO pull
    mujoco.mj_step(model, data)
  drift = np.linalg.norm(data.xpos[cube_bid] - settled) * 1000
  hold = 0.0
  for pull in np.arange(0.25, 40.0, 0.25):
    data.xfrc_applied[cube_bid] = [0, 0, -pull, 0, 0, 0]
    for _ in range(400):
      mujoco.mj_step(model, data)
    if np.linalg.norm(data.xpos[cube_bid] - settled) * 1000 > SLIP_MM:
      break
    hold = pull
  data.xfrc_applied[cube_bid] = 0.0
  return hold, drift


def probe(model, data, p_cmd, d_cmd, pitch=0.0):
  settle(model, data, p_cmd, d_cmd, pitch)
  con = contacts(model, data)
  q = [data.qpos[model.joint(n).qposadr[0]] for n in ("left_1", "left_2")]
  zl, fl = con["left_2"]
  zr, fr = con["right_2"]
  if not zl and not zr:
    return q, 0, 0, "none", 0.0, 0.0, 0.0
  span = f"{min(zl):+.1f}..{max(zl):+.1f}" if zl else "none"
  hold, drift = pullout(model, data)
  return q, len(zl), len(zr), span, max(fl, fr), hold, drift


print(f"flat gripping face spans link z {PAD_LO:+.1f} .. {PAD_HI:+.1f} mm")
print(f"cube {CUBE * 1000:.0f} mm {MASS * 1000:.0f} g -> weight {WEIGHT:.2f} N")

# --- (1) deeper curl ---------------------------------------------------------
rows = []
for multiccd in (True, False):
  model, data = build(multiccd=multiccd)
  tag = "ON (MuJoCo 3.8 default)" if multiccd else "OFF (mjDSBL_MULTICCD)"
  print(f"\n=== (1) curl sweep, free cube, multiccd {tag} ===")
  print(
    f"{'cmd p/d':>13} {'settled p/d':>15} {'pts L/R':>8} {'contact z (L)':>18}"
    f" {'Fn/side':>8} {'pull-out':>9} {'x wt':>6}"
  )
  for p_cmd, d_cmd in CASES:
    q, nl, nr, span, fn, hold, drift = probe(model, data, p_cmd, d_cmd)
    if nl == 0 and nr == 0:
      print(f"{p_cmd:+6.2f}/{d_cmd:+6.2f} {q[0]:+7.3f}/{q[1]:+7.3f}   (no contact)")
    else:
      print(
        f"{p_cmd:+6.2f}/{d_cmd:+6.2f} {q[0]:+7.3f}/{q[1]:+7.3f} "
        f"{nl:3d}/{nr:<4d} {span:>18} {fn:7.1f}N {hold:8.2f}N {hold / WEIGHT:5.1f}x"
        f"   drift@0N {drift:.2f} mm"
      )
    if multiccd:
      rows.append((p_cmd, d_cmd, q, nl, nr, span, fn, hold))

# --- (2) cube tilt at fixed curl ---------------------------------------------
model, data = build()
p_cmd, d_cmd = PITCH_AT
print(f"\n=== (2) cube pitch sweep at cmd {p_cmd:+.2f}/{d_cmd:+.2f} ===")
print(
  f"{'pitch':>7} {'deg':>6} {'pts L/R':>8} {'contact z (L)':>18}"
  f" {'Fn/side':>8} {'pull-out':>9} {'x wt':>6}"
)
best = None
for pitch in PITCHES:
  q, nl, nr, span, fn, hold, drift = probe(model, data, p_cmd, d_cmd, pitch)
  print(
    f"{pitch:+7.2f} {np.degrees(pitch):+6.1f} {nl:3d}/{nr:<4d} {span:>18} "
    f"{fn:7.1f}N {hold:8.2f}N {hold / WEIGHT:5.1f}x"
  )
  if best is None or hold > best[1]:
    best = (pitch, hold, nl, nr, span)
print(
  f"\nbest pitch {best[0]:+.2f} rad ({np.degrees(best[0]):+.1f} deg): "
  f"pull-out {best[1]:.2f} N = {best[1] / WEIGHT:.1f}x weight, "
  f"pts {best[2]}/{best[3]}, span {best[4]} mm"
)

# --- render the curl panels --------------------------------------------------
RENDER = {-0.40, -0.85, -1.20, -1.50}
rmodel, rdata = build(alpha_cube=0.55)
panels, labels = [], []
for p, d, q, nl, nr, span, fn, hold in rows:
  if d not in RENDER:
    continue
  settle(rmodel, rdata, p, d)
  renderer = mujoco.Renderer(rmodel, height=H, width=W)
  opt = mujoco.MjvOption()
  opt.geomgroup[1] = 1
  opt.geomgroup[3] = 0
  opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
  rmodel.vis.scale.contactwidth = 0.012
  rmodel.vis.scale.contactheight = 0.012
  rmodel.vis.scale.forcewidth = 0.006
  rmodel.vis.map.force = 0.004
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(rmodel, cam)
  cam.lookat[:] = _GRASP_SITE_POS
  cam.distance, cam.azimuth, cam.elevation = 0.19, 90.0, -12.0
  renderer.update_scene(rdata, camera=cam, scene_option=opt)
  panels.append(renderer.render())
  renderer.close()
  labels.append((d, q, nl, nr, span, fn, hold))

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 21)
  small = ImageFont.truetype("DejaVuSans.ttf", 17)
except OSError:
  font = small = ImageFont.load_default()
for i, (d, q, nl, nr, span, fn, hold) in enumerate(labels):
  ok = nl >= 2 and nr >= 2
  col = (120, 255, 140) if ok else (255, 170, 110)
  draw.text((i * W + 12, 10), f"distal cmd {d:+.2f}", fill=(255, 255, 255), font=font)
  draw.text(
    (i * W + 12, 38),
    f"settled {q[0]:+.3f}/{q[1]:+.3f}",
    fill=(200, 200, 200),
    font=small,
  )
  draw.text((i * W + 12, 60), f"contact pts  L={nl}  R={nr}", fill=col, font=small)
  draw.text((i * W + 12, 82), f"span {span} mm", fill=col, font=small)
  draw.text(
    (i * W + 12, 104),
    f"Fn {fn:.1f} N   pull-out {hold:.2f} N ({hold / WEIGHT:.1f}x weight)",
    fill=col,
    font=small,
  )
  draw.text(
    (i * W + 12, 128),
    "LINE CONTACT" if ok else "point contact" if nl or nr else "NO CONTACT",
    fill=col,
    font=font,
  )
sheet.save(OUT)
print(f"\nwrote {OUT}")
