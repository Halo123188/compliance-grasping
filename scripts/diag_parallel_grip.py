"""Does a parallel-pad grip actually hold the cube, where the current one doesn't?

  uv run python scripts/diag_parallel_grip.py [OUT.png]

diag_pad_parallel.py showed the pad plane is 2.5 deg off the distal link's own -x
axis, and both finger joints turn about the SAME y axis, so the pad's tilt in the
world is just (proximal + distal) - 0.044 rad. The two pads are parallel to each
other, and square to the cube, on exactly one line:

    proximal + distal = +0.044 rad

The grip the policy currently settles into sits at proximal + distal = -0.156,
which is 0.20 rad = 11.5 deg off, so the pads meet the cube as a 23 deg wedge and
can only touch it at one point each. Squeezing a wedge harder does not add
contact points, which is exactly what the earlier force sweep found (7 -> 29 N
changed nothing).

This closes on the cube ALONG that line -- squeeze harder by walking proximal
down and distal up together -- and measures contact count and pull-out force
against the same test the wedge grip failed at 0.25 N.
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

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/parallel_grip.png"
CUBE, MASS = 0.050, 0.05
WEIGHT = MASS * 9.81
PAD_OFFSET = 0.044  # rad; proximal + distal must equal this for parallel pads
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}
SLIP_MM = 5.0
W, H = 560, 660

# Squeeze along the parallel line: proximal p, distal PAD_OFFSET - p. Lower p =
# tighter. p = -0.180 is where the gap is exactly 50 mm, so below that is preload.
PARALLEL = [(p, PAD_OFFSET - p) for p in (-0.10, -0.18, -0.26, -0.34, -0.45)]
# The wedge grips the policy currently produces, for comparison.
WEDGE = [(0.10, -0.40), (0.00, -0.60), (-0.10, -0.85)]


def build(alpha=1.0):
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
  b = spec.worldbody.add_body(name="cube", pos=list(_GRASP_SITE_POS))
  b.add_freejoint(name="cube_joint")
  g = b.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[CUBE / 2] * 3,
    mass=MASS,
    rgba=[0.9, 0.9, 0.9, alpha],
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
  model.opt.gravity[:] = 0.0
  return model, mujoco.MjData(model)


model, data = build()
cube_gid = model.geom("cube_geom").id
cube_bid = model.body("cube").id


def pad_facet(geom_name: str):
  """Centroid + normal of the inward-facing flat gripping face, in link frame."""
  gid = model.geom(geom_name).id
  mid = model.geom_dataid[gid]
  vadr, vnum = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
  fadr, fnum = model.mesh_faceadr[mid], model.mesh_facenum[mid]
  verts = model.mesh_vert[vadr : vadr + vnum].reshape(-1, 3)
  faces = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3)
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, model.geom_quat[gid])
  verts = verts @ R.reshape(3, 3).T + model.geom_pos[gid]
  groups = []
  for f in faces:
    a, bb, c = verts[f]
    n = np.cross(bb - a, c - a)
    A = np.linalg.norm(n) / 2
    if A <= 0:
      continue
    n = n / (2 * A)
    for gr in groups:
      if np.dot(gr["n"], n) > 0.999:
        gr["A"] += A
        gr["v"].extend(verts[f])
        break
    else:
      groups.append({"n": n, "A": A, "v": list(verts[f])})
  want = -1.0 if geom_name.startswith("left") else +1.0
  gr = max(
    (g for g in groups if np.dot(g["n"], [want, 0, 0]) > 0.9), key=lambda g: g["A"]
  )
  return gr["n"], np.array(gr["v"]).mean(axis=0)


_nL, _cL = pad_facet("left_2_col")
_nR, _cR = pad_facet("right_2_col")


def pad_centre(p: float, d: float):
  """World midpoint of the two pad faces at a pose, ignoring contact.

  The parallel poses put the pads somewhere quite different from where the wedge
  pose puts them, so holding the cube at _GRASP_SITE_POS would just miss the hand
  entirely. In the real task the ARM chooses where the hand goes, so the fair
  comparison places the cube where this jaw actually closes -- and the offset
  from _GRASP_SITE_POS is then the hand-placement change the arm has to make.
  """
  data.qpos[:] = 0
  for n_, v in (("left_1", p), ("left_2", d), ("right_1", -p), ("right_2", -d)):
    data.qpos[model.joint(n_).qposadr[0]] = v
  mujoco.mj_kinematics(model, data)
  pts = []
  for body, c in (("left_2", _cL), ("right_2", _cR)):
    bid = model.body(body).id
    R = data.xmat[bid].reshape(3, 3)
    pts.append(data.xpos[bid] + R @ c)
  return 0.5 * (pts[0] + pts[1])


def settle(p_cmd, d_cmd, cube_pos=None):
  mujoco.mj_resetData(model, data)
  for n, v in HOVER.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  adr = model.jnt_qposadr[model.joint("cube_joint").id]
  data.qpos[adr : adr + 3] = _GRASP_SITE_POS if cube_pos is None else cube_pos
  data.qpos[adr + 3] = 1.0
  for n, v in (
    ("left_1", p_cmd),
    ("left_2", d_cmd),
    ("right_1", -p_cmd),
    ("right_2", -d_cmd),
  ):
    data.ctrl[model.actuator(n).id] = v
  for _ in range(4000):
    mujoco.mj_step(model, data)


def report():
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


def pullout(warm=True):
  settled = data.xpos[cube_bid].copy()
  _ = warm
  for _ in range(400):
    mujoco.mj_step(model, data)
  drift = np.linalg.norm(data.xpos[cube_bid] - settled) * 1000
  hold = 0.0
  for pull in np.arange(0.25, 60.0, 0.25):
    data.xfrc_applied[cube_bid] = [0, 0, -pull, 0, 0, 0]
    for _ in range(400):
      mujoco.mj_step(model, data)
    if np.linalg.norm(data.xpos[cube_bid] - settled) * 1000 > SLIP_MM:
      break
    hold = pull
  data.xfrc_applied[cube_bid] = 0.0
  return hold, drift


print(f"cube {CUBE * 1000:.0f} mm {MASS * 1000:.0f} g -> weight {WEIGHT:.2f} N")
print(f"pads are parallel when proximal + distal = {PAD_OFFSET:+.3f} rad\n")
print(
  f"{'cmd p/d':>14} {'settled p/d':>16} {'sum':>7} {'tilt':>6} {'pts L/R':>8}"
  f" {'span L (mm)':>16} {'Fn/side':>8} {'pull-out':>9} {'x wt':>6}"
)

# Where the parallel jaw closes, and how far that is from where the wedge grip
# holds the cube today. The arm has to make up this offset.
PAR_CENTRE = pad_centre(-0.180, PAD_OFFSET + 0.180)
off = (PAR_CENTRE - np.asarray(_GRASP_SITE_POS)) * 1000
print(f"parallel jaw closes at {PAR_CENTRE.round(4)}, i.e. {off.round(1)} mm")
print("from the current grasp site -- the arm must move the hand by that much\n")

rows = []
for tag, cases in (("wedge (current)", WEDGE), ("parallel", PARALLEL)):
  print(f"--- {tag}")
  for p_cmd, d_cmd in cases:
    settle(p_cmd, d_cmd, None if tag.startswith("wedge") else PAR_CENTRE)
    con = report()
    q = [data.qpos[model.joint(n).qposadr[0]] for n in ("left_1", "left_2")]
    zl, fl = con["left_2"]
    zr, fr = con["right_2"]
    s = q[0] + q[1]
    tilt = np.degrees(abs(s - PAD_OFFSET))
    if not zl and not zr:
      print(
        f"{p_cmd:+6.2f}/{d_cmd:+6.2f} {q[0]:+7.3f}/{q[1]:+7.3f} {s:+7.3f}"
        f" {tilt:5.1f}   (no contact)"
      )
      continue
    span = f"{min(zl):+.1f}..{max(zl):+.1f}" if zl else "none"
    hold, drift = pullout()
    fn = max(fl, fr)
    print(
      f"{p_cmd:+6.2f}/{d_cmd:+6.2f} {q[0]:+7.3f}/{q[1]:+7.3f} {s:+7.3f} {tilt:5.1f}"
      f" {len(zl):3d}/{len(zr):<4d} {span:>16} {fn:7.1f}N {hold:8.2f}N"
      f" {hold / WEIGHT:5.1f}x   drift@0N {drift:.2f} mm"
    )
    rows.append((tag, p_cmd, d_cmd, q, s, tilt, len(zl), len(zr), span, fn, hold))

# --- render one wedge grip beside one parallel grip ---------------------------
pick = [r for r in rows if r[0] == "wedge (current)"][:1]
par = sorted([r for r in rows if r[0] == "parallel"], key=lambda r: -r[10])[:1]
model, data = build(alpha=0.55)
cube_gid = model.geom("cube_geom").id
cube_bid = model.body("cube").id
panels, labels = [], []
for tag, p_cmd, d_cmd, q, _s, tilt, nl, nr, span, _fn, hold in pick + par:
  settle(p_cmd, d_cmd)
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
  cam.lookat[:] = _GRASP_SITE_POS
  cam.distance, cam.azimuth, cam.elevation = 0.19, 90.0, -12.0
  renderer.update_scene(data, camera=cam, scene_option=opt)
  panels.append(renderer.render())
  renderer.close()
  labels.append((tag, q, tilt, nl, nr, span, fn, hold))

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 22)
  small = ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
  font = small = ImageFont.load_default()
for i, (tag, q, tilt, nl, nr, span, fn, hold) in enumerate(labels):
  ok = hold > WEIGHT
  col = (120, 255, 140) if ok else (255, 150, 110)
  draw.text((i * W + 12, 10), tag, fill=(255, 255, 255), font=font)
  draw.text(
    (i * W + 12, 40),
    f"settled {q[0]:+.3f}/{q[1]:+.3f}",
    fill=(205, 205, 205),
    font=small,
  )
  draw.text((i * W + 12, 64), f"pad tilt {tilt:.1f} deg", fill=col, font=small)
  draw.text((i * W + 12, 88), f"contact pts L={nl} R={nr}", fill=col, font=small)
  draw.text((i * W + 12, 112), f"span {span} mm", fill=col, font=small)
  draw.text((i * W + 12, 136), f"Fn {fn:.1f} N", fill=col, font=small)
  draw.text(
    (i * W + 12, 162),
    f"pull-out {hold:.2f} N = {hold / WEIGHT:.1f}x weight",
    fill=col,
    font=font,
  )
sheet.save(OUT)
print(f"\nwrote {OUT}")
