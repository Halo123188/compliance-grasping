"""Is there ANY pose where the flat pad lies parallel on the cube face?

  uv run python scripts/diag_pad_normal.py

diag_deep_curl.py found the contact point pinned at link z = -31.3 mm and the
normal force pinned at 7.0 N across +-20 deg of cube rotation. A face-on-face
contact cannot be invariant to rotating one of the faces, so the cube must be
bearing on a sharp convex corner -- and z = -31.3 is exactly the boundary between
the flat gripping face (-31.5..-16.9) and the taper below it.

Two ways to check that, both here:

(a) Geometric: pull the left_2 convex hull, fit the flat gripping face, and
    report the angle between its outward normal and the cube's face normal at
    each settled curl. If that angle is large, no cube rotation in the +-20 deg
    range swept earlier could ever have reached parallel.

(b) Empirical: sweep distal curl against cube pitch over a MUCH wider range and
    report the contact count over the whole grid. If a parallel pose exists it
    shows up as a cell with more than one contact point per finger.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  _GRASP_SITE_POS,
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

CUBE, MASS = 0.050, 0.05
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}
PAD_LO, PAD_HI = -0.0315, -0.0169  # flat gripping face span in link metres


def build():
  spec = mujoco.MjSpec.from_file(str(HAND_XML))
  b = spec.worldbody.add_body(name="cube", pos=list(_GRASP_SITE_POS))
  b.add_freejoint(name="cube_joint")
  g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3, mass=MASS)
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

# --- (a) the pad plane, straight off the collision hull ----------------------
# The collision geom is the mesh's convex hull; MuJoCo stores the hull faces, so
# the gripping face is whichever hull facet has its centroid inside the pad's z
# span and faces inward (+x, toward the other finger).
gid = model.geom("left_2_col").id
mid = model.geom_dataid[gid]
vadr, vnum = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
fadr, fnum = model.mesh_faceadr[mid], model.mesh_facenum[mid]
verts = model.mesh_vert[vadr : vadr + vnum].reshape(-1, 3)
faces = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3)
# Mesh verts are in the mesh frame; the geom carries the compiled offset.
gpos, gquat = model.geom_pos[gid], model.geom_quat[gid]
R = np.zeros(9)
mujoco.mju_quat2Mat(R, gquat)
R = R.reshape(3, 3)
verts_link = verts @ R.T + gpos

areas, normals, cents = [], [], []
for f in faces:
  a, b, c = verts_link[f]
  n = np.cross(b - a, c - a)
  A = np.linalg.norm(n) / 2
  if A <= 0:
    continue
  areas.append(A)
  normals.append(n / np.linalg.norm(n))
  cents.append((a + b + c) / 3)
areas, normals, cents = np.array(areas), np.array(normals), np.array(cents)

# Group coplanar facets: the hull triangulates the flat pad into many triangles
# sharing one normal. Cluster by normal direction and sum area.
groups = []
for n, A, ct, f in zip(normals, areas, cents, faces, strict=False):
  for g_ in groups:
    if np.dot(g_["n"], n) > 0.999:
      g_["A"] += A
      g_["c"].append(ct)
      g_["v"].extend(verts_link[f])
      break
  else:
    groups.append({"n": n, "A": A, "c": [ct], "v": list(verts_link[f])})
for g_ in groups:
  g_["c"] = np.mean(g_["c"], axis=0)
  v = np.array(g_["v"])
  g_["zlo"], g_["zhi"] = v[:, 2].min(), v[:, 2].max()
  g_["ylo"], g_["yhi"] = v[:, 1].min(), v[:, 1].max()
groups.sort(key=lambda g_: -g_["A"])

# Which way is "inward"? Take it from the physics rather than guessing: settle a
# grip and read the sign of the contact normal MuJoCo reports on the left distal.
mujoco.mj_resetData(model, data)
for _n, _v in HOVER.items():
  data.qpos[model.joint(_n).qposadr[0]] = _v
_adr = model.jnt_qposadr[model.joint("cube_joint").id]
data.qpos[_adr : _adr + 3] = _GRASP_SITE_POS
for _n, _v in (
  ("left_1", 0.10),
  ("left_2", -0.40),
  ("right_1", -0.10),
  ("right_2", 0.40),
):
  data.ctrl[model.actuator(_n).id] = _v
for _ in range(3000):
  mujoco.mj_step(model, data)
_bid, _cg = model.body("left_2").id, model.geom("cube_geom").id
inward_link = None
for i in range(data.ncon):
  c = data.contact[i]
  gs = {c.geom1, c.geom2}
  if _cg not in gs or model.geom_bodyid[(gs - {_cg}).pop()] != _bid:
    continue
  nrm = c.frame[:3] if c.geom1 == _cg else -c.frame[:3]
  inward_link = data.xmat[_bid].reshape(3, 3).T @ nrm
  print(f"contact normal on left_2, in the link frame: {inward_link.round(3)}")
  print(
    f"contact point z = {(data.xmat[_bid].reshape(3, 3).T @ (c.pos - data.xpos[_bid]))[2] * 1000:+.1f} mm\n"
  )
  break
assert inward_link is not None

print("flat facets of the left_2 collision hull, in the link frame:")
print(
  f"{'area mm2':>10} {'normal xyz':>26} {'z span mm':>16} {'width mm':>9} {'dot(inward)':>12}"
)
pad = None
for g_ in groups[:8]:
  d = float(np.dot(g_["n"], inward_link))
  print(
    f"{g_['A'] * 1e6:10.1f} {g_['n'][0]:+8.3f}{g_['n'][1]:+8.3f}{g_['n'][2]:+8.3f}"
    f" {g_['zlo'] * 1000:+7.1f}..{g_['zhi'] * 1000:+6.1f} {(g_['yhi'] - g_['ylo']) * 1000:8.1f}"
    f" {d:+12.3f}" + ("   <- PAD" if pad is None and d > 0.9 else "")
  )
  if pad is None and d > 0.9:
    pad = g_
assert pad is not None, "no facet faces the way the contact normal points"
print(
  f"\npad: {pad['A'] * 1e6:.0f} mm2, spans link z {pad['zlo'] * 1000:+.1f}"
  f"..{pad['zhi'] * 1000:+.1f} mm\n"
)

# --- (b) curl x pitch grid ---------------------------------------------------
CURLS = (-0.40, -0.60, -0.85, -1.00, -1.20)
PROX = {-0.40: 0.10, -0.60: 0.00, -0.85: -0.10, -1.00: -0.10, -1.20: -0.10}
PITCHES = np.arange(-1.2, 1.21, 0.15)
cube_gid = model.geom("cube_geom").id


def settle(p_cmd, d_cmd, pitch):
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
  for _ in range(3000):
    mujoco.mj_step(model, data)


def count(body):
  bid = model.body(body).id
  n = 0
  for i in range(data.ncon):
    c = data.contact[i]
    gs = {c.geom1, c.geom2}
    if cube_gid in gs and model.geom_bodyid[(gs - {cube_gid}).pop()] == bid:
      n += 1
  return n


print("\npad-normal vs cube-face-normal angle (deg) at each settled curl,")
print("and contact-point count on the left distal over a wide cube-pitch sweep.")
print("A face-on-face grip needs the angle near 0 and the count above 1.\n")
hdr = "  curl  settled_d  padang  " + "".join(f"{np.degrees(p):+6.0f}" for p in PITCHES)
print(hdr)
for d_cmd in CURLS:
  settle(PROX[d_cmd], d_cmd, 0.0)
  bid = model.body("left_2").id
  Rw = data.xmat[bid].reshape(3, 3)
  nw = Rw @ pad["n"]
  # Cube face normal is world x (the jaw closes along x). Angle to the pad
  # normal, folded to [0, 90] since either face of the cube will do.
  ang = np.degrees(np.arccos(min(1.0, abs(nw[0]))))
  qd = data.qpos[model.joint("left_2").qposadr[0]]
  cells = []
  for pitch in PITCHES:
    settle(PROX[d_cmd], d_cmd, pitch)
    cells.append(count("left_2"))
  print(f"{d_cmd:+6.2f} {qd:+10.3f} {ang:7.1f}  " + "".join(f"{c:6d}" for c in cells))
