"""Which joint angles make the two gripping faces PARALLEL to each other?

  uv run python scripts/diag_pad_parallel.py [OUT.png]

diag_pad_normal.py established the real geometry: the gripping face is the whole
inner side of the distal link (812 mm2, 33 mm wide, spanning link z -55..+8), and
at the grip pose the task actually uses it sits 11.4 deg off the cube face --
which gets WORSE with deeper curl, not better (19.6 deg at -0.60, 30.9 at -0.85,
39.6 at -1.20).

Mirrored, that means the two pads form a wedge of 2x the tilt. A cube squeezed
between two non-parallel planes cannot lie flush on both; it touches each at one
point, which is the point contact the pull-out test has been measuring all along.

So the question is not "curl deeper", it is "what (proximal, distal) makes the
two pads parallel AND leaves a 50 mm gap between them". This is pure kinematics
-- no cube, no contact, just mj_kinematics on a grid -- so it answers exactly
that, and then checks the answer against the joint limits.
"""

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.two_finger_hand.constants import _GRASP_SITE_POS, HAND_XML

OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/pad_parallel.png"
CUBE = 0.050
W, H = 560, 660

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
g = b.add_geom(
  type=mujoco.mjtGeom.mjGEOM_BOX,
  size=[CUBE / 2] * 3,
  contype=0,
  conaffinity=0,
  rgba=[0.9, 0.9, 0.9, 0.5],
)
g.name = "cube_geom"
model = spec.compile()
data = mujoco.MjData(model)


def pad_facet(geom_name: str):
  """Largest inward-facing flat facet of a distal collision hull, in link frame."""
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
  # Inward = toward the other finger. Left closes toward -x, right toward +x.
  want = -1.0 if geom_name.startswith("left") else +1.0
  cand = [gr for gr in groups if np.dot(gr["n"], [want, 0, 0]) > 0.9]
  gr = max(cand, key=lambda g_: g_["A"])
  gr["c"] = np.array(gr["v"]).mean(axis=0)
  return gr["n"], gr["c"], gr["A"]


nL, cL, aL = pad_facet("left_2_col")
nR, cR, aR = pad_facet("right_2_col")
print(f"left  pad {aL * 1e6:6.0f} mm2  normal {nL.round(3)}  centroid {cL.round(4)}")
print(f"right pad {aR * 1e6:6.0f} mm2  normal {nR.round(3)}  centroid {cR.round(4)}\n")


def measure(p: float, d: float):
  """Pad tilt off the cube face, and the gap between the two pad planes."""
  data.qpos[:] = 0
  for n_, v in (("left_1", p), ("left_2", d), ("right_1", -p), ("right_2", -d)):
    data.qpos[model.joint(n_).qposadr[0]] = v
  mujoco.mj_kinematics(model, data)
  out = []
  for body, nl, cl in (("left_2", nL, cL), ("right_2", nR, cR)):
    bid = model.body(body).id
    R = data.xmat[bid].reshape(3, 3)
    out.append((R @ nl, data.xpos[bid] + R @ cl))
  (nw_l, pw_l), (nw_r, pw_r) = out
  # Tilt of each pad off the cube's x-facing face, folded into [0, 90].
  tilt_l = np.degrees(np.arccos(min(1.0, abs(nw_l[0]))))
  tilt_r = np.degrees(np.arccos(min(1.0, abs(nw_r[0]))))
  # Angle between the two pads: 0 = parallel, which is what a flush grip needs.
  wedge = np.degrees(np.arccos(min(1.0, abs(float(np.dot(nw_l, nw_r))))))
  # Gap: project the separation of the two pad points onto the left pad normal.
  gap = abs(float(np.dot(pw_r - pw_l, nw_l)))
  return tilt_l, tilt_r, wedge, gap


print("tilt = one pad off the cube face; wedge = angle BETWEEN the two pads")
print("gap  = clear distance between the pad planes; the cube needs 50 mm\n")
PROX = np.arange(-0.20, 1.01, 0.10)
DIST = np.arange(-1.20, 1.01, 0.10)
print("wedge angle (deg), rows = proximal, cols = distal")
print("  p\\d " + "".join(f"{d:+6.1f}" for d in DIST))
for p in PROX:
  print(f"{p:+5.2f} " + "".join(f"{measure(p, d)[2]:6.1f}" for d in DIST))

print("\ngap (mm), rows = proximal, cols = distal")
print("  p\\d " + "".join(f"{d:+6.1f}" for d in DIST))
for p in PROX:
  print(f"{p:+5.2f} " + "".join(f"{measure(p, d)[3] * 1000:6.1f}" for d in DIST))

# The two conditions together: pads parallel to each other AND to the cube face,
# with a 50 mm gap. Scan finely for the best simultaneous solution.
best = None
for p in np.arange(-0.30, 1.21, 0.005):
  for d in np.arange(-1.30, 1.21, 0.005):
    tl, tr, wedge, gap = measure(p, d)
    if abs(gap * 1000 - CUBE * 1000) > 0.5:
      continue
    score = max(tl, tr)
    if best is None or score < best[0]:
      best = (score, p, d, tl, tr, wedge, gap)
print(
  f"\nbest simultaneous solution: proximal {best[1]:+.3f}, distal {best[2]:+.3f}\n"
  f"  pad tilt off cube face: left {best[3]:.2f} deg, right {best[4]:.2f} deg\n"
  f"  wedge between pads: {best[5]:.2f} deg\n"
  f"  gap: {best[6] * 1000:.1f} mm"
)
lo, hi = model.jnt_range[model.joint("left_2").id]
print(f"  left_2 XML range {lo:+.2f}..{hi:+.2f} -> reachable: {lo <= best[2] <= hi}")
lo, hi = model.jnt_range[model.joint("left_1").id]
print(f"  left_1 XML range {lo:+.2f}..{hi:+.2f} -> reachable: {lo <= best[1] <= hi}")

# --- render: the current grip pose vs the parallel-pad pose -------------------
CURRENT = (0.203, -0.359)  # what the pull-out test settles at
PANELS = (
  ("current grip", *CURRENT),
  ("parallel pads", best[1], best[2]),
)
panels, labels = [], []
for name, p, d in PANELS:
  data.qpos[:] = 0
  for n_, v in (("left_1", p), ("left_2", d), ("right_1", -p), ("right_2", -d)):
    data.qpos[model.joint(n_).qposadr[0]] = v
  mujoco.mj_forward(model, data)
  renderer = mujoco.Renderer(model, height=H, width=W)
  opt = mujoco.MjvOption()
  opt.geomgroup[1] = 1
  opt.geomgroup[3] = 0
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = _GRASP_SITE_POS
  cam.distance, cam.azimuth, cam.elevation = 0.19, 90.0, -12.0
  renderer.update_scene(data, camera=cam, scene_option=opt)
  panels.append(renderer.render())
  renderer.close()
  labels.append((name, p, d, measure(p, d)))

sheet = Image.fromarray(np.hstack(panels))
draw = ImageDraw.Draw(sheet)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 22)
  small = ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
  font = small = ImageFont.load_default()
for i, (name, p, d, (tl, _tr, wedge, gap)) in enumerate(labels):
  col = (120, 255, 140) if wedge < 3 else (255, 150, 110)
  draw.text((i * W + 12, 10), name, fill=(255, 255, 255), font=font)
  draw.text(
    (i * W + 12, 40), f"prox {p:+.3f}  dist {d:+.3f}", fill=(205, 205, 205), font=small
  )
  draw.text((i * W + 12, 64), f"pad tilt {tl:.1f} deg", fill=col, font=small)
  draw.text(
    (i * W + 12, 88), f"wedge between pads {wedge:.1f} deg", fill=col, font=small
  )
  draw.text(
    (i * W + 12, 112), f"gap {gap * 1000:.1f} mm (cube 50)", fill=col, font=small
  )
sheet.save(OUT)
print(f"\nwrote {OUT}")
