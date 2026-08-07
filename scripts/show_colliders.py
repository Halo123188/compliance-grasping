"""Render the hand's collision geometry against the visual mesh, and measure
the jaw's closing axis and the real finger-pad slope.

  uv run python scripts/show_colliders.py OUT_DIR
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/yiboc")
XML = "/home/yiboc/compliance-grasping/assets/two_finger_hand/two_finger_hand.xml"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
qa = {
  n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
  for n in ("left_1", "right_1", "left_2", "right_2")
}
gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731

# ---- 1. Jaw closing axis: which way do the pads actually face? --------------
d.qpos[:] = 0
mujoco.mj_forward(m, d)
axis = d.geom_xpos[gid("right_2_col")] - d.geom_xpos[gid("left_2_col")]
axis /= np.linalg.norm(axis)
print(f"jaw closing axis (palm frame) = {axis.round(4)}")
print(f"  angle from +X = {np.degrees(np.arctan2(axis[1], axis[0])):.1f} deg")
print("  -> a cube with yaw=0 is met CORNER-first unless this is 0 or 90 deg\n")

# ---- 2. Real finger pad: is it flat, and what is its slope? -----------------
b = bid("left_2")
for g in range(m.ngeom):
  if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
    continue
  mid = m.geom_dataid[g]
  V = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
  # Inner face = vertices near the max-x surface (the side facing the other finger)
  inner = V[V[:, 0] > V[:, 0].max() - 0.0015]
  print(f"inner-pad vertices: {len(inner)} of {len(V)}")
  print(
    f"  pad extent  y [{inner[:, 1].min():+.4f},{inner[:, 1].max():+.4f}]  "
    f"z [{inner[:, 2].min():+.4f},{inner[:, 2].max():+.4f}]"
  )
  # Fit a plane to the inner surface -> its normal gives the pad slope.
  c = inner.mean(0)
  n = np.linalg.svd(inner - c)[2][-1]
  if n[0] < 0:
    n = -n
  print(f"  pad plane normal = {n.round(4)}  centre = {c.round(4)}")
  print(
    f"  slope from vertical (about the hinge axis Y) = "
    f"{np.degrees(np.arctan2(n[2], n[0])):+.1f} deg"
  )
  print(
    f"  slope sideways                                = "
    f"{np.degrees(np.arctan2(n[1], n[0])):+.1f} deg"
  )
  # How far does the real surface deviate from that plane (i.e. is it flat)?
  dev = (inner - c) @ n
  print(f"  flatness: max |deviation| = {np.abs(dev).max() * 1000:.2f} mm\n")

# ---- 3. Render: collision solid, visual mesh translucent --------------------
for g in range(m.ngeom):
  if m.geom_group[g] == 1:  # visual meshes
    m.geom_rgba[g] = [0.75, 0.78, 0.82, 0.30]
  elif m.geom_group[g] == 3:  # colliders
    m.geom_rgba[g] = [1.0, 0.25, 0.15, 0.85]

opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = 1
opt.geomgroup[3] = 1

m.vis.global_.offheight, m.vis.global_.offwidth = 640, 640
r = mujoco.Renderer(m, height=640, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, -0.03]
cam.distance = 0.26

tiles = []
for jaw, tag in ((0.90, "open 0.90"), (0.30, "grip 0.30"), (0.0, "neutral 0.0")):
  d.qpos[:] = 0
  d.qpos[qa["left_1"]], d.qpos[qa["right_1"]] = jaw, -jaw
  mujoco.mj_forward(m, d)
  row = []
  for az, el in ((90, -10), (0, -10), (90, -70)):
    cam.azimuth, cam.elevation = az, el
    r.update_scene(d, camera=cam, scene_option=opt)
    row.append(r.render())
  tiles.append(np.concatenate(row, axis=1))
  print(f"rendered {tag}")

imageio.imwrite(OUT / "colliders.png", np.concatenate(tiles, axis=0))
print(
  f"wrote {OUT / 'colliders.png'}  (rows: open / grip / neutral;"
  f" cols: front, side, top)"
)
