"""Is the pad site actually on the finger's inner gripping face?

  uv run python scripts/diag_pad_site.py

``two_finger_hand.xml`` places ``left_pad``/``right_pad`` by hand and calls them
"Centre of the real inner gripping face". That claim is checkable: the collision
mesh IS the real finger, so find its inner face and compare.

Method: take the collision mesh's vertices in the LINK frame (the mesh geom
carries a compiled pos/quat, so raw mesh coordinates are not link coordinates),
keep the ones on the inner side, and fit the flat pad -- the inner face is the
one large planar patch facing the other finger.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

model = mujoco.MjSpec.from_file(str(HAND_XML)).compile()

for side, inner_sign in (("left", +1.0), ("right", -1.0)):
  gid = model.geom(f"{side}_2_col").id
  mid = model.geom_dataid[gid]
  v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
  verts = model.mesh_vert[v0 : v0 + nv].astype(np.float64)

  # Mesh -> link frame: the geom's own pos/quat. Skipping this is the mistake the
  # XML comment warns about; it is worth 10-26 mm here.
  q = model.geom_quat[gid]
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, q)
  verts = verts @ R.reshape(3, 3).T + model.geom_pos[gid]

  # The inner face is the extreme surface in the closing direction. Take the
  # vertices within 1 mm of that extreme: on a flat pad that is the pad itself.
  x = verts[:, 0] * inner_sign
  pad = verts[np.abs(x - x.max()) < 1e-3]
  site = model.site(f"{side}_pad").pos

  print(f"=== {side} finger ===")
  print("  collision mesh spans (link frame, mm):")
  for ax, name in enumerate("xyz"):
    print(
      f"    {name}: {verts[:, ax].min() * 1000:+7.1f} .. {verts[:, ax].max() * 1000:+7.1f}"
    )
  print(f"  inner-face patch: {len(pad)} verts")
  print(
    f"    centroid = ({pad[:, 0].mean() * 1000:+.2f}, {pad[:, 1].mean() * 1000:+.2f}, "
    f"{pad[:, 2].mean() * 1000:+.2f}) mm"
  )
  print(
    f"    z span   = {pad[:, 2].min() * 1000:+.1f} .. {pad[:, 2].max() * 1000:+.1f} mm"
  )
  print(
    f"    y span   = {pad[:, 1].min() * 1000:+.1f} .. {pad[:, 1].max() * 1000:+.1f} mm"
  )
  print(
    f"  site pos   = ({site[0] * 1000:+.2f}, {site[1] * 1000:+.2f}, {site[2] * 1000:+.2f}) mm"
  )
  d = site - np.array([pad[:, 0].mean(), pad[:, 1].mean(), pad[:, 2].mean()])
  print(
    f"  site - face centroid = ({d[0] * 1000:+.2f}, {d[1] * 1000:+.2f}, "
    f"{d[2] * 1000:+.2f}) mm   |error| = {np.linalg.norm(d) * 1000:.2f} mm"
  )
  print(f"  tip of finger is at z = {verts[:, 2].min() * 1000:+.1f} mm")
