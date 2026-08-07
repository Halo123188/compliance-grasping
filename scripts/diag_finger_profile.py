"""Profile the finger's inner surface, so a pad site can be moved along it.

  uv run python scripts/diag_finger_profile.py

Moving the pad site toward the tip is not a matter of changing z alone: below
the flat pad the finger tapers, so the inner surface pulls back in x and narrows
in y. A site that keeps the pad's x while sliding down z ends up floating in
free space off the front of the finger.

Reports, per 2 mm slice of z, where the inner surface actually is.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

model = mujoco.MjSpec.from_file(str(HAND_XML)).compile()

gid = model.geom("left_2_col").id
mid = model.geom_dataid[gid]
v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
verts = model.mesh_vert[v0 : v0 + nv].astype(np.float64)
R = np.zeros(9)
mujoco.mju_quat2Mat(R, model.geom_quat[gid])
verts = verts @ R.reshape(3, 3).T + model.geom_pos[gid]
verts *= 1000.0  # mm

site = model.site("left_pad").pos * 1000.0
print(f"left_pad site now at ({site[0]:+.2f}, {site[1]:+.2f}, {site[2]:+.2f}) mm")
print(f"finger spans z {verts[:, 2].min():+.1f} .. {verts[:, 2].max():+.1f} mm")
print("\n   z slice    inner x   y extent        width   n")
for z0 in np.arange(-54, -14, 2.0):
  sl = verts[(verts[:, 2] >= z0) & (verts[:, 2] < z0 + 2.0)]
  if len(sl) < 4:
    continue
  xin = sl[:, 0].max()
  near = sl[sl[:, 0] > xin - 1.0]  # vertices on the inner surface of this slice
  y0, y1 = near[:, 1].min(), near[:, 1].max()
  mark = "  <- site" if z0 <= site[2] < z0 + 2.0 else ""
  print(
    f"  {z0:+5.0f}..{z0 + 2:+5.0f}  {xin:+7.2f}   {y0:+6.1f}..{y1:+6.1f}  "
    f"{y1 - y0:6.1f}  {len(sl):4d}{mark}"
  )
