"""Pick a hover pose that is already shaped like the grasp, and wide enough.

  uv run python scripts/diag_hover_pose.py

The hover pose the task starts from has the distals straight (left_2 = 0), so the
policy has to discover the inward curl AND the close. The curl is what the
pull-out test says decides whether the grasp holds at all (5.2-6.1 N with it,
1.3-3.1 N without), so starting straight makes the policy find the one thing it
has never found.

Constraint on how far it can be pre-closed: the cube spawns at a uniform random
yaw, so the jaw has to clear its DIAGONAL, 50*sqrt(2) = 70.7 mm, not its 50 mm
face. This sweeps the proximal at a fixed inward curl and reports the jaw gap so
a pose can be picked against that bar rather than guessed.

Also reports the pad face's extent in y, which is the axis the pad site should
be CENTRED on -- the cube covers the pad completely along it, so a single
contact point carries no information about where in y the load sits.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import HAND_XML

CUBE, DIAG = 0.050, 0.050 * np.sqrt(2)
CURL = -0.40  # the pull-out-test curl

model = mujoco.MjSpec.from_file(str(HAND_XML)).compile()
data = mujoco.MjData(model)
sl, sr = model.site("left_pad").id, model.site("right_pad").id


def gap(l1: float, l2: float) -> float:
  data.qpos[:] = 0.0
  for n, v in (("left_1", l1), ("left_2", l2), ("right_1", -l1), ("right_2", -l2)):
    data.qpos[model.joint(n).qposadr[0]] = v
  mujoco.mj_forward(model, data)
  return float(np.linalg.norm(data.site_xpos[sl] - data.site_xpos[sr]))


print(f"cube {CUBE * 1000:.0f} mm, diagonal {DIAG * 1000:.1f} mm -- the jaw must clear")
print(f"\njaw gap vs proximal, distal held at the grasp curl {CURL:+.2f}:")
for a in np.arange(0.20, 1.35, 0.10):
  g = gap(a, CURL)
  mark = "  <- clears the diagonal" if g > DIAG + 0.008 else ""
  print(f"  left_1={a:+.2f}  gap = {g * 1000:6.1f} mm{mark}")

print("\nfor reference, the current hover (left_1=+0.70, distals straight):")
print(f"  gap = {gap(0.70, 0.0) * 1000:.1f} mm")
print(f"the pull-out-test grasp (left_1=+0.10, curl {CURL:+.2f}):")
print(f"  gap = {gap(0.10, CURL) * 1000:.1f} mm")

# --- where should the pad site sit along the pad's width (y)? ---
gid = model.geom("left_2_col").id
mid = model.geom_dataid[gid]
v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
verts = model.mesh_vert[v0 : v0 + nv].astype(np.float64)
R = np.zeros(9)
mujoco.mju_quat2Mat(R, model.geom_quat[gid])
verts = verts @ R.reshape(3, 3).T + model.geom_pos[gid]
face = verts[np.abs(verts[:, 0] - verts[:, 0].max()) < 1e-3]
print(
  f"\nleft pad inner face, y extent: {face[:, 1].min() * 1000:+.1f} .."
  f" {face[:, 1].max() * 1000:+.1f} mm   centre {face[:, 1].mean() * 1000:+.2f} mm"
)
print(
  f"a {CUBE * 1000:.0f} mm cube spans {CUBE * 1000:.0f} mm > the"
  f" {(face[:, 1].max() - face[:, 1].min()) * 1000:.1f} mm face, so it covers"
  " the pad in y"
)
print(f"current left_pad site y = {model.site('left_pad').pos[1] * 1000:+.2f} mm")
