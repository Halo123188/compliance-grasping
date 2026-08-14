"""How wide is the jaw at each height up the finger, as it closes?

  uv run python scripts/wide_finger_profile.py

The claw is a HOOK, not a parallel jaw, so "jaw opening" is not one number -- the
separation between the two fingers depends on how far up the finger you measure.
That matters for small objects: the flat inner gripping face sits 12.5-27.5 mm
above the fingertip, so an object shorter than ~25 mm cannot be reached by that
face without driving the tip under the work surface. But the TAPERED TIP below
that face swings inward as the fingers close, and it can pinch a small object at
a height the flat face can never reach.

This measures the whole inner profile rather than one face: for each closure
angle it reports the finger-to-finger separation at a series of heights above the
fingertip. Read a row as "at this closure, an object of THIS width is pinched at
THIS height above the tip", and compare that height against the object's own
height to see whether the tip clears the surface.

The proximal links are excluded: for objects this small the distal pair is what
closes first, and including the proximals would report their separation as the
binding one at heights they cannot actually reach.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  TWOFINGER_ARM_HOME,
  twofinger_arm_spec,
)
from mjlab.asset_zoo.robots.twofinger_wide.hand_urdf import (
  MESH_DIR,
  load_hand_cad,
  stl_vertices,
  tip_marker_pos,
)

HEIGHTS_MM = (0, 2, 4, 6, 9, 12, 16, 20, 25)
CLOSURES = (0.0, -0.0754, -0.15, -0.22, -0.30, -0.38, -0.45, -0.55)
BAND = 0.0015  # half-thickness of the height slice sampled, metres

cad = load_hand_cad()
VERTS = {
  s: stl_vertices(MESH_DIR / f"{cad.links[f'{s}_2'].mesh}.stl")
  for s in ("left", "right")
}
spec = twofinger_arm_spec()
model = spec.compile()
data = mujoco.MjData(model)


def _pose(q: float):
  """Arm at its hover pose, fingers at closure `q`; returns world meshes."""
  for name, val in (TWOFINGER_ARM_HOME.joint_pos or {}).items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
      data.qpos[model.jnt_qposadr[jid]] = val
  for name, val in (
    ("left_1_tf", q),
    ("right_1_tf", -q),
    ("left_2_tf", 0.0),
    ("right_2_tf", 0.0),
  ):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = val
  mujoco.mj_forward(model, data)

  world = {}
  for side in ("left", "right"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_2_tf")
    rot = data.xmat[bid].reshape(3, 3)
    world[side] = data.xpos[bid] + VERTS[side] @ rot.T

  sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
  axis = data.site_xpos[sid("right_pad_tf")] - data.site_xpos[sid("left_pad_tf")]
  axis /= np.linalg.norm(axis)
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_2_tf")
  tip = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array(
    tip_marker_pos("right")
  )
  return world, axis, float(tip[2])


def main() -> None:
  print("separation between the two fingers, mm, by height above the fingertip\n")
  head = "".join(f"{h:>7}" for h in HEIGHTS_MM)
  print(f"{'q (rad)':>9}  {'tip z':>7}   {head}")
  print(f"{'':>9}  {'(mm)':>7}   " + "".join(f"{'mm':>7}" for _ in HEIGHTS_MM))
  for q in CLOSURES:
    world, axis, tip_z = _pose(q)
    cells = []
    for h in HEIGHTS_MM:
      z = tip_z + h / 1000.0
      near = {s: world[s][np.abs(world[s][:, 2] - z) < BAND] for s in ("left", "right")}
      if len(near["left"]) < 3 or len(near["right"]) < 3:
        cells.append(f"{'--':>7}")
        continue
      gap = (near["right"] @ axis).min() - (near["left"] @ axis).max()
      cells.append(f"{gap * 1e3:7.1f}")
    flag = "  <- current limit" if abs(q + 0.0754) < 1e-6 else ""
    print(f"{q:9.3f}  {tip_z * 1e3:7.1f}   " + "".join(cells) + flag)
  print(
    "\nA negative separation means the two fingers have crossed at that height."
    "\nFor an object of width W sitting on the surface, find the row where some"
    "\nheight <= the object's own height reads W: that closure grips it with the"
    "\ntip at that height above the surface."
  )


if __name__ == "__main__":
  main()
