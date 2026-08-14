"""What the WIDE claw actually collides with, next to what it looks like.

  uv run python scripts/wide_show_colliders.py OUT_DIR

The renderer draws visual MESHES (group 1, contype=0) while the physics uses
fitted collision BOXES (group 3) -- a 4-band staircase per distal link, see
`fit_collision_boxes`. A box fitted to a mesh's bounds encloses it, so the
collision surface sits proud of the visible one, and a grasp can look like it
has a gap while being a genuine contact. This renders both so the difference is
visible rather than argued about, at the pose the claw actually grips in.

Three panels: the meshes alone, the collision boxes alone, and the boxes drawn
over the meshes.
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import twofinger_arm_spec

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "videos/colliders")
OUT.mkdir(parents=True, exist_ok=True)
W, H = 900, 700
# The angle the fingers SETTLE at when driven onto a 50 mm cube and stopped by
# it, measured in scripts/wide_calibrate.py section 2. Rendering the commanded
# full-close angle instead would draw the fingers inside the cube.
GRIP = {"left_1_tf": -0.0023, "right_1_tf": 0.0023, "left_2_tf": 0.0, "right_2_tf": 0.0}
CUBE_HALF = 0.025

spec = twofinger_arm_spec()
# A cube where the claw grips, so the gap under discussion is in frame. Free
# joint omitted on purpose: this is a geometry picture, not a simulation, and a
# static cube cannot be nudged out of place by the render's own settling.
cube_body = spec.worldbody.add_body(name="viz_cube")
g = cube_body.add_geom(
  name="viz_cube",
  type=mujoco.mjtGeom.mjGEOM_BOX,
  size=(CUBE_HALF,) * 3,
  rgba=(0.85, 0.85, 0.88, 1.0),
)
g.contype, g.conaffinity = 0, 0
# Group 0, which every panel below includes: the cube is the REFERENCE the gap
# is measured against, so it has to stay on screen in the collision-only panel
# too -- putting it in group 1 made it vanish exactly where it was needed.
g.group = 0

m = spec.compile()
d = mujoco.MjData(m)
for name, q in GRIP.items():
  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
  d.qpos[m.jnt_qposadr[jid]] = q
mujoco.mj_forward(m, d)

sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
mid = 0.5 * (d.site_xpos[sid("left_pad_tf")] + d.site_xpos[sid("right_pad_tf")])
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "viz_cube")
m.body_pos[bid] = mid
mujoco.mj_forward(m, d)
print(
  f"pad midpoint {mid}, pad separation "
  f"{np.linalg.norm(d.site_xpos[sid('left_pad_tf')] - d.site_xpos[sid('right_pad_tf')]) * 1e3:.1f} mm"
)

# A TRUE SIDE VIEW, derived rather than guessed. Two directions define the
# claw: the CLOSING axis between the pads, and the APPROACH axis along which the
# fingers extend from the palm. A side view has to be perpendicular to BOTH --
# looking along either one foreshortens the fingers into the cube and shows
# nothing, which is what a hand-picked azimuth with elevation 0 produced.
close_ax = d.site_xpos[sid("right_pad_tf")] - d.site_xpos[sid("left_pad_tf")]
close_ax /= np.linalg.norm(close_ax)
palm = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_base_tf")]
approach = mid - palm
approach /= np.linalg.norm(approach)
view = np.cross(close_ax, approach)
view /= np.linalg.norm(view)
print(f"closing axis {close_ax}\napproach     {approach}\nside view    {view}")

# HIDE THE ARM. The spec carries the whole Rizon4S, and at any distance close
# enough to see a 1 mm gap the camera sits INSIDE one of its links -- the first
# two attempts rendered the inside of the forearm and nothing else. Only the
# claw's own bodies and the cube are kept.
KEEP = (
  "hand_base_tf",
  "left_1_tf",
  "left_2_tf",
  "right_1_tf",
  "right_2_tf",
  "viz_cube",
)
keep_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in KEEP} - {-1}
for i in range(m.ngeom):
  if m.geom_bodyid[i] not in keep_bodies:
    m.geom_rgba[i] = (0.0, 0.0, 0.0, 0.0)
  elif m.geom_group[i] == 3:
    # Collision boxes in translucent red over the blue meshes; they otherwise
    # render near the mesh colour and are indistinguishable from it.
    m.geom_rgba[i] = (0.95, 0.20, 0.15, 0.50)

renderer = mujoco.Renderer(m, height=H, width=W)
PANELS = {
  "1_visual_meshes": (0, 1),
  "2_collision_boxes": (0, 3),
  "3_both": (0, 1, 3),
}
# Both ways along the side-view normal: they differ only in which finger is
# nearer the camera, and one of the two usually reads better.
for tag, sign in (("sideA", 1.0), ("sideB", -1.0)):
  u = view * sign
  cam = mujoco.MjvCamera()
  cam.lookat[:] = mid
  cam.azimuth = float(np.degrees(np.arctan2(u[1], u[0])))
  cam.elevation = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
  cam.distance = 0.13
  for label, groups in PANELS.items():
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for gr in groups:
      opt.geomgroup[gr] = 1
    renderer.update_scene(d, camera=cam, scene_option=opt)
    imageio.imwrite(OUT / f"wide_claw_{tag}_{label}.png", renderer.render())
  print(f"wrote {tag}  az {cam.azimuth:7.1f}  el {cam.elevation:6.1f}")
