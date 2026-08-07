"""Render preview images of the tabletop grasp scene."""

import os
import sys

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene import CUBE_HALF, D435, build_scene, ik_hover

OUT = os.environ["OUTDIR"]
spec = build_scene(cube_pos=(0.45, 0.0, CUBE_HALF))
m = spec.compile()
d = mujoco.MjData(m)
ik_hover(m, d, [0.45, 0.0, 0.16])  # gripper hovering over the cube

W, H = D435["res"]
r = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption()


def free(fn, lookat, az, el, dist):
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(m, cam)
  cam.lookat[:] = lookat
  cam.azimuth, cam.elevation, cam.distance = az, el, dist
  r.update_scene(d, cam, opt)
  iio.imwrite(fn, r.render())


# 1) overview
free(f"{OUT}/scene_overview.png", [0.4, 0.0, 0.1], 215, -20, 1.7)
# 2) side view (look along -y) to show camera-arm proximity
free(f"{OUT}/scene_side.png", [0.25, 0.0, 0.25], 90, -6, 1.0)
# 3) D435 RGB
r.update_scene(d, "d435")
iio.imwrite(f"{OUT}/d435_rgb.png", r.render())
# 4) D435 depth (clip to D435 range; near=bright)
r.enable_depth_rendering()
r.update_scene(d, "d435")
depth = r.render()
r.disable_depth_rendering()
valid = (depth > D435["min_z"]) & (depth < D435["max_z"])
near = depth[valid].min() if valid.any() else D435["min_z"]
far = np.percentile(depth[valid], 97) if valid.any() else D435["max_z"]
dv = 1 - np.clip((depth - near) / (far - near + 1e-6), 0, 1)
dv[~valid] = 0
iio.imwrite(f"{OUT}/d435_depth.png", (np.stack([dv] * 3, -1) * 255).astype(np.uint8))

cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "d435")
print("camera pos:", d.cam_xpos[cam].round(3), "fovy:", D435["depth_fovy"])
print("depth valid range (m): %.3f - %.3f" % (near, far))
print("rendered scene_overview / scene_side / d435_rgb / d435_depth")
