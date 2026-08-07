"""Render the rebuilt scene: third-person + top-down + the D435 camera view.

Adds non-colliding axis markers on the table so the world frame is unambiguous:
  RED capsule   = +X   (out from the arm base, toward the cube region)
  BLUE capsule  = +Y   (arm's left)
  WHITE sphere  = world origin / arm base mount point
"""

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import TABLE_H
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

OUT = "/work/yiboc"
Z = TABLE_H + 0.004  # just proud of the table top


def get_axes_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="axes")
  specs = [
    ("plus_x", (0.02, 0.0, Z), (0.55, 0.0, Z), (0.9, 0.1, 0.1, 1.0)),
    ("minus_x", (-0.02, 0.0, Z), (-0.09, 0.0, Z), (1.0, 0.5, 0.5, 0.6)),
    ("plus_y", (0.0, 0.02, Z), (0.0, 0.37, Z), (0.1, 0.3, 0.95, 1.0)),
    ("minus_y", (0.0, -0.02, Z), (0.0, -0.37, Z), (0.35, 0.55, 1.0, 0.6)),
  ]
  for name, a, b, rgba in specs:
    g = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=(*a, *b),
      size=(0.006, 0.0, 0.0),
      rgba=rgba,
    )
    g.contype, g.conaffinity = 0, 0
  g = body.add_geom(
    name="origin",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(0.0, 0.0, Z),
    size=(0.015, 0.0, 0.0),
    rgba=(1.0, 1.0, 1.0, 1.0),
  )
  g.contype, g.conaffinity = 0, 0
  return spec


cfg = load_env_cfg("Mjlab-Grasp-TwoFinger-Flexiv", play=True)
cfg.scene.num_envs = 4
cfg.terminations = {}
cfg.scene.entities["axes"] = EntityCfg(spec_fn=get_axes_spec)
# Fixed entities only land on their env_origin if a reset event writes their
# mocap pose (see the "reset_table" note in env_cfgs).
cfg.events["reset_axes"] = EventTermCfg(
  func=reset_root_state_uniform,
  mode="reset",
  params={
    "pose_range": {},
    "velocity_range": {},
    "asset_cfg": SceneEntityCfg("axes"),
  },
)

views = {
  "thirdperson": dict(
    lookat=(0.4, 0.0, 0.45), distance=2.0, elevation=-18, azimuth=135
  ),
  "front": dict(lookat=(0.45, 0.0, 0.45), distance=1.7, elevation=-12, azimuth=180),
  "topdown": dict(lookat=(0.45, 0.0, 0.40), distance=2.4, elevation=-89, azimuth=90),
}


def viewer(**kw) -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    env_idx=0,
    max_extra_envs=0,
    # 16:9 to match the policy camera (128x72): MuJoCo derives the horizontal
    # FOV from fovy and the viewport aspect, so a 4:3 viewport would render the
    # D435 with a narrower cone than it really has.
    height=540,
    width=960,
    **kw,
  )


cfg.viewer = viewer(**views["thirdperson"])
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
torch.manual_seed(0)
env.reset()
nact = env.action_manager.total_action_dim
for _ in range(25):
  env.step(torch.zeros(4, nact, device="cuda:0"))

robot = env.scene["robot"]
m = env.sim.mj_model
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/base")
print(f"arm base body pos  = {m.body_pos[bid]}   (table top z = {TABLE_H})")
print(
  f"cube z after settle = {env.scene['cube'].data.root_link_pos_w[0, 2].item():.4f}"
)
sid = robot.site_names.index("grasp_site")
print(f"grasp_site world    = {robot.data.site_pos_w[0, sid].tolist()}")

cam = env._offline_renderer._cam
# Views are specified in env-local coords; env 0 lives at its own grid origin.
o = env.scene.env_origins[0].cpu().numpy()
for name, kw in views.items():
  cam.lookat[:] = np.asarray(kw["lookat"]) + o
  cam.distance = kw["distance"]
  cam.elevation = kw["elevation"]
  cam.azimuth = kw["azimuth"]
  imageio.imwrite(f"{OUT}/scene_{name}.png", env.render())
  print(f"wrote scene_{name}.png")

r = env._offline_renderer
r.update(env.sim.data, camera="robot/d435")
imageio.imwrite(f"{OUT}/scene_d435_axes.png", r.render())
print("wrote scene_d435_axes.png")

# Same shot with the debug axis markers hidden: this is exactly what the vision
# policy sees. geom_rgba is a per-world field the renderer syncs from sim_model.
gids = [
  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"axes/{n}")
  for n in ("plus_x", "minus_x", "plus_y", "minus_y", "origin")
]
env.sim.model.geom_rgba[:, gids, 3] = 0.0
r.update(env.sim.data, camera="robot/d435")
imageio.imwrite(f"{OUT}/scene_d435.png", r.render())
print("wrote scene_d435.png")
