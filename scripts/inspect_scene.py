import imageio.v2 as imageio
import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

cfg = load_env_cfg("Mjlab-Grasp-TwoFinger-Flexiv", play=True)
cfg.scene.num_envs = 4
cfg.terminations = {}
cfg.viewer = ViewerConfig(
  origin_type=ViewerConfig.OriginType.WORLD,
  lookat=(0.4, 0.0, 0.1),
  distance=1.6,
  elevation=-20,
  azimuth=135,
  env_idx=0,
  max_extra_envs=0,
  height=480,
  width=640,
)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
torch.manual_seed(0)
env.reset()
m = env.sim.mj_model
for gi in range(m.ngeom):
  n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom{gi}"
  if any(k in n for k in ["terrain", "table", "cube", "floor", "ground"]):
    print(
      f"{n:22} type={m.geom_type[gi]} pos={m.geom_pos[gi]} "
      f"size={m.geom_size[gi]} contype={m.geom_contype[gi]} "
      f"conaff={m.geom_conaffinity[gi]} group={m.geom_group[gi]}"
    )
nact = env.action_manager.total_action_dim
z0 = env.scene["cube"].data.root_link_pos_w[0, 2].item()
for _ in range(40):
  env.step(torch.zeros(4, nact, device="cuda:0"))
z1 = env.scene["cube"].data.root_link_pos_w[0, 2].item()
print(f"cube z: spawn={z0:.4f}  after_settle={z1:.4f}  drift={z1 - z0:+.4f}")
imageio.imwrite("/work/yiboc/scene_frame.png", env.render())
print("wrote /work/yiboc/scene_frame.png")
