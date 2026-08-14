"""Roll out a trained checkpoint and render the most successful episode.

  uv run python scripts/render_rollout.py TASK CKPT OUT.mp4

Runs N envs, reports how high each one got the cube, then re-renders the single
best env. Picking the best (rather than a random) env is deliberate: if even the
most favourable episode shows no lift, that is a real result, not bad luck.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import geometry_for  # noqa: E402
from tools.video_out import video_path  # noqa: E402

TASK, CKPT = sys.argv[1], sys.argv[2]
OUT = video_path(sys.argv[3])
# The surface heights are read off the TASK: the wide-claw bench measures cube
# rise from the top of 50 mm of foam, the old one from the table top.
_GEO = geometry_for(TASK)
LIFT_HEIGHT, TABLE_H = _GEO.lift_height, _GEO.surface_z
CUBE_HALF = 0.025
CAM_DIST = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
N, STEPS, SEED, DEV = 32, 300, 0, "cuda:0"


def build(render: bool, env_idx: int = 0):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}  # continuous rollout, no reset jumps mid-video
  if render:
    cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="cube",
      env_idx=env_idx,
      distance=CAM_DIST,
      elevation=-20.0,
      azimuth=140.0,
      max_extra_envs=0,
      height=540,
      width=960,
    )
  return ManagerBasedRlEnv(
    cfg=cfg, device=DEV, render_mode="rgb_array" if render else None
  )


def make_policy(env):
  agent_cfg = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=DEV)
  runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
  return wrapped, runner.get_inference_policy(device=DEV)


def rollout(env, wrapped, policy, render: bool):
  torch.manual_seed(SEED)
  obs = wrapped.reset()[0]
  heights, frames = [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    z = env.scene["cube"].data.root_link_pos_w[:, 2]
    origin_z = env.scene.env_origins[:, 2]
    heights.append((z - origin_z - TABLE_H).clone())  # height above the table top
    if render:
      frames.append(env.render())
  return torch.stack(heights), frames


env = build(render=False)
wrapped, policy = make_policy(env)
heights, _ = rollout(env, wrapped, policy, render=False)
peak = heights.max(dim=0).values  # [N] best height each env reached
best = int(peak.argmax())
rest = CUBE_HALF  # cube half-height: its centre when resting on the table

print(f"checkpoint: {CKPT}")
print(f"cube height above table top, peak over {STEPS} steps ({N} envs):")
print(f"  resting height   = {rest:.4f} m")
print(f"  best env ({best:2d})     = {peak[best]:.4f} m")
print(f"  median           = {peak.median():.4f} m")
print(f"  envs above +1cm  = {int((peak > rest + 0.01).sum())}/{N}")
print(
  f"  envs above {LIFT_HEIGHT:.2f}m  = {int((peak > LIFT_HEIGHT).sum())}/{N}  <- success bar"
)
env.close()

env2 = build(render=True, env_idx=best)
wrapped2, policy2 = make_policy(env2)
_, frames = rollout(env2, wrapped2, policy2, render=True)
env2.close()

frames = [f for f in frames if f is not None]
imageio.mimwrite(OUT, frames, fps=30, quality=8)
print(f"wrote {len(frames)} frames -> {OUT}")
