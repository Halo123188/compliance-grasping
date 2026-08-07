"""Render a video of the two-finger arm approaching the cube.

Deterministic (inference) policy. The learned reaching is reliable even though
the lift is not, so this shows the arm descending to the cube. We pick the env
whose grasp site gets closest to its cube and lock the camera onto that env.
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

TASK = "Mjlab-Grasp-TwoFinger-Flexiv"
CKPT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/approach.mp4"
N = 32
STEPS = 200
SEED = 0
DEV = "cuda:0"


def build(render: bool, env_idx: int = 0):
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = N
  env_cfg.terminations = {}  # continuous rollout, no reset jumps
  if render:
    env_cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="cube",
      env_idx=env_idx,
      distance=1.15,
      elevation=-22.0,
      azimuth=135.0,
      max_extra_envs=0,
      height=480,
      width=640,
    )
  mode = "rgb_array" if render else None
  return ManagerBasedRlEnv(cfg=env_cfg, device=DEV, render_mode=mode)


def make_policy(env):
  agent_cfg = load_rl_cfg(TASK)
  runner_cls = load_runner_cls(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(wrapped, asdict(agent_cfg), device=DEV)
  runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
  return wrapped, runner.get_inference_policy(device=DEV)


def grasp_site_pos(env):
  robot = env.scene["robot"]
  sid = robot.site_names.index("grasp_site")
  return robot.data.site_pos_w[:, sid]  # [N, 3]


def rollout(env, wrapped, policy, render):
  torch.manual_seed(SEED)
  obs = wrapped.reset()[0]
  dists, frames = [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    d = torch.norm(grasp_site_pos(env) - env.scene["cube"].data.root_link_pos_w, dim=-1)
    dists.append(d.clone())
    if render:
      frames.append(env.render())
  return torch.stack(dists), frames  # dists: [T, N]


def main():
  # Pass 1: find the env whose grasp site gets closest to its cube.
  env = build(render=False)
  wrapped, policy = make_policy(env)
  dists, _ = rollout(env, wrapped, policy, render=False)
  closest = dists.min(dim=0).values  # [N]
  k = int(closest.argmin())
  print(f"best-approach env={k}  min ee->cube dist={closest[k]:.4f} m")
  print(f"top-5 closest: {sorted(closest.tolist())[:5]}")
  env.close()

  # Pass 2: same seed, render env k.
  env2 = build(render=True, env_idx=k)
  wrapped2, policy2 = make_policy(env2)
  _, frames = rollout(env2, wrapped2, policy2, render=True)
  env2.close()

  frames = [f for f in frames if f is not None]
  Path(OUT).parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(OUT, frames, fps=30, quality=8)
  print(f"wrote {len(frames)} frames -> {OUT}")


if __name__ == "__main__":
  main()
