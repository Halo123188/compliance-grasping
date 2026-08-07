"""Render the grasp square-on to the jaw, not from a fixed world angle.

  uv run python scripts/render_faceon.py TASK CKPT OUT_PREFIX [DIST]

``render_rollout.py`` uses a fixed world azimuth. That is unreadable here: cube
yaw is randomised over the full circle and the wrist rolls to match it, so the
jaw faces a different way every episode and a fixed camera lands somewhere
between edge-on and behind the hand.

The closing axis is not guessed from a body frame -- it is measured, every
frame, as the vector between the two pad sites. Looking ALONG that axis hides
one pad behind the other, so the face-on camera sits PERPENDICULAR to it
(azimuth + 90 degrees), which is the view where the two pads open and close
left-to-right across the image with the cube between them.

Writes <prefix>_faceon.mp4 for the best episode.

A steep top-down view was tried and dropped: at the close distance this camera
needs, tilting to -72 degrees puts the eye inside the cube, and most frames come
back solid white. If a top view is ever wanted, back the camera off first --
elevation and distance are not independent here.
"""

import math
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  CUBE_HALF,
  LIFT_HEIGHT,
  TABLE_H,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

TASK, CKPT, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
DIST = float(sys.argv[4]) if len(sys.argv) > 4 else 0.28
N, STEPS, SEED, DEV = 32, 300, 0, "cuda:0"

# Near-level: high enough to read the cube's tilt against the table, low enough
# that neither finger is foreshortened into the other.
ELEVATION = -12.0


def build(render: bool, env_idx: int = 0, elevation: float = -12.0):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}  # continuous rollout, no reset jumps mid-video
  if render:
    cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="cube",
      env_idx=env_idx,
      distance=DIST,
      elevation=elevation,
      max_extra_envs=0,
      height=720,
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


def pad_ids(env):
  names = list(env.scene["robot"].site_names)
  return names.index("left_pad"), names.index("right_pad")


def rollout(env, wrapped, policy, render: bool, env_idx: int = 0):
  """Roll out; when rendering, re-aim the camera at the live closing axis."""
  torch.manual_seed(SEED)
  li, ri = pad_ids(env)
  cam = env._offline_renderer._cam if render else None  # noqa: SLF001
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
      pads = env.scene["robot"].data.site_pos_w
      d = (pads[env_idx, ri] - pads[env_idx, li]).tolist()
      cam.azimuth = math.degrees(math.atan2(d[1], d[0])) + 90.0
      frames.append(env.render())
  return torch.stack(heights), frames


env = build(render=False)
wrapped, policy = make_policy(env)
heights, _ = rollout(env, wrapped, policy, render=False)
peak = heights.max(dim=0).values  # [N] best height each env reached
best = int(peak.argmax())

print(f"checkpoint: {CKPT}")
print(f"cube height above table top, peak over {STEPS} steps ({N} envs):")
print(f"  resting height   = {CUBE_HALF:.4f} m")
print(f"  best env ({best:2d})     = {peak[best]:.4f} m")
print(f"  median           = {peak.median():.4f} m")
print(f"  envs above +1cm  = {int((peak > CUBE_HALF + 0.01).sum())}/{N}")
print(
  f"  envs above {LIFT_HEIGHT:.2f}m  = {int((peak > LIFT_HEIGHT).sum())}/{N}  <- bar"
)
env.close()

Path(PREFIX).parent.mkdir(parents=True, exist_ok=True)
e = build(render=True, env_idx=best, elevation=ELEVATION)
w, p = make_policy(e)
_, frames = rollout(e, w, p, render=True, env_idx=best)
e.close()
frames = [f for f in frames if f is not None]
out = f"{PREFIX}_faceon.mp4"
imageio.mimwrite(out, frames, fps=30, quality=8)
print(f"wrote {len(frames)} frames -> {out}")
