"""Classify WHY episodes fail, then render one representative failure.

  uv run python scripts/diag_failure_modes.py TASK CKPT OUT_PREFIX

`render_faceon.py` always picks the BEST episode, which is the wrong sample when
the question is why 64% of them fail. This rolls out, classifies every env by
what went wrong, prints the distribution, and then renders one env drawn from the
biggest failure bucket -- so the video is representative rather than cherry-
picked from the tail.

Buckets, decided from the per-step traces rather than the final state:

  never-gripped   both fingertips never in contact with the cube at the same time
  knocked-away    the cube moved > 40 mm horizontally while never being gripped
  slipped-early   gripped, but the cube never got more than 20 mm off the table
  dropped         got above 20 mm, then contact was lost and it fell back
  short-lift      held all the way up but peaked below the 100 mm bar
  hold-timeout    crossed 100 mm but not for the full 1 s
  success         crossed 100 mm and held 1 s
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
N, STEPS, DEV = 64, 300, "cuda:0"
HOLD_STEPS = 50  # 1 s at 50 Hz


def build(render: bool, env_idx: int = 0):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  if render:
    cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="cube",
      env_idx=env_idx,
      distance=0.30,
      elevation=-12.0,
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


def rollout(env, wrapped, policy, render: bool, env_idx: int = 0):
  torch.manual_seed(0)
  robot, cube = env.scene["robot"], env.scene["cube"]
  names = list(robot.site_names)
  li, ri = names.index("left_pad"), names.index("right_pad")
  cam = env._offline_renderer._cam if render else None  # noqa: SLF001
  obs = wrapped.reset()[0]
  H, G, XY, YAW, frames = [], [], [], [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    pos = cube.data.root_link_pos_w
    H.append((pos[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone())
    fl = torch.norm(
      env.scene["left_cube_contact"].data.force.reshape(N, -1, 3), dim=-1
    ).amax(-1)
    fr = torch.norm(
      env.scene["right_cube_contact"].data.force.reshape(N, -1, 3), dim=-1
    ).amax(-1)
    G.append((torch.minimum(fl, fr) > 0.1).clone())
    XY.append(pos[:, :2].clone())
    pads = robot.data.site_pos_w[:, [li, ri]]
    ax = pads[:, 1] - pads[:, 0]
    jaw = torch.atan2(ax[:, 1], ax[:, 0])
    q = cube.data.root_link_quat_w
    cy = torch.atan2(
      2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    d = torch.rad2deg((jaw - cy) % (torch.pi / 2))
    YAW.append(torch.minimum(d, 90.0 - d).clone())
    if render:
      p = robot.data.site_pos_w
      dd = (p[env_idx, ri] - p[env_idx, li]).tolist()
      cam.azimuth = math.degrees(math.atan2(dd[1], dd[0])) + 90.0
      frames.append(env.render())
  return (
    torch.stack(H),
    torch.stack(G),
    torch.stack(XY),
    torch.stack(YAW),
    frames,
  )


env = build(render=False)
wrapped, policy = make_policy(env)
H, G, XY, YAW, _ = rollout(env, wrapped, policy, render=False)
env.close()

peak = H.max(dim=0).values
ever = G.any(dim=0)
drift = torch.norm(XY - XY[0], dim=-1).max(dim=0).values
# Longest run of consecutive steps above the bar
above = H > LIFT_HEIGHT
best_run = torch.zeros(N, dtype=torch.long)
cur = torch.zeros(N, dtype=torch.long)
for t in range(STEPS):
  cur = torch.where(above[t].cpu(), cur + 1, torch.zeros_like(cur))
  best_run = torch.maximum(best_run, cur)
# Was it still gripped at its highest point?
top = H.argmax(dim=0)
grip_at_top = G[top, torch.arange(N, device=G.device)]

buckets: dict[str, list[int]] = {
  k: []
  for k in (
    "success",
    "hold-timeout",
    "short-lift",
    "dropped",
    "slipped-early",
    "knocked-away",
    "never-gripped",
  )
}
for i in range(N):
  if best_run[i] >= HOLD_STEPS:
    buckets["success"].append(i)
  elif peak[i] > LIFT_HEIGHT:
    buckets["hold-timeout"].append(i)
  elif not ever[i]:
    buckets["knocked-away" if drift[i] > 0.040 else "never-gripped"].append(i)
  elif peak[i] < CUBE_HALF + 0.020:
    buckets["slipped-early"].append(i)
  elif not grip_at_top[i]:
    buckets["dropped"].append(i)
  else:
    buckets["short-lift"].append(i)

print(f"checkpoint: {CKPT}")
print(f"{N} envs, {STEPS} steps. Bar = {LIFT_HEIGHT:.2f} m for {HOLD_STEPS} steps.\n")
print(f"{'bucket':>15} {'n':>4} {'%':>6} {'median peak':>12} {'median yaw err':>15}")
for k, idx in buckets.items():
  if not idx:
    print(f"{k:>15} {0:4d} {0.0:6.1f}")
    continue
  t = torch.tensor(idx)
  print(
    f"{k:>15} {len(idx):4d} {100 * len(idx) / N:6.1f}"
    f" {peak[t].median():12.4f} {YAW[:, t].median():15.1f}"
  )

order = [
  "dropped",
  "short-lift",
  "slipped-early",
  "never-gripped",
  "knocked-away",
  "hold-timeout",
]
pick_bucket = max((k for k in order if buckets[k]), key=lambda k: len(buckets[k]))
t = torch.tensor(buckets[pick_bucket])
pick = int(t[peak[t].argsort()[len(t) // 2]])  # median example of that bucket
print("\nrendering a MEDIAN example of the biggest failure bucket:")
print(
  f"  bucket '{pick_bucket}', env {pick}, peak {peak[pick]:.4f} m,"
  f" ever gripped {bool(ever[pick])}, drift {drift[pick] * 1000:.1f} mm,"
  f" median yaw err {YAW[:, pick].median():.1f} deg"
)

Path(PREFIX).parent.mkdir(parents=True, exist_ok=True)
e = build(render=True, env_idx=pick)
w, p = make_policy(e)
*_, frames = rollout(e, w, p, render=True, env_idx=pick)
e.close()
frames = [f for f in frames if f is not None]
out = f"{PREFIX}_fail_{pick_bucket}.mp4"
imageio.mimwrite(out, frames, fps=30, quality=8)
print(f"wrote {len(frames)} frames -> {out}")
