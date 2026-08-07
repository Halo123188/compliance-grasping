"""Wide fixed-camera rollout, so the LIFT is actually visible.

  uv run python scripts/render_wide.py TASK CKPT OUT_PREFIX [ENV_IDX]

``render_faceon.py`` and ``diag_failure_modes.py`` both park the camera on the
cube (``origin_type=ASSET_ROOT``), which is the right view for reading the jaw
but the wrong one for reading height: the cube stays dead centre no matter how
far it rises, so a 91 mm short-lift and a 107 mm success look identical.

This uses a fixed WORLD camera framing the table, plus a drawn reference line at
the 100 mm success bar, so how high the cube actually gets is readable directly.
Renders a success and a failure side by side when both are available.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  LIFT_HEIGHT,
  TABLE_H,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

TASK, CKPT, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
N, STEPS, DEV = 64, 300, "cuda:0"
HOLD_STEPS = 50
W, H_PX = 960, 720


def build(render: bool, env_idx: int = 0):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  if render:
    # Track the ROBOT root, not the world and not the cube. WORLD puts the
    # camera at the global origin while each env is offset, so it frames empty
    # floor; ASSET_ROOT on the cube keeps the cube centred and hides the very
    # thing being measured. The arm base is fixed, so tracking it gives a camera
    # that is stationary in the env frame and lets the cube rise through it.
    cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      env_idx=env_idx,
      distance=1.05,
      elevation=-10.0,
      azimuth=125.0,
      max_extra_envs=0,
      height=H_PX,
      width=W,
    )
  return ManagerBasedRlEnv(
    cfg=cfg, device=DEV, render_mode="rgb_array" if render else None
  )


def make_policy(env):
  a = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
  runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
  return wrapped, runner.get_inference_policy(device=DEV)


def rollout(env, wrapped, policy, render: bool):
  torch.manual_seed(0)
  cube = env.scene["cube"]
  obs = wrapped.reset()[0]
  hs, frames = [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    hs.append(
      (cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone()
    )
    if render:
      frames.append(env.render())
  return torch.stack(hs), frames


env = build(render=False)
wrapped, policy = make_policy(env)
Hh, _ = rollout(env, wrapped, policy, render=False)
env.close()

peak = Hh.max(dim=0).values
above = (Hh > LIFT_HEIGHT).cpu()
run = torch.zeros(N, dtype=torch.long)
cur = torch.zeros(N, dtype=torch.long)
for t in range(STEPS):
  cur = torch.where(above[t], cur + 1, torch.zeros_like(cur))
  run = torch.maximum(run, cur)
succ = (run >= HOLD_STEPS).nonzero().flatten().tolist()
fail = [i for i in range(N) if i not in succ]
print(f"success {len(succ)}/{N}; rendering one of each")

picks = []
if succ:
  s = torch.tensor(succ)
  picks.append(("SUCCESS", int(s[peak[s].argsort()[len(s) // 2]])))
if fail:
  f = torch.tensor(fail)
  picks.append(("FAILURE", int(f[peak[f].argsort()[len(f) // 2]])))

Path(PREFIX).parent.mkdir(parents=True, exist_ok=True)
try:
  font = ImageFont.truetype("DejaVuSans.ttf", 26)
  small = ImageFont.truetype("DejaVuSans.ttf", 20)
except OSError:
  font = small = ImageFont.load_default()

clips = []
for want, idx in picks:
  e = build(render=True, env_idx=idx)
  w, p = make_policy(e)
  hh, frames = rollout(e, w, p, render=True)
  e.close()
  frames = [fr for fr in frames if fr is not None]

  # Score THIS rollout, never the selection pass. Physics on GPU is not
  # bit-reproducible: two rollouts from identical cube poses with the same
  # deterministic policy diverge by up to 25.1 mm of peak height (measured,
  # scripts/_chk_rollout.py -- env 0 peaked at 111.1 mm on one pass and 100.5 mm
  # on the next). That straddles the 100 mm bar, so a label imported from the
  # selection pass can call a rendered success a failure, which is exactly what
  # the earlier clips did: live height topped out at 97.0 mm while the caption
  # read "peak 93.1 mm" from a different rollout.
  h_env = hh[:, idx]
  pk = float(h_env.max())
  ab = (h_env > LIFT_HEIGHT).cpu()
  r = c = 0
  for v in ab.tolist():
    c = c + 1 if v else 0
    r = max(r, c)
  tag = "SUCCESS" if r >= HOLD_STEPS else "FAILURE"
  note = "" if tag == want else f"  (selection pass said {want})"
  print(
    f"  env {idx}: render pass peak {pk * 1000:.1f} mm, held {r} steps -> {tag}{note}"
  )

  out = []
  for t, fr in enumerate(frames):
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im)
    h_mm = float(hh[t, idx]) * 1000
    ok = h_mm > LIFT_HEIGHT * 1000
    d.text((14, 12), f"{tag}  env {idx}", fill=(255, 255, 255), font=font)
    d.text(
      (14, 46),
      f"cube {h_mm:6.1f} mm   bar {LIFT_HEIGHT * 1000:.0f} mm",
      fill=(130, 255, 150) if ok else (255, 180, 120),
      font=small,
    )
    d.text(
      (14, 72),
      f"peak {pk * 1000:.1f} mm  held {r} of {HOLD_STEPS} steps",
      fill=(190, 190, 190),
      font=small,
    )
    # bar chart of height so the lift is unmistakable even without the number
    x0, y1 = 40, H_PX - 40
    scale = 1.6  # px per mm
    d.rectangle(
      [x0, y1 - LIFT_HEIGHT * 1000 * scale, x0 + 8, y1], outline=(120, 120, 120)
    )
    d.rectangle(
      [x0, y1 - max(h_mm, 0) * scale, x0 + 8, y1],
      fill=(130, 255, 150) if ok else (255, 180, 120),
    )
    out.append(np.asarray(im))
  clips.append(out)
  imageio.mimwrite(f"{PREFIX}_wide_{want.lower()}.mp4", out, fps=30, quality=8)
  print(f"wrote {PREFIX}_wide_{want.lower()}.mp4  peak {pk * 1000:.1f} mm -> {tag}")

if len(clips) == 2:
  n = min(len(clips[0]), len(clips[1]))
  side = [np.hstack([clips[0][i], clips[1][i]]) for i in range(n)]
  imageio.mimwrite(f"{PREFIX}_wide_sidebyside.mp4", side, fps=30, quality=8)
  print(f"wrote {PREFIX}_wide_sidebyside.mp4")
