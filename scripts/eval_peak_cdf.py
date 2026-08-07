"""Peak-height CDF: what fraction of envs ever get the cube above X mm.

  uv run python scripts/eval_peak_cdf.py NAME=TASK:CKPT [NAME=TASK:CKPT ...]

``eval_deploy.py`` reports one bar (100 mm held 1 s) plus a single near-miss
bucket [90, 100). That hides two different things. The AlignCurr failure clip
peaked at 104.4 mm and held only 2 of the required 50 steps, so it is a failure
that never enters the near-miss bucket either -- it cleared the bar and dropped.
A CDF over peak height separates "how high does it get" from "does it stay
there", which the single number cannot.

Prints, per threshold, the fraction of envs whose peak ever exceeded it, and
alongside it the fraction that ALSO held above 100 mm for the full second.
"""

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  LIFT_HEIGHT,
  TABLE_H,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

N, STEPS, DEV = 128, 300, "cuda:0"
HOLD = 50
THRESH_MM = [40, 50, 60, 70, 80, 90, 100, 110]


def evaluate(task: str, ckpt: str):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  # Distillation checkpoints carry the student, PPO ones the actor.
  is_distilled = "student_state_dict" in torch.load(
    ckpt, map_location="cpu", weights_only=False
  )
  runner.load(
    ckpt,
    load_cfg={"student": True} if is_distilled else {"actor": True},
    strict=True,
    map_location=DEV,
  )
  policy = runner.get_inference_policy(device=DEV)

  cube = env.scene["cube"]
  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  hs = []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    hs.append(
      (cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone()
    )
  h = torch.stack(hs).cpu()
  env.close()

  peak = h.max(0).values
  above = h > LIFT_HEIGHT
  run = torch.zeros(N, dtype=torch.long)
  cur = torch.zeros(N, dtype=torch.long)
  for t in range(STEPS):
    cur = torch.where(above[t], cur + 1, torch.zeros_like(cur))
    run = torch.maximum(run, cur)
  succ = run >= HOLD
  return peak, succ, run


runs = []
for spec in sys.argv[1:]:
  name, rest = spec.split("=", 1)
  task, ckpt = rest.split(":", 1)
  runs.append((name, *evaluate(task, ckpt)))

print(f"\n{N} envs, {STEPS} steps, deterministic policy. Cube rests at 25 mm.\n")
hdr = "".join(f"{t:>8}" for t in THRESH_MM)
print(f"{'run':>14}  fraction with PEAK above (mm):{hdr}")
for name, peak, _, _ in runs:
  row = "".join(
    f"{float((peak > t / 1000).float().mean()) * 100:7.1f}%" for t in THRESH_MM
  )
  print(f"{name:>14}  {'':>29}{row}")

print(f"\n{'run':>14} {'peak>80mm':>10} {'success':>9} {'80mm but no hold':>18}")
for name, peak, succ, _ in runs:
  p80 = peak > 0.080
  print(
    f"{name:>14} {float(p80.float().mean()) * 100:9.1f}%"
    f" {float(succ.float().mean()) * 100:8.1f}%"
    f" {float((p80 & ~succ).float().mean()) * 100:17.1f}%"
  )

print(f"\n{'run':>14} {'median peak':>12} {'median hold (of 50)':>21}")
for name, peak, _, run in runs:
  print(
    f"{name:>14} {float(peak.median()) * 1000:11.1f}mm {float(run.float().median()):20.0f}"
  )
