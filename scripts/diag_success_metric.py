"""Why does `Metrics/lift_height/episode_success` read 0.0000 for a policy that
scores 72.7% on `scripts/eval_deploy.py`?

  uv run python scripts/diag_success_metric.py TASK CKPT [teacher] [std=A,B,C]

Runs the policy in the TRAINING env config (corruption on, terminations on,
1000-step episodes) and separates the two effects that both push the logged
number toward zero.

1. THE EXECUTION NOISE. DAgger executes ``student(obs, stochastic_output=True)``,
   not the mean, so training rollouts carry the student's action noise while
   ``eval_deploy.py`` scores the deterministic policy. Sweeping the std shows
   how much of the gap that alone accounts for.

2. THE METRIC DEFINITION. Two different success criteria are in play:

     cube_lifted   TERMINATION: cube >= LIFT_HEIGHT above the table, held
                   HOLD_S. Height only. This is what eval_deploy.py scores.
     at_goal       METRIC: 3D ||goal - cube|| < success_threshold, where the
                   goal sits over the cube's SPAWN at a height sampled per
                   episode in [10, 20] cm.

   `cube_lifted` ENDS the episode the moment the cube has been 10 cm up for 1 s.
   Whenever the goal was sampled above ~15 cm the episode is terminated by
   success before the cube can get within 5 cm of the goal, so the metric can
   never latch. The per-goal-height split at the bottom isolates this.
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  HOLD_S,
  LIFT_HEIGHT,
  TABLE_H,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
ROLE = "teacher" if "teacher" in sys.argv else "student"
STDS = [0.0]
for arg in sys.argv[3:]:
  if arg.startswith("std="):
    STDS = [float(s) for s in arg[4:].split(",")]
N, STEPS, DEV = 256, 1000, "cuda:0"

# TRAINING config, not play: this is the configuration whose metric reads zero.
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
a = load_rl_cfg(TASK)
wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
if ROLE == "teacher":
  runner.load(CKPT, load_cfg={"teacher": True, "iteration": False}, map_location=DEV)
  runner.alg.eval_mode()
  policy = runner.alg.teacher
else:
  runner.load(CKPT, load_cfg={"student": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)

cmd = env.command_manager.get_term("lift_height")
cube = env.scene["cube"]
thresh = cmd.cfg.success_threshold


def rollout(std: float):
  """One full sweep at a given execution noise. std=0 takes the mean action."""
  if std > 0:
    with torch.no_grad():
      policy.distribution.std_param.fill_(std)
  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  peak = torch.zeros(N, device=DEV)
  at_goal = torch.zeros(N, device=DEV)
  goal = (cmd.target_pos[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone()
  out = {"peak": [], "goal_hit": [], "goal_h": []}
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs, stochastic_output=True) if std > 0 else policy(obs)
    obs, _, dones, _ = wrapped.step(act)
    h = cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - TABLE_H
    peak = torch.maximum(peak, h)
    at_goal = torch.maximum(at_goal, cmd.metrics["at_goal"])
    d = dones.bool()
    if d.any():
      out["peak"].append(peak[d].clone())
      out["goal_hit"].append(at_goal[d].clone())
      out["goal_h"].append(goal[d].clone())
      peak[d] = 0.0
      at_goal[d] = 0.0
      goal[d] = (cmd.target_pos[d, 2] - env.scene.env_origins[d, 2] - TABLE_H).clone()
  return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


print(f"task={TASK}\npolicy={CKPT} ({ROLE})")
print(f"{N} envs x {STEPS} steps in the TRAINING config")
print(
  f"success_threshold = {thresh} m (3D), LIFT_HEIGHT = {LIFT_HEIGHT} m, "
  f"hold = {HOLD_S} s\n"
)
print(
  f"{'exec std':>9} {'episodes':>9} {'lifted 10cm':>12} {'metric':>8} "
  f"{'median peak':>12}"
)
results = {}
for std in STDS:
  r = rollout(std)
  results[std] = r
  lifted = (r["peak"] >= LIFT_HEIGHT).mean()
  label = "0 (mean)" if std == 0 else f"{std:.3f}"
  print(
    f"{label:>9} {len(r['peak']):9d} {lifted * 100:11.1f}% "
    f"{r['goal_hit'].mean() * 100:7.1f}% {np.median(r['peak']) * 1000:11.1f}mm"
  )
print("\n  'lifted 10cm' is what eval_deploy.py scores; 'metric' is what the")
print("  training log prints as Metrics/lift_height/episode_success.")

# Effect 2, isolated at the lowest noise so it is not confounded by effect 1.
r = results[min(STDS)]
reachable = r["goal_h"] <= LIFT_HEIGHT + thresh
print(f"\nmetric split by sampled goal height (at exec std {min(STDS)}):")
for label, sel in (
  (f"goal <= {(LIFT_HEIGHT + thresh) * 100:.0f}cm", reachable),
  (f"goal >  {(LIFT_HEIGHT + thresh) * 100:.0f}cm", ~reachable),
):
  if sel.sum():
    print(
      f"  {label:>12}: lifted {(r['peak'][sel] >= LIFT_HEIGHT).mean() * 100:5.1f}%"
      f"   metric {r['goal_hit'][sel].mean() * 100:5.1f}%   n={sel.sum()}"
    )
print("  identical lifting, different metric => the metric is measuring the")
print("  goal draw, not the policy.")
env.close()
