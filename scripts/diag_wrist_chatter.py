"""Does the closed-form wrist solver chatter at the fold boundary?

  uv run python scripts/diag_wrist_chatter.py

Run 49073 (SolvedWrist) trained to `grasp` 0.0000 and 0% success even though the
solver demonstrably squares the jaw (2.8 deg mean error under random actions vs
24.6 deg for the baseline). Its `action_rate_l2` came out -0.0871 against the
baseline's -0.0245, which points at the wrist moving violently rather than at the
alignment being wrong.

The suspect is the +-45 deg fold. The target is `q7 + fold(jaw - cube)`, a
deadbeat law with no hysteresis, so an error sitting near the boundary flips
branch and commands a 90 deg swing -- and then flips back. This measures the
per-step wrist motion directly instead of inferring it.
"""

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

N, STEPS, DEV = 64, 120, "cuda:0"
TASKS = {
  "baseline": "Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalmAlign",
  "solved": "Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-SolvedWrist",
}
BIG = 20.0  # deg of wrist motion in one control step that counts as a slam


def run(task: str):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  robot = env.scene["robot"]
  j7 = list(robot.joint_names).index("joint7")

  torch.manual_seed(0)
  env.reset()
  dim = env.action_manager.total_action_dim
  qs = []
  for _ in range(STEPS):
    with torch.inference_mode():
      env.step(0.3 * torch.randn(N, dim, device=DEV))
    qs.append(robot.data.joint_pos[:, j7].clone())
  env.close()
  q = np.degrees(torch.stack(qs).cpu().numpy())
  d = np.abs(np.diff(q, axis=0))
  return q, d


print(f"\nRandom actions, {N} envs, {STEPS} steps. joint7 motion per control step.\n")
print(
  f"{'task':>10} {'|dq7| mean':>11} {'p95':>8} {'max':>8}"
  f" {'steps >20deg':>13} {'envs affected':>14}"
)
for name, task in TASKS.items():
  _, d = run(task)
  slam = d > BIG
  print(
    f"{name:>10} {d.mean():10.2f}d {np.percentile(d, 95):7.2f}d {d.max():7.2f}d"
    f" {slam.mean() * 100:12.2f}% {(slam.any(0)).mean() * 100:13.1f}%"
  )

print(
  "\nA fold-boundary flip commands a ~90 deg swing. If 'solved' shows slams that"
  "\nthe baseline does not, the deadbeat law needs hysteresis or a lower gain."
)
