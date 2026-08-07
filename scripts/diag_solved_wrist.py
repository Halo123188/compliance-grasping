"""Does the closed-form wrist solver actually square the jaw? Check before training.

  uv run python scripts/diag_solved_wrist.py

Rolls out RANDOM actions -- no policy -- on the solved-wrist task and on the
plain baseline task, and reports the jaw-vs-cube yaw error over time. The solver
runs inside the action term, so it should hold the error near zero from the first
step regardless of what the policy does. If it does not, the training run would
be measuring a broken solver.

Under uniform cube yaw a frozen wrist averages 22.5 deg of error, and the probe
measured 14.1 deg for the trained baseline (the fingers rotate the cube square on
contact). Anything the solver produces should be far below both.
"""

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

N, STEPS, DEV = 64, 60, "cuda:0"
TASKS = {
  "baseline": "Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalmAlign",
  "solved": "Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-SolvedWrist",
}


def run(task: str):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  robot, cube = env.scene["robot"], env.scene["cube"]
  sn = list(robot.site_names)
  li, ri = sn.index("left_pad"), sn.index("right_pad")
  j7 = list(robot.joint_names).index("joint7")

  torch.manual_seed(0)
  env.reset()
  dim = env.action_manager.total_action_dim
  errs, j7s = [], []
  for _ in range(STEPS):
    act = 0.3 * torch.randn(N, dim, device=DEV)
    with torch.inference_mode():
      env.step(act)
    pads = robot.data.site_pos_w[:, [li, ri]]
    ax = pads[:, 1] - pads[:, 0]
    jaw = torch.atan2(ax[:, 1], ax[:, 0])
    q = cube.data.root_link_quat_w
    cy = torch.atan2(
      2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    e = torch.rad2deg(jaw - cy) % 90.0
    errs.append(torch.minimum(e, 90.0 - e).clone())
    j7s.append(robot.data.joint_pos[:, j7].clone())
  env.close()
  return torch.stack(errs).cpu().numpy(), np.degrees(torch.stack(j7s).cpu().numpy())


out = {k: run(v) for k, v in TASKS.items()}

print(f"\nRandom actions, {N} envs. Jaw-vs-cube yaw error, folded into [0, 45] deg.")
print(f"\n{'step':>6}" + "".join(f"{k:>20}" for k in TASKS))
for t in (0, 1, 2, 5, 10, 20, 40, STEPS - 1):
  row = "".join(
    f"{out[k][0][t].mean():13.1f}d +-{out[k][0][t].std():4.1f}" for k in TASKS
  )
  print(f"{t:>6}{row}")

print(f"\n{'':>6}" + "".join(f"{k:>20}" for k in TASKS))
for label, fn in (
  ("mean err", lambda a: a[0].mean()),
  ("joint7 std", lambda a: a[1].std()),
):
  print(f"{label:>10}" + "".join(f"{fn(out[k]):15.1f}d" for k in TASKS))
print("\nSolver is working if 'solved' collapses to a few deg within a step or two")
print("and joint7's spread is LARGE (it now tracks cube yaw instead of parking).")
