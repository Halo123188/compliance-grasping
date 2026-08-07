"""What does a distillation behaviour loss of 1.0 actually mean?

  uv run python scripts/diag_teacher_actions.py TASK CKPT

MSE is only interpretable against the variance of what is being predicted. A
student that has learned NOTHING converges to the teacher's mean action, and
scores exactly the teacher's action variance. So this rolls the teacher out and
reports that variance, per dimension and averaged -- the number the training
log's "Mean behavior loss" has to be read against.

Also reports the variance of the residual after subtracting a per-timestep mean,
which is roughly what a student could reach using proprioception alone (i.e.
without ever locating the cube).
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = sys.argv[1]
CKPT = sys.argv[2]
N, STEPS, DEV = 256, 200, "cuda:0"

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
a = load_rl_cfg(TASK)
wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
runner.load(CKPT, load_cfg={"teacher": True, "iteration": False}, map_location=DEV)
runner.alg.eval_mode()
teacher = runner.alg.teacher

torch.manual_seed(0)
obs = wrapped.reset()[0]
acts = []
for _ in range(STEPS):
  with torch.inference_mode():
    act = teacher(obs)
  acts.append(act.clone())
  obs = wrapped.step(act)[0]
A = torch.stack(acts).cpu().numpy()  # (T, N, dim)
env.close()

names = list(env.action_manager.get_term("joint_pos").target_names)
var_total = A.var(axis=(0, 1))
# Residual after removing the per-timestep cross-env mean: what is left is the
# part that depends on where the cube is, i.e. the part vision has to supply.
var_resid = (A - A.mean(axis=1, keepdims=True)).var(axis=(0, 1))

print(f"task={TASK}\nteacher={CKPT}\n{N} envs x {STEPS} steps\n")
print(f"{'joint':>10} {'mean':>8} {'std':>8} {'var':>8} {'var|t':>8}")
for i, n in enumerate(names):
  print(
    f"{n:>10} {A[..., i].mean():8.3f} {A[..., i].std():8.3f} "
    f"{var_total[i]:8.3f} {var_resid[i]:8.3f}"
  )
print(
  f"\nMSE of a constant (mean-action) student : {var_total.mean():.4f}"
  f"\nMSE floor for a cube-blind student      : {var_resid.mean():.4f}"
  f"\n  -- a training loss at or above the first number means the student has"
  f"\n     learned nothing; between the two means it uses proprioception only."
)
print(f"\naction range: [{A.min():.2f}, {A.max():.2f}]")
print(f"fraction of |action| > 3: {np.mean(np.abs(A) > 3):.4f}")
