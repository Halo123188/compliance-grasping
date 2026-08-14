"""Physics throughput at training scale, for whatever collision geometry is set.

  CG_HAND_COLLISION=coacd_t0.2 uv run python scripts/wide_throughput.py 4096

The one cost of switching the hand's collision representation that cannot be
read off a mesh: box-box is a dedicated cheap primitive in the narrowphase while
convex hulls go through a general path, so the price is per-CONTACT-PAIR rather
than per-geom and no amount of counting hulls predicts it.

Steps the real training task with random actions -- the same env, scene and
substep count PPO would drive -- and reports environment-steps/second after a
warmup that covers the mujoco-warp JIT.
"""

import sys
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-FaceLevel"
NENV = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
WARMUP, MEASURE = 30, 120

cfg = load_env_cfg(TASK)
cfg.scene.num_envs = NENV
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
torch.manual_seed(0)
env.reset()
dim = env.action_manager.total_action_dim


def run(n: int) -> float:
  t0 = time.perf_counter()
  for _ in range(n):
    env.step(torch.rand(NENV, dim, device="cuda:0") * 2.0 - 1.0)
  torch.cuda.synchronize()
  return time.perf_counter() - t0


run(WARMUP)  # JIT + cache warm; mujoco-warp compiles kernels on first use
dt = run(MEASURE)
print(
  f"  {NENV} envs x {MEASURE} steps in {dt:.2f} s"
  f"  ->  {NENV * MEASURE / dt / 1e3:.1f}k env-steps/s"
  f"  ({dt / MEASURE * 1e3:.1f} ms/step)"
)
env.close()
