"""Two checks: does the cube reset mid-rollout, and are two rollouts identical?"""

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import TABLE_H
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, STEPS, DEV = 16, 300, "cuda:0"


def run():
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(TASK)
  w = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  r = load_runner_cls(TASK)(w, asdict(a), device=DEV)
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
  pol = r.get_inference_policy(device=DEV)
  torch.manual_seed(0)
  obs = w.reset()[0]
  P, H = [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = pol(obs)
    obs, _, _, _ = w.step(act)
    p = env.scene["cube"].data.root_link_pos_w.clone()
    P.append(p)
    H.append((p[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone())
  print("episode_length_s =", cfg.episode_length_s)
  env.close()
  return torch.stack(P).cpu(), torch.stack(H).cpu()


P1, H1 = run()
P2, H2 = run()
# 1. mid-rollout teleports: a jump larger than anything physics could do in 20 ms
jump = (P1[1:] - P1[:-1]).norm(dim=-1)  # [T-1, N]
big = jump > 0.05
print(f"\nCHECK 1  cube teleports > 50 mm in one 20 ms step: {int(big.sum())} events")
if big.any():
  t, e = big.nonzero()[0].tolist()
  print(f"  first at step {t}, env {e}: {jump[t, e] * 1000:.0f} mm")
print(f"  start pos env0 {P1[0, 0].tolist()}  end pos env0 {P1[-1, 0].tolist()}")

# 2. are the two rollouts the same episode?
same_start = torch.allclose(P1[0], P2[0], atol=1e-6)
print(f"\nCHECK 2  two rollouts start from identical cube poses: {same_start}")
print(
  f"  max |peak1 - peak2| over envs: "
  f"{(H1.max(0).values - H2.max(0).values).abs().max() * 1000:.1f} mm"
)
for i in range(4):
  print(
    f"  env {i}: peak1 {H1[:, i].max() * 1000:6.1f} mm   peak2 {H2[:, i].max() * 1000:6.1f} mm"
  )
