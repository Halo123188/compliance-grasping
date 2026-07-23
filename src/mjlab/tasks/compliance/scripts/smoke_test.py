"""Headless functional smoke test for the Stage-1 compliance env (plan §8 E0).

Builds the env, runs zero-action rollouts (pure analytical impedance), and prints:
  * EE-to-goal distance over time (should settle low when undisturbed),
  * human force profile (should ramp, peak <= F_max, no spikes),
  * NaN / divergence guards.

Run:  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.smoke_test
"""

from __future__ import annotations

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.compliance.config.flexiv.env_cfg import flexiv_reach_env_cfg


def main(num_envs: int = 8, steps: int = 500, device: str = "cpu") -> None:
  cfg = flexiv_reach_env_cfg()
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=device)

  human = env.event_manager.get_term_cfg("human_disturbance").func
  reach = env.command_manager.get_term("reach")

  obs, _ = env.reset()
  n_act = env.action_manager.total_action_dim
  print(f"obs actor dim: {obs['actor'].shape}, action dim: {n_act}")

  peak_force = 0.0
  max_dist = 0.0
  for t in range(steps):
    action = torch.zeros(num_envs, n_act, device=device)
    obs, rew, term, trunc, info = env.step(action)

    f = torch.norm(human.external_force(), dim=-1)
    dist = torch.norm(reach.command - reach.ee_pos_w(), dim=-1)
    peak_force = max(peak_force, float(f.max()))
    max_dist = max(max_dist, float(dist.max()))

    if torch.isnan(obs["actor"]).any():
      print(f"[FAIL] NaN in obs at step {t}")
      return
    if t % 50 == 0:
      print(
        f"t={t:3d}  dist[min/mean/max]="
        f"{dist.min():.3f}/{dist.mean():.3f}/{dist.max():.3f}  "
        f"F[max]={f.max():6.2f}N  phase={human._phase.tolist()}"
      )

  print(f"\npeak human force over run: {peak_force:.2f} N  (cap 40 N)")
  print(f"max EE-to-goal distance:   {max_dist:.3f} m")
  print("OK: no NaNs, ran to completion.")


if __name__ == "__main__":
  tyro.cli(main)
