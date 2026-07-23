"""Evaluation protocol for the Stage-1 compliance task (plan §7).

Computes the five headline metrics at the final (level-3) displacement
distribution, for either the analytic baseline or a trained checkpoint:

  1. Yield ratio      = arm displacement along u during the push / d
                        (0 = hard brace, 1 = fully compliant)
  2. Peak ‖F_ext‖ (N)
  3. Recovery time (s) = release -> re-success
  4. K anisotropy      = K∥ / K⊥ in the pushing window  (< 1 is the target)
  5. Success rate + episode length

Analytic baseline (E1, zero action = pure impedance):
  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.eval --analytic

Trained policy (after E2):
  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.eval \
    --checkpoint logs/rsl_rl/compliance_reach_flexiv/<run>/model_5000.pt
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.compliance.mdp.curriculum import _LEVELS
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_TASK = "Mjlab-Compliance-Reach-Flexiv"


def _stats(x: list[float]) -> str:
  if not x:
    return "n/a"
  a = np.array(x)
  return (
    f"mean={a.mean():.3f}  median={np.median(a):.3f}  std={a.std():.3f}  n={len(a)}"
  )


def _resolve_ckpt(path: str) -> Path:
  p = Path(path)
  if p.is_dir():
    ckpts = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    assert ckpts, f"no model_*.pt in {p}"
    return ckpts[-1]
  return p


def main(
  checkpoint: str | None = None,
  analytic: bool = False,
  task: str = _TASK,
  n_episodes: int = 200,
  num_envs: int = 64,
  device: str = "cpu",
  seed: int = 0,
  early_stop: bool = False,
) -> None:
  if checkpoint is None and not analytic:
    raise SystemExit("Pass --analytic or --checkpoint <path>.")

  # Build the env from the registered task so ablation variants (scalar-K env,
  # MLP policy, no-curriculum) all evaluate through the same code path.
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  cfg.curriculum = {}  # eval is always at fixed level 3 (plan §6/§7)
  cfg.seed = seed
  cfg.observations["actor"].enable_corruption = False
  # By default run full-length episodes so every episode experiences its push
  # (a fast policy would otherwise succeed and exit before t_push, starving the
  # yield/anisotropy statistics). Success is tracked manually below either way.
  if not early_stop:
    cfg.terminations.pop("success", None)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)

  human = env.event_manager.get_term_cfg("human_disturbance").func
  reach = env.command_manager.get_term("reach")
  impedance = env.action_manager.get_term("impedance")
  human.d_range = _LEVELS[3]

  # Policy: analytic (zero action) or a loaded checkpoint.
  if analytic:
    tag = "analytic baseline"
    wrapped = None

    def act(_obs):
      return torch.zeros(num_envs, impedance.action_dim, device=device)

    def reset_policy(_dones):
      pass

    obs, _ = env.reset()
  else:
    ckpt = _resolve_ckpt(checkpoint)  # type: ignore[arg-type]
    tag = f"{task} / {ckpt.name}"
    agent_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task)
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    def act(obs):
      with torch.no_grad():
        return policy(obs)

    def reset_policy(dones):
      if hasattr(policy, "reset"):
        policy.reset(dones)

    obs = wrapped.get_observations()

  def step(action):
    # Returns (obs, done, success). `success` = ended by the goal-reached
    # termination (not timeout); read from the termination type because
    # auto_reset clears episode_success before we could read it post-step.
    if wrapped is None:
      o, _, term, trunc, _ = env.step(action)
      return env.observation_manager.compute(), (term | trunc), term.bool()
    o, _, dones, extras = wrapped.step(action)
    timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
    return o, dones.bool(), dones.bool() & ~timeout

  n, dev = num_envs, device
  x_start = torch.zeros(n, 3, device=dev)
  u_dir = torch.zeros(n, 3, device=dev)
  d_mag = torch.zeros(n, device=dev)
  peak_f = torch.zeros(n, device=dev)
  k_par_sum = torch.zeros(n, device=dev)
  k_perp_sum = torch.zeros(n, device=dev)
  push_steps = torch.zeros(n, device=dev)
  was_pushing = torch.zeros(n, dtype=torch.bool, device=dev)
  release_t = torch.full((n,), -1.0, device=dev)
  t_env = torch.zeros(n, device=dev)
  ever_success = torch.zeros(n, dtype=torch.bool, device=dev)

  yields: list[float] = []
  peaks: list[float] = []
  recoveries: list[float] = []
  anisotropies: list[float] = []
  successes: list[float] = []
  ep_lengths: list[float] = []
  # Per-step, push-direction-aligned stiffness pooled over all pushing steps.
  # More robust than the per-episode ratio-of-sums in ``anisotropies`` (whose
  # heavy tail biases the median toward 1); this is the honest headline number.
  n_onsets = 0  # pushes that started (denominator for loop completion)
  step_kpar_sum = 0.0
  step_kperp_sum = 0.0
  step_ratios: list[float] = []

  step_i, max_steps = 0, n_episodes * 40
  while len(successes) < n_episodes and step_i < max_steps:
    step_i += 1
    action = act(obs)
    obs, dones, succ = step(action)
    t_env += env.step_dt

    pushing = human.is_pushing()
    ee = reach.ee_pos_w()
    f = torch.norm(human.external_force(), dim=-1)
    k = impedance.stiffness

    onset = pushing & ~was_pushing
    n_onsets += int(onset.sum())
    if onset.any():
      x_start[onset] = ee[onset]
      u_dir[onset] = human.push_dir()[onset]
      d_mag[onset] = torch.norm(human._x_target[onset] - human._x_start[onset], dim=-1)
      peak_f[onset] = 0.0
      k_par_sum[onset] = 0.0
      k_perp_sum[onset] = 0.0
      push_steps[onset] = 0.0

    if pushing.any():
      peak_f = torch.where(pushing, torch.maximum(peak_f, f), peak_f)
      k_par = (u_dir**2 * k).sum(dim=-1)
      k_perp = (k.sum(dim=-1) - k_par) / 2.0
      k_par_sum = torch.where(pushing, k_par_sum + k_par, k_par_sum)
      k_perp_sum = torch.where(pushing, k_perp_sum + k_perp, k_perp_sum)
      push_steps = torch.where(pushing, push_steps + 1.0, push_steps)
      step_kpar_sum += float(k_par[pushing].sum())
      step_kperp_sum += float(k_perp[pushing].sum())
      step_ratios.extend((k_par[pushing] / k_perp[pushing].clamp(min=1e-6)).tolist())

    release = ~pushing & was_pushing
    for i in release.nonzero(as_tuple=False).squeeze(-1).tolist():
      if d_mag[i] < 1e-4 or push_steps[i] < 1:
        continue
      yields.append(torch.dot(ee[i] - x_start[i], u_dir[i]).item() / d_mag[i].item())
      peaks.append(peak_f[i].item())
      anisotropies.append((k_par_sum[i] / k_perp_sum[i].clamp(min=1e-6)).item())
      release_t[i] = t_env[i]

    at_goal = reach.success()
    ever_success |= at_goal

    rec = (release_t >= 0) & at_goal
    for i in rec.nonzero(as_tuple=False).squeeze(-1).tolist():
      recoveries.append((t_env[i] - release_t[i]).item())
      release_t[i] = -1.0

    was_pushing = pushing.clone()

    # With early_stop the env terminates on success (`succ`); otherwise episodes
    # only end on timeout and success is the manually-tracked `ever_success`.
    if dones.any():
      for i in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
        successes.append(float((succ[i] | ever_success[i]).item()))
        ep_lengths.append(t_env[i].item())
      t_env[dones] = 0.0
      release_t[dones] = -1.0
      was_pushing[dones] = False
      ever_success[dones] = False
      reset_policy(dones)

  print(f"\n==== Stage-1 compliance evaluation ({tag}, level 3) ====")
  print(f"episodes: pushes={len(yields)}  full-episodes={len(successes)}")
  print(f"1. Yield ratio        : {_stats(yields)}")
  print(f"2. Peak |F_ext| (N)   : {_stats(peaks)}")
  print(f"3. Recovery time (s)  : {_stats(recoveries)}")
  print(f"4. K anisotropy K∥/K⊥ : {_stats(anisotropies)}  (per-episode; biased)")
  pooled = step_kpar_sum / max(step_kperp_sum, 1e-6)
  print(
    f"   K∥/K⊥ per-step       : pooled={pooled:.3f}  "
    f"median={np.median(step_ratios) if step_ratios else float('nan'):.3f}  "
    f"(<1 = softer along push; honest headline)"
  )
  # Loop completion: of all pushes that STARTED, how many ended with the arm
  # released and back at the goal.  Success rate alone is misleading -- an arm
  # that just braces at the goal under load scores a perfect success rate while
  # never yielding at all (e11/e12).
  rel_rate = len(yields) / max(n_onsets, 1)
  loop_rate = len(recoveries) / max(n_onsets, 1)
  print(
    f"6. Loop completion    : {loop_rate:.3f}  (released+returned / {n_onsets} pushes)"
    f"   release-rate={rel_rate:.3f}"
  )
  s = np.array(successes) if successes else np.array([0.0])
  print(f"5. Success rate       : {s.mean():.3f}  (n={len(successes)})")
  print(f"   Episode length (s) : {_stats(ep_lengths)}")
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
