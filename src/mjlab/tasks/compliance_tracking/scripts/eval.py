"""Evaluation for the compliance-tracking task (spec §3.4 diagnostics).

Runs a fixed seeded batch of episodes and reports the diagnostics that actually
distinguish the two ways of scoring well on a tracking reward:

  * tracking error split by regime — unperturbed / during a pull / post-release /
    post-grasp.  The aggregate is dominated by unperturbed steps and hides
    everything interesting.
  * commanded ``K_∥`` vs ``K_⊥`` about the pull direction.  This is the one that
    decides whether the run worked.  A policy that tracks ``x_t`` by stiffening
    every axis is *resisting* the teacher's yield and dragging the human along;
    it can still score well on position error.  ``K_∥/K_⊥ < 1`` while pulled is
    the signature of genuine compliance.
  * release transient — how far past ``x_ref`` the arm springs on let-go.
  * grasp-force error during post-grasp pushes.
  * ``s`` progression, to confirm freezing and resumption.

``--baseline`` runs the privileged oracle instead of a checkpoint, giving the
tracking ceiling to compare a student against.

  uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
      --stage A --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import FT_NORMAL_AXIS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.cartesian_impedance import CartesianImpedanceAction
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.compliance_tracking.config.flexiv.env_cfg import FORCE_SENSORS
from mjlab.tasks.compliance_tracking.mdp.observations import total_grasp_force
from mjlab.tasks.compliance_tracking.mdp.teacher import GRASP_HOLDING, TeacherCommand
from mjlab.tasks.compliance_tracking.scripts.baseline import OracleController
from mjlab.tasks.compliance_tracking.tracking_env_cfg import IMPEDANCE, TEACHER
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_TASK_IDS = {
  "A": "Mjlab-ComplianceTracking-StageA-Flexiv",
  "B": "Mjlab-ComplianceTracking-StageB-Flexiv",
  "C": "Mjlab-ComplianceTracking-StageC-Flexiv",
}


def _resolve_ckpt(path: str) -> Path:
  p = Path(path)
  if p.is_dir():
    ckpts = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    assert ckpts, f"no model_*.pt in {p}"
    return ckpts[-1]
  return p


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
  n = mask.sum()
  if n == 0:
    return float("nan")
  return float((values * mask).sum() / n)


def evaluate(
  env: ManagerBasedRlEnv, policy, steps: int, with_object: bool
) -> dict[str, float]:
  teacher = env.command_manager.get_term(TEACHER)
  assert isinstance(teacher, TeacherCommand)
  impedance = env.action_manager.get_term(IMPEDANCE)
  assert isinstance(impedance, CartesianImpedanceAction)

  acc: dict[str, list[torch.Tensor]] = defaultdict(list)
  was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  release_left = torch.zeros(env.num_envs, device=env.device)
  prev_ref_err = torch.zeros(env.num_envs, device=env.device)

  obs, _ = env.reset()
  for _ in range(steps):
    with torch.no_grad():
      action = policy(obs)
    obs, _, _, _, _ = env.step(action)

    active = teacher.perturbation.active
    released_now = was_active & ~active
    release_left = torch.where(
      released_now,
      torch.full_like(release_left, 0.6),
      (release_left - env.step_dt).clamp(min=0.0),
    )
    was_active = active.clone()
    in_release = (release_left > 0.0) & ~active
    grasped = teacher.grasp_state == GRASP_HOLDING

    err = torch.norm(teacher.ee_pos_w() - teacher.x_t, dim=-1)
    ref_err = torch.norm(teacher.ee_pos_w() - teacher.x_ref(), dim=-1)

    k = impedance.stiffness
    u = teacher.perturbation.direction
    k_par = (u * u * k).sum(dim=-1)
    k_perp = (k.sum(dim=-1) - k_par) / 2.0

    acc["err"].append(err)
    acc["ref_err"].append(ref_err)
    acc["active"].append(active.float())
    acc["release"].append(in_release.float())
    acc["grasped"].append(grasped.float())
    acc["unperturbed"].append((~active & ~in_release & ~grasped).float())
    acc["k_par"].append(k_par)
    acc["k_perp"].append(k_perp)
    acc["s"].append(teacher.s)
    acc["s_rate"].append(teacher.s_dot / teacher.path.s_rate)
    acc["overshoot"].append((ref_err - prev_ref_err).clamp(min=0.0))
    acc["f_ext"].append(torch.norm(teacher.perturbation.force, dim=-1))
    prev_ref_err = ref_err
    if with_object:
      f = total_grasp_force(env, FORCE_SENSORS, FT_NORMAL_AXIS)
      acc["grasp_err"].append((f - teacher.finger_force_target).abs())
      acc["welded"].append(teacher.welded.float())

  d = {k: torch.stack(v) for k, v in acc.items()}
  active_m, release_m = d["active"].bool(), d["release"].bool()
  grasp_m, unpert_m = d["grasped"].bool(), d["unperturbed"].bool()

  out = {
    "track_err_all_cm": float(d["err"].mean()) * 100,
    "track_err_unperturbed_cm": _masked_mean(d["err"], unpert_m) * 100,
    "track_err_pull_cm": _masked_mean(d["err"], active_m) * 100,
    "track_err_release_cm": _masked_mean(d["err"], release_m) * 100,
    "pulled_fraction": float(d["active"].mean()),
    "peak_f_ext_N": float(d["f_ext"].max()),
    "K_parallel_pull": _masked_mean(d["k_par"], active_m),
    "K_perp_pull": _masked_mean(d["k_perp"], active_m),
    # THIS is the anisotropy figure to read: E[K_par] / E[K_perp].
    #
    # The per-step ratio averaged instead (K_anisotropy_meanratio below) is
    # convex in K_par and therefore upward-biased by Jensen whenever the ratio
    # is high-variance -- measured at 1.09 vs 1.80 on the Stage B policy, i.e.
    # the bias can be larger than the effect. Both are reported so the gap
    # between them stays visible; a large gap means the per-step distribution is
    # heavy-tailed and only the ratio of means is interpretable.
    "K_anisotropy_pull": (
      _masked_mean(d["k_par"], active_m)
      / max(_masked_mean(d["k_perp"], active_m), 1e-3)
    ),
    "K_anisotropy_meanratio": _masked_mean(
      d["k_par"] / d["k_perp"].clamp(min=1e-3), active_m
    ),
    "release_overshoot_cm": _masked_mean(d["overshoot"], release_m) * 100,
    # Peak over the episode, not the last sample: episodes auto-reset, so the
    # final step of the trace often lands just past a boundary where s is 0.
    "s_peak": float(d["s"].max(dim=0).values.mean()),
    "s_rate_pulled": _masked_mean(d["s_rate"], active_m),
    "s_rate_free": _masked_mean(d["s_rate"], ~active_m),
  }
  if with_object:
    out["track_err_post_grasp_cm"] = _masked_mean(d["err"], grasp_m) * 100
    out["grasp_force_err_N"] = _masked_mean(d["grasp_err"], grasp_m)
    out["grasp_force_err_pushed_N"] = _masked_mean(d["grasp_err"], grasp_m & active_m)
    out["welded_fraction"] = float(d["welded"].mean())
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", default="A", choices=("A", "B", "C"))
  parser.add_argument("--checkpoint", default=None, help="policy .pt to evaluate")
  parser.add_argument("--baseline", action="store_true", help="run the oracle instead")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=800)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  if not args.baseline and args.checkpoint is None:
    parser.error("pass --checkpoint, or --baseline for the analytic ceiling")

  torch.manual_seed(args.seed)
  with_object = args.stage in ("B", "C")
  task = _TASK_IDS[args.stage]
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = args.num_envs
  # The oracle is privileged anyway; evaluating a student on noisy sensors is
  # the honest test, so corruption stays on for a checkpoint.
  cfg.observations["actor"].enable_corruption = not args.baseline
  env = ManagerBasedRlEnv(cfg, device=args.device)

  if args.baseline:
    oracle = OracleController(
      env,
      force_sensor_names=FORCE_SENSORS if with_object else (),
      normal_axis=FT_NORMAL_AXIS,
    )
    policy = lambda _obs: oracle.act()  # noqa: E731
    label = "analytic oracle (privileged)"
  else:
    ckpt = _resolve_ckpt(args.checkpoint)
    agent_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task)
    assert runner_cls is not None, f"task {task} has no runner class registered"
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
      str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device
    )
    policy = runner.get_inference_policy(device=args.device)
    label = str(ckpt)

  results = evaluate(env, policy, args.steps, with_object)

  print(f"\n=== Stage {args.stage}: {label} ===")
  print(f"{args.num_envs} envs x {args.steps} steps, seed {args.seed}\n")
  width = max(len(k) for k in results)
  for key, value in results.items():
    print(f"  {key:<{width}}  {value:10.4f}")
  ratio = results["K_anisotropy_pull"]
  verdict = "softens along the pull" if ratio < 0.9 else "NOT softening along the pull"
  print(f"\n  K_parallel/K_perp while pulled = {ratio:.3f} -> {verdict}")
  if args.baseline:
    print(
      "  (expected for the oracle: it commands a fixed isotropic stiffness, so\n"
      "   1.000 is correct here. It bounds tracking error, not compliance.)"
    )
  env.close()


if __name__ == "__main__":
  main()
