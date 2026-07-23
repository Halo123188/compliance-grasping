"""OOD generalization sweep for the Stage-1 compliance policy.

Probes the trained policy outside its training distribution along the axes the
demo actually cares about:

  * how hard the person pulls   (``f_max``, trained 35-80 N)
  * how stiff their arm is      (``kh_range``, trained 100-3000 N/m)
  * how many times they pull    (``second_push_prob``, trained 0.3; the human
    re-rolls on every release, so 1.0 means the arm is harassed all episode)
  * where they grab             (``grab_bodies``, trained on link5/6/7 +
    gripper_base; link3/link4 are unseen and much closer to the base)
  * how far they drag           (``d_range``, curriculum topped out at 0.05-0.30)

Reported metrics are the honest ones.  Success rate and loop completion are both
near-perfect for a fully braced arm (e14 scored 1.000 / 0.557 while never leaving
the goal), so the headline pair is **yield ratio** and **recovery time** -- a
braced arm scores ~0 on both because it was never pulled away.

  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.eval_ood \
    --checkpoint logs/rsl_rl/compliance_reach_flexiv/<run> --device cuda:0
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.compliance.mdp.curriculum import _LEVELS
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_TASK = "Mjlab-Compliance-Reach-Flexiv-Full"
_EVENT = "human_disturbance"


def _resolve_ckpt(path: str) -> Path:
  p = Path(path)
  if p.is_dir():
    ckpts = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    assert ckpts, f"no model_*.pt in {p}"
    return ckpts[-1]
  return p


def measure(
  ckpt: Path,
  task: str,
  device: str,
  num_envs: int,
  steps: int,
  d_range: tuple[float, float],
  overrides: dict[str, Any],
) -> dict[str, float]:
  """Run one condition and return the honest metric set."""
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  cfg.curriculum = {}  # fixed displacement, no adaptation during eval
  cfg.seed = 0
  cfg.observations["actor"].enable_corruption = False
  cfg.terminations.pop("success", None)  # full episodes: every push plays out
  cfg.events[_EVENT].params.update(overrides)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)

  human = env.event_manager.get_term_cfg(_EVENT).func
  reach = env.command_manager.get_term("reach")
  impedance = env.action_manager.get_term("impedance")
  human.d_range = d_range

  agent_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  obs = wrapped.get_observations()

  n = num_envs
  x_start = torch.zeros(n, 3, device=device)
  u_dir = torch.zeros(n, 3, device=device)
  d_mag = torch.zeros(n, device=device)
  peak_f = torch.zeros(n, device=device)
  push_steps = torch.zeros(n, device=device)
  t_env = torch.zeros(n, device=device)
  release_t = torch.full((n,), -1.0, device=device)
  was_pushing = torch.zeros(n, dtype=torch.bool, device=device)
  ever_success = torch.zeros(n, dtype=torch.bool, device=device)

  yields: list[float] = []
  peaks: list[float] = []
  recoveries: list[float] = []
  successes: list[float] = []
  n_onsets = 0
  k_par_sum = 0.0
  k_perp_sum = 0.0

  for _ in range(steps):
    with torch.no_grad():
      action = policy(obs)
    obs, _, dones, _ = wrapped.step(action)
    dones = dones.bool()
    t_env += env.step_dt

    pushing = human.is_pushing()
    ee = reach.ee_pos_w()
    f = torch.norm(human.external_force(), dim=-1)

    onset = pushing & ~was_pushing
    n_onsets += int(onset.sum())
    if onset.any():
      x_start[onset] = ee[onset]
      u_dir[onset] = human.push_dir()[onset]
      d_mag[onset] = torch.norm(human._x_target[onset] - human._x_start[onset], dim=-1)
      peak_f[onset] = 0.0
      push_steps[onset] = 0.0

    if pushing.any():
      peak_f = torch.where(pushing, torch.maximum(peak_f, f), peak_f)
      push_steps = torch.where(pushing, push_steps + 1.0, push_steps)
      k = impedance.stiffness
      k_par = (u_dir**2 * k).sum(dim=-1)
      k_par_sum += float(k_par[pushing].sum())
      k_perp_sum += float(((k.sum(dim=-1) - k_par) / 2.0)[pushing].sum())

    release = ~pushing & was_pushing
    for i in release.nonzero(as_tuple=False).squeeze(-1).tolist():
      if d_mag[i] < 1e-4 or push_steps[i] < 1:
        continue
      yields.append(torch.dot(ee[i] - x_start[i], u_dir[i]).item() / d_mag[i].item())
      peaks.append(peak_f[i].item())
      release_t[i] = t_env[i]

    at_goal = reach.success()
    ever_success |= at_goal
    rec = (release_t >= 0) & at_goal
    for i in rec.nonzero(as_tuple=False).squeeze(-1).tolist():
      recoveries.append((t_env[i] - release_t[i]).item())
      release_t[i] = -1.0

    was_pushing = pushing.clone()

    if dones.any():
      for i in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
        successes.append(float(ever_success[i].item()))
      t_env[dones] = 0.0
      release_t[dones] = -1.0
      was_pushing[dones] = False
      ever_success[dones] = False
      if hasattr(policy, "reset"):
        policy.reset(dones)

  env.close()

  def med(a: list[float]) -> float:
    return float(np.median(a)) if a else float("nan")

  return {
    "yield": med(yields),
    "peakF": med(peaks),
    "recov": med(recoveries),
    "loop": len(recoveries) / max(n_onsets, 1),
    "succ": float(np.mean(successes)) if successes else float("nan"),
    "kratio": k_par_sum / max(k_perp_sum, 1e-6),
    "pushes": float(n_onsets),
  }


# (section header, row label, d_range, param overrides).  "*" marks OOD rows.
def _conditions(base_d, far_d):
  return [
    ("", "BASELINE (training distribution)", base_d, {}),
    (
      "pull force (trained f_max 35-80 N)",
      "f_max = 20 N  *weak",
      base_d,
      {"f_max_range": (20.0, 20.0)},
    ),
    ("", "f_max = 35 N", base_d, {"f_max_range": (35.0, 35.0)}),
    ("", "f_max = 55 N", base_d, {"f_max_range": (55.0, 55.0)}),
    ("", "f_max = 80 N", base_d, {"f_max_range": (80.0, 80.0)}),
    ("", "f_max = 110 N  *strong", base_d, {"f_max_range": (110.0, 110.0)}),
    ("", "f_max = 150 N  *very strong", base_d, {"f_max_range": (150.0, 150.0)}),
    (
      "hand stiffness (trained K_h 100-3000)",
      "K_h = 30-100  *limp hand",
      base_d,
      {"kh_range": (30.0, 100.0)},
    ),
    ("", "K_h = 3000-8000  *rigid hand", base_d, {"kh_range": (3000.0, 8000.0)}),
    (
      "repeat pulls (trained p=0.3)",
      "p(re-pull) = 0.0  single pull",
      base_d,
      {"second_push_prob": 0.0},
    ),
    ("", "p(re-pull) = 0.3  (trained)", base_d, {"second_push_prob": 0.3}),
    ("", "p(re-pull) = 1.0  *relentless", base_d, {"second_push_prob": 1.0}),
    (
      "",
      "p = 1.0, short gaps  *relentless+",
      base_d,
      {"second_push_prob": 1.0, "second_delay_range": (0.15, 0.4)},
    ),
    (
      "grab location (trained link5/6/7/grip)",
      "grab = gripper_base",
      base_d,
      {"grab_bodies": ("gripper_base",)},
    ),
    ("", "grab = link7 (wrist)", base_d, {"grab_bodies": ("link7",)}),
    ("", "grab = link6", base_d, {"grab_bodies": ("link6",)}),
    ("", "grab = link5", base_d, {"grab_bodies": ("link5",)}),
    ("", "grab = link4  *unseen", base_d, {"grab_bodies": ("link4",)}),
    ("", "grab = link3  *unseen, near base", base_d, {"grab_bodies": ("link3",)}),
    (
      "drag distance (curriculum max 0.30)",
      "d = 0.05-0.30 (curriculum max)",
      far_d,
      {},
    ),
    ("", "d = 0.30-0.45  *farther", (0.30, 0.45), {}),
  ]


def main(
  checkpoint: str,
  task: str = _TASK,
  device: str = "cpu",
  num_envs: int = 256,
  steps: int = 1450,
) -> None:
  ckpt = _resolve_ckpt(checkpoint)
  # base_d matches eval.py's level 3 so the BASELINE row is directly comparable
  # to the headline eval numbers.
  conds = _conditions(_LEVELS[3], _LEVELS[4])

  hdr = (
    f"{'condition':40s} {'yield':>7s} {'peakF':>7s} {'recov':>7s} "
    f"{'loop':>6s} {'K/K':>6s} {'succ':>6s}"
  )
  print(f"\n==== OOD sweep: {ckpt.parent.name}/{ckpt.name} ====")
  print("headline = yield + recov; a braced arm reads ~0 on both yet succ ~1.0")
  print(f"\n{hdr}\n{'-' * len(hdr)}")
  for section, label, d_range, ov in conds:
    if section:
      print(f"\n-- {section} --")
    m = measure(ckpt, task, device, num_envs, steps, d_range, ov)
    print(
      f"{label:40s} {m['yield']:7.3f} {m['peakF']:6.1f}N {m['recov']:6.2f}s "
      f"{m['loop']:6.3f} {m['kratio']:6.3f} {m['succ']:6.3f}",
      flush=True,
    )


if __name__ == "__main__":
  tyro.cli(main)
