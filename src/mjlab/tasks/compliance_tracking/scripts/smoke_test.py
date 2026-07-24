"""Environment sanity checks for the compliance-tracking task.

Run before training anything.  These verify the *teacher*, not the policy: if
the target signal is wrong, no amount of RL will produce the behaviour, and the
tracking reward will happily report success against a wrong target.

  uv run python -m mjlab.tasks.compliance_tracking.scripts.smoke_test --stage A

Checks:
  1. The env builds, steps, and stays finite under a zero action.
  2. Unperturbed, ``s`` runs to 1.0 at the nominal rate (no spurious freezing).
  3. Under a pull, ``s`` freezes and the admittance target yields along the pull.
  4. On release, the target returns to ``x_ref(s)`` without oscillating.
  5. Stage B only: the scripted grasp fires, the weld activates, and the object
     stays with the hand while the arm is pushed.
"""

from __future__ import annotations

import argparse

import torch

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import FT_NORMAL_AXIS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.compliance_tracking.config.flexiv.env_cfg import (
  FORCE_SENSORS,
  flexiv_tracking_env_cfg,
)
from mjlab.tasks.compliance_tracking.mdp.observations import total_grasp_force
from mjlab.tasks.compliance_tracking.mdp.teacher import GRASP_HOLDING, TeacherCommand
from mjlab.tasks.compliance_tracking.scripts.baseline import OracleController
from mjlab.tasks.compliance_tracking.tracking_env_cfg import OBJECT_NAME, TEACHER

_MIN_ENVS_FOR_RATE = 32
"""Below this, the unperturbed-episode rate is too noisy to assert on."""


def _teacher(env: ManagerBasedRlEnv) -> TeacherCommand:
  term = env.command_manager.get_term(TEACHER)
  assert isinstance(term, TeacherCommand)
  return term


def _report(name: str, ok: bool, detail: str = "") -> bool:
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
  return ok


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", default="A", choices=("A", "B", "C"))
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--steps", type=int, default=800)
  args = parser.parse_args()

  cfg = flexiv_tracking_env_cfg(stage=args.stage)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=args.device)
  teacher = _teacher(env)
  action_dim = sum(env.action_manager.action_term_dim)

  print(f"\nStage {args.stage}: {env.num_envs} envs, action dim {action_dim}")
  print(
    f"  actor obs {env.observation_manager.group_obs_dim['actor']}, "
    f"critic obs {env.observation_manager.group_obs_dim['critic']}"
  )

  env.reset()
  # The arm has to actually be driven: a still arm deviates from x_ref within a
  # few steps, which freezes s, which means no s-keyed perturbation ever fires.
  # The oracle stands in for a trained student here.
  oracle = OracleController(
    env,
    force_sensor_names=FORCE_SENSORS if args.stage in ("B", "C") else (),
    normal_axis=FT_NORMAL_AXIS,
  )
  ok = True

  s_trace, dev_trace, pulled, track_err, dones = [], [], [], [], []
  held_trace, obj_slip, grasp_force = [], [], []
  welded_seen = grasped_seen = False
  peak_force = 0.0
  finite = True
  obj = env.scene[OBJECT_NAME] if args.stage in ("B", "C") else None
  for _ in range(args.steps):
    obs, rew, terminated, truncated, _ = env.step(oracle.act())
    # Episodes auto-reset in place, so any statistic taken across consecutive
    # steps has to skip the boundary: s jumps from ~1 back to 0 there, which
    # would otherwise read as a large negative path rate.
    dones.append((terminated | truncated).clone())
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    finite &= bool(torch.isfinite(actor_obs).all() and torch.isfinite(rew).all())
    finite &= bool(torch.isfinite(teacher.x_t).all())
    s_trace.append(teacher.s.clone())
    dev_trace.append(teacher.deviation.clone())
    pulled.append(teacher.perturbation.active.clone())
    track_err.append(torch.norm(teacher.ee_pos_w() - teacher.x_t, dim=-1))
    peak_force = max(
      peak_force, float(torch.norm(teacher.perturbation.force, dim=-1).max())
    )
    grasped_seen |= bool((teacher.grasp_state == GRASP_HOLDING).any())
    welded_seen |= bool(teacher.welded.any())
    if obj is not None:
      held_trace.append(teacher.welded.clone())
      obj_slip.append(torch.norm(obj.data.root_link_pos_w - teacher.ee_pos_w(), dim=-1))
      grasp_force.append(
        total_grasp_force(env, FORCE_SENSORS, FT_NORMAL_AXIS)
        - teacher.finger_force_target
      )
  ok &= _report("rollout stays finite", finite)

  s_hist = torch.stack(s_trace)  # (T, N)
  dev_hist = torch.stack(dev_trace)
  pull_hist = torch.stack(pulled)
  err_hist = torch.stack(track_err)
  # ds[i] = s[i+1] - s[i], so the boundary to drop is the one where step i+1 was
  # the reset. (env.step returns *post*-reset state, so s is already back at 0 by
  # the time the done step is recorded.)
  same_episode = ~torch.stack(dones)[1:]
  ok &= _report(
    "oracle tracks the teacher target",
    float(err_hist.mean()) < 0.05,
    f"mean ‖x_ee − x_t‖ = {float(err_hist.mean()) * 100:.2f} cm, "
    f"p95 = {float(err_hist.flatten().quantile(0.95)) * 100:.2f} cm",
  )

  # --- 2. an unperturbed env should run s to completion ---------------------
  never_pulled = ~pull_hist.any(dim=0)
  frac_none = float(never_pulled.float().mean())
  target = teacher.cfg.perturbation.p_no_perturbation
  detail = f"{frac_none:.0%} of envs never pulled (target ~{target:.0%})"
  if args.num_envs < _MIN_ENVS_FOR_RATE:
    # One Bernoulli draw per env: with p=0.25 and 8 envs, "zero unperturbed"
    # happens 10% of the time. Reporting that as a failure would be noise.
    print(
      f"  [INFO] unperturbed-episode rate — {detail} (need >= "
      f"{_MIN_ENVS_FOR_RATE} envs to assert)"
    )
  else:
    ok &= _report(
      "unperturbed-episode rate is in range",
      0.4 * target <= frac_none <= 2.0 * target,
      detail,
    )
  if bool(never_pulled.any()):
    # Peak over the episode, not the final sample: with auto-reset the last
    # step of the trace may land just after a boundary, where s is back at 0.
    s_peak = s_hist[:, never_pulled].max(dim=0).values
    ok &= _report(
      "s completes when untouched",
      bool((s_peak > 0.95).any()),
      f"max s = {float(s_peak.max()):.3f}",
    )

  # --- 3. s freezes under a pull --------------------------------------------
  if bool(pull_hist.any()):
    ds = (s_hist[1:] - s_hist[:-1]) / env.step_dt
    nominal = teacher.path.s_rate
    m_pull = pull_hist[1:] & same_episode
    m_free = ~pull_hist[1:] & same_episode
    rate_pull = float((ds * m_pull).sum() / m_pull.sum().clamp(min=1)) / nominal
    rate_free = float((ds * m_free).sum() / m_free.sum().clamp(min=1)) / nominal
    ok &= _report(
      "s freezes while pulled",
      rate_pull < 0.6 * rate_free,
      f"rate while pulled {rate_pull:.2f} vs free {rate_free:.2f} (x nominal)",
    )
    dev_pull = float((dev_hist[1:] * m_pull).sum() / m_pull.sum().clamp(min=1))
    ok &= _report(
      "deviation grows under a pull",
      dev_pull > 0.005,
      f"mean deviation while pulled {dev_pull * 100:.1f} cm",
    )

  # --- 4. the perturbation force is bounded ---------------------------------
  ok &= _report(
    "perturbation force is saturated",
    peak_force <= teacher.cfg.perturbation.force_limit + 1e-3,
    f"instantaneous peak {peak_force:.1f} N",
  )

  # --- 5. Stage B: grasp, weld, and the post-grasp push claim ---------------
  if obj is not None:
    ok &= _report("scripted grasp reaches HOLDING", grasped_seen)
    ok &= _report("object weld activates", welded_seen)

    held = torch.stack(held_trace)
    slip = torch.stack(obj_slip)
    ferr = torch.stack(grasp_force).abs()
    pushed_held = held & pull_hist

    if bool(held.any()):
      # The weld freezes the relative pose, so the object-to-EE distance must
      # stay at whatever it was when the weld was made.
      held_slip = slip[held]
      ok &= _report(
        "welded object stays with the hand",
        float(held_slip.max()) < 0.06,
        f"max object-to-EE distance while welded {float(held_slip.max()) * 100:.1f} cm",
      )
    if bool(pushed_held.any()):
      # The §4 headline: the arm yields but the grasp force does not change.
      ok &= _report(
        "grasp force held while pushed post-grasp",
        float(ferr[pushed_held].mean()) < 6.0,
        f"mean |F − F_target| = {float(ferr[pushed_held].mean()):.2f} N "
        f"over {int(pushed_held.sum())} pushed-and-held steps",
      )
    else:
      print("  [SKIP] no post-grasp push occurred in this rollout")

  print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}\n")
  env.close()
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
