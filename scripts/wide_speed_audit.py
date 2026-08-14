"""How fast does a trained policy actually drive the arm?

  uv run python scripts/wide_speed_audit.py NAME=TASK:CKPT [NAME=TASK:CKPT ...]
  uv run python scripts/wide_speed_audit.py envs=64 NAME=TASK:CKPT

Append ``:teacher`` to a distillation checkpoint to audit the frozen teacher
instead of the student, exactly as in ``eval_deploy.py``.

WHY THIS EXISTS. The deployed policy moves the arm faster than the bench is
comfortable with, and nothing in the training config bounds that. The only
speed-related terms are soft: ``joint_vel_hinge`` charges the SQUARE of the
excess over 0.5 rad/s at a final weight of -0.1, and ``action_rate_l2`` charges
-0.01. Neither caps anything, so before choosing a limit there has to be a
measurement of what the policy is doing without one.

THREE DIFFERENT SPEEDS, and they are not interchangeable:

  joint velocity      what the joint actually does. This is what
                      `joint_vel_hinge` reads and the only one currently
                      penalized at all.
  commanded rate      d(position target)/dt. THIS is the number a slew limiter
                      would clamp and the one the real controller is handed --
                      the arm's own tracking error means it can differ from the
                      measured velocity by a lot. With arm scale 0.40 rad at
                      50 Hz, a one-step action swing of +-1 asks for 20 rad/s.
  TCP speed           how fast the hand moves through the workspace, in m/s.
                      This is what "the arm moves too fast" means to a person
                      standing next to it, and it is the number to quote when
                      picking a limit.

Reported per joint and split by PHASE -- before the cube leaves the surface
(free-space approach and the grasp) versus after (the carry). The split is what
decides whether a limit can be bought cheaply: throttling a free-space approach
costs a fraction of a second, throttling the grasp and lift changes the task.

The threshold tables at the end answer the sizing question directly: for each
candidate limit, how much of the rollout it would have clipped.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import geometry_for  # noqa: E402

STEPS, DEV = 300, "cuda:0"
# The cube being clear of the surface is the phase boundary: everything before
# it is approach + grasp, everything after is the carry.
#
# Measured as a RISE from each env's own first frame, not as an absolute height.
# The absolute version reads the cube CENTRE, which starts a half-edge above the
# surface and is lifted a further 5-10 mm by the DR spawn, so it is already past
# any small threshold at step 0 -- the first run of this script reported the
# whole rollout as "carry". Relative is also the only version that survives
# `dr_cube_scale`, which makes the half-edge itself a random variable.
LIFTED = 0.02
ARM_LIMITS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
TCP_LIMITS = (0.1, 0.25, 0.5, 0.75, 1.0)


def _load_policy(runner, ckpt: str, role: str):
  """Load `ckpt` into `runner` and return the network that should act."""
  if role == "teacher":
    runner.load(ckpt, load_cfg={"teacher": True, "iteration": False}, map_location=DEV)
    runner.alg.eval_mode()
    return runner.alg.teacher
  is_distilled = "student_state_dict" in torch.load(
    ckpt, map_location="cpu", weights_only=False
  )
  load_cfg = {"student": True} if is_distilled else {"actor": True}
  runner.load(ckpt, load_cfg=load_cfg, strict=True, map_location=DEV)
  return runner.get_inference_policy(device=DEV)


def rollout(task: str, ckpt: str, role: str, num_envs: int) -> dict:
  geo = geometry_for(task)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  # No terminations, so one continuous window is measured per env rather than a
  # mixture of episodes that reset at different times (`eval_deploy` does the
  # same). Successful envs simply hold the cube for the remainder, which is
  # honest: holding still is part of what the policy does.
  cfg.terminations = {}
  # And no goal resample inside the window. `LiftingCommand` RESPAWNS the cube
  # when it resamples, and the play cfg resamples every 5 s -- i.e. at step 250
  # of 300 the cube teleports and the policy lunges after it, which lands in the
  # carry-phase peaks as if the policy had done it while holding the cube.
  cfg.commands["lift_height"].resampling_time_range = (1e9, 1e9)
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  policy = _load_policy(runner, ckpt, role)

  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, BaseAction)
  robot, cube = env.scene["robot"], env.scene["cube"]
  qids = term.target_ids
  names = list(term.target_names)
  tcp = list(robot.site_names).index(geo.grasp_site)

  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  QV, CMD, REQ, TCP, H = [], [], [], [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    QV.append(robot.data.joint_vel[:, qids].clone())
    # TWO different targets, and conflating them is what the first version of
    # this script did:
    #
    #   published  what the servo (and the robot host) actually receives. Read
    #              off the entity, so it is whatever the action term ended up
    #              writing -- rate-limited, low-passed or neither.
    #   requested  what the network asked for before any of that. Equal to the
    #              published one on the unlimited tasks, and on a rate-limited
    #              one it is unbounded by construction: the policy is free to
    #              lean on a limiter that absorbs the excess, and measured on
    #              the -Slew arms it does exactly that.
    #
    # Only `published` is a statement about the hardware. `requested` is a
    # statement about how hard the policy is leaning on the limiter, which is
    # what says whether removing it would be catastrophic.
    CMD.append(robot.data.joint_pos_target[:, qids].clone())
    REQ.append((term.raw_action * term.scale + term.offset).clone())
    TCP.append(robot.data.site_pos_w[:, tcp].clone())
    H.append(
      (
        cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - geo.surface_z
      ).clone()
    )
  dt = env.step_dt
  out = dict(
    names=names,
    dt=dt,
    qv=torch.stack(QV).abs().cpu().numpy(),
    cmd=torch.stack(CMD).cpu().numpy(),
    req=torch.stack(REQ).cpu().numpy(),
    tcp=torch.stack(TCP).cpu().numpy(),
    h=torch.stack(H).cpu().numpy(),
  )
  env.close()
  return out


def _derive(r: dict) -> dict:
  """Commanded rate, TCP speed and the phase mask, all aligned on step 1..T-1."""
  dt = r["dt"]
  cmd_rate = np.abs(np.diff(r["cmd"], axis=0)) / dt  # (T-1, N, dim)
  req_rate = np.abs(np.diff(r["req"], axis=0)) / dt
  tcp_speed = np.linalg.norm(np.diff(r["tcp"], axis=0), axis=-1) / dt  # (T-1, N)
  # "Carrying" from the first step the cube is clear of the surface onward, so a
  # momentary dip back under the line does not flip the phase back.
  rise = r["h"] - r["h"][0]
  carry = np.maximum.accumulate(rise > LIFTED, axis=0)[1:]
  return dict(
    qv=r["qv"][1:],
    cmd_rate=cmd_rate,
    req_rate=req_rate,
    tcp_speed=tcp_speed,
    rise=rise,
    carry=carry,
    arm=[i for i, n in enumerate(r["names"]) if n.startswith("joint")],
  )


def _pct(x: np.ndarray, q: float) -> float:
  return float(np.percentile(x, q)) if x.size else float("nan")


def report(name: str, r: dict) -> None:
  d = _derive(r)
  qv, rate, arm = d["qv"], d["cmd_rate"], d["arm"]
  approach, carry = ~d["carry"], d["carry"]

  print(f"\n{'=' * 78}\n{name}   {r['dt'] * 1000:.0f} ms control step")
  print(f"{'=' * 78}")
  print(
    f"\n{'joint':>10} | {'|q_dot| rad/s':^26} | {'PUBLISHED |dq_cmd/dt| rad/s':^26}"
    f"\n{'':>10} | {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}"
    f"  | {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}"
  )
  for i, n in enumerate(r["names"]):
    v, c = qv[:, :, i], rate[:, :, i]
    print(
      f"{n:>10} | {_pct(v, 50):7.2f} {_pct(v, 95):7.2f} {_pct(v, 99):7.2f}"
      f" {v.max():7.2f}  | {_pct(c, 50):7.2f} {_pct(c, 95):7.2f}"
      f" {_pct(c, 99):7.2f} {c.max():7.2f}"
    )

  # Per control step, the fastest arm joint -- what a single scalar limit binds.
  worst = qv[:, :, arm].max(-1)
  worst_cmd = rate[:, :, arm].max(-1)
  worst_req = d["req_rate"][:, :, arm].max(-1)
  tcp = d["tcp_speed"]
  # p99.9 alongside max, because `max` is not a robust statistic here: this
  # script clears terminations, and `nonfinite_state` is one of them, so a
  # single mujoco-warp blow-up in one env of 128 lands in the max column and
  # nowhere else. Where the two disagree by a lot, believe p99.9.
  print(f"\n{'':<24}{'p50':>8}{'p95':>8}{'p99':>8}{'p99.9':>8}{'max':>8}")
  for label, x in (
    ("fastest arm joint", worst),
    ("published cmd rate", worst_cmd),
    ("requested cmd rate", worst_req),
    ("TCP speed (m/s)", tcp),
  ):
    print(
      f"{label:<24}{_pct(x, 50):8.2f}{_pct(x, 95):8.2f}{_pct(x, 99):8.2f}"
      f"{_pct(x, 99.9):8.2f}{x.max():8.2f}"
    )
  # How hard the policy leans on the limiter. 1.0 means it never asks for more
  # than it can have; a large number means the action has become bang-bang and
  # the limiter is the only thing shaping the motion -- which is a statement
  # about DEPLOYMENT RISK, not about the sim: run that checkpoint without a
  # matching limiter on the robot host and it publishes the requested rate.
  lean = _pct(worst_req, 95) / max(_pct(worst_cmd, 95), 1e-9)
  print(f"{'requested / published':<24}{lean:8.1f}x  at p95")

  # The onset below comes out at ~0.2 s, which is fast enough to be a phase-mask
  # artifact rather than a measurement, so print the trace it is derived from.
  rise, tcp_t = d["rise"][1:], d["tcp_speed"]
  marks = [t for t in (0, 2, 5, 10, 15, 20, 30, 50, 100, 200, 290) if t < len(rise)]
  print(f"\n{'step':>10}" + "".join(f"{t:8d}" for t in marks))
  print(f"{'t (s)':>10}" + "".join(f"{t * r['dt']:8.2f}" for t in marks))
  print(
    f"{'rise (mm)':>10}" + "".join(f"{np.median(rise[t]) * 1e3:8.1f}" for t in marks)
  )
  print(f"{'tcp (m/s)':>10}" + "".join(f"{np.median(tcp_t[t]):8.2f}" for t in marks))

  ever = carry[-1]
  onset = np.where(ever, carry.argmax(0), -1)
  print(
    f"\n  {ever.mean() * 100:.0f}% of envs lift the cube at all; those that do"
    f" leave the surface at step {int(np.median(onset[ever])) if ever.any() else -1}"
    f" ({np.median(onset[ever]) * r['dt'] if ever.any() else float('nan'):.1f} s)."
  )
  print(
    f"\n{'phase':<22}{'share':>8}{'arm p95':>9}{'arm p99.9':>10}"
    f"{'cmd p95':>9}{'tcp p95':>9}{'tcp p99.9':>10}"
  )
  for label, m in (("approach + grasp", approach), ("carry", carry)):
    if not m.any():
      print(f"{label:<22}{'0.0%':>8}" + f"{'--':>9}" * 5)
      continue
    w, c, t = worst[m], worst_cmd[m], tcp[m]
    print(
      f"{label:<22}{m.mean() * 100:7.1f}%{_pct(w, 95):9.2f}{_pct(w, 99.9):10.2f}"
      f"{_pct(c, 95):9.2f}{_pct(t, 95):9.2f}{_pct(t, 99.9):10.2f}"
    )

  print(
    f"\n  what a limit would clip:"
    f"\n{'v_max rad/s':>12}{'steps over':>12}{'joint-samples':>15}"
    f"{'mean excess':>13}{'  (approach / carry)':>22}"
  )
  for lim in ARM_LIMITS:
    over = qv[:, :, arm] > lim
    excess = (qv[:, :, arm] - lim).clip(min=0)
    a_over = (worst > lim)[approach].mean() if approach.any() else float("nan")
    c_over = (worst > lim)[carry].mean() if carry.any() else float("nan")
    print(
      f"{lim:12.2f}{(worst > lim).mean() * 100:11.1f}%{over.mean() * 100:14.1f}%"
      f"{excess[over].mean() if over.any() else 0.0:13.2f}"
      f"{a_over * 100:14.1f}% /{c_over * 100:6.1f}%"
    )

  print(f"\n{'tcp m/s':>12}{'steps over':>12}")
  for lim in TCP_LIMITS:
    print(f"{lim:12.2f}{(tcp > lim).mean() * 100:11.1f}%")


def main() -> None:
  specs, num_envs = [], 128
  for arg in sys.argv[1:]:
    if arg.startswith("envs="):
      num_envs = int(arg.split("=", 1)[1])
      continue
    name, rest = arg.split("=", 1)
    task, ckpt = rest.split(":", 1)
    role = "policy"
    if ckpt.endswith((":teacher", ":student")):
      ckpt, tail = ckpt.rsplit(":", 1)
      role = "teacher" if tail == "teacher" else "policy"
    specs.append((name, task, ckpt, role))

  if not specs:
    print(__doc__)
    raise SystemExit(2)

  for name, task, ckpt, role in specs:
    report(name, rollout(task, ckpt, role, num_envs))

  print(
    f"\n{num_envs} envs x {STEPS} steps, deterministic policy, terminations off."
    "\nThe hinge penalty currently in the config charges only |q_dot| above"
    " 0.5 rad/s,\nat a final weight of -0.1. Read the 0.50 row against that."
  )


if __name__ == "__main__":
  main()
