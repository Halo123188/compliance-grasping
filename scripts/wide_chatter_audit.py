"""How much does the arm SHAKE, as opposed to how fast does it move?

  uv run python scripts/wide_chatter_audit.py NAME=TASK:CKPT [NAME=TASK:CKPT ...]
  uv run python scripts/wide_chatter_audit.py envs=64 NAME=TASK:CKPT:teacher

Companion to ``wide_speed_audit.py``, which measures the wrong thing on its own.
That script prices peak speed, the slew arms cut it by 42%, and the resulting
policy is visibly worse to stand next to: a rate limit bounds SPEED but not
DIRECTION CHANGES, so a policy can sit exactly at the cap and reverse every few
steps. Peak speed cannot see that. These columns can.

  swing deg     THE HEADLINE. Mean joint travel between one reversal and the
                next, in degrees -- how big the shake is. Rate alone gets the
                ranking backwards, because it scores a fast small jitter above
                a slow large sway; a rate limiter lowers the frequency and
                raises the amplitude, which is the trade a person notices.
  reversals/s   sign changes of the published command's velocity, per second,
                per joint. The frequency half of the same picture, and only
                meaningful next to the swing.
  |accel|       mean magnitude of the command's second difference, rad/s^2.
                Reversals count events; this weighs them by how violent each
                one is, which is what the structure feels.
  tcp jerk      third difference of the tool point, m/s^3. The same quantity a
                person means by "jerky", and unlike the joint columns it does
                not need a per-joint table to read.
  dwell         fraction of carry-phase steps whose command rate is under 5% of
                the cap. How much of the time the arm is actually holding
                still, which is the thing a settled policy does and a chattering
                one never does.

Split by phase on the same rule as the speed audit -- approach and grasp before
the cube leaves the surface, carry after -- because the two phases have opposite
expectations. Reversals during the approach are a search; reversals during the
carry are a policy that cannot stop.
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
from wide_speed_audit import LIFTED, _load_policy  # noqa: E402

STEPS, DEV = 300, "cuda:0"
# A reversal is only counted once the joint is actually moving. Without this the
# columns are dominated by numerical sign flips in a command that is standing
# still, which is the opposite of what the metric is for.
MOVING = 0.02  # rad/s


def rollout(task: str, ckpt: str, role: str, num_envs: int) -> dict:
  geo = geometry_for(task)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  # Same two corrections as the speed audit: one continuous window per env, and
  # no goal resample, because `LiftingCommand` respawns the cube on resample and
  # the lunge after it lands in the carry-phase numbers.
  cfg.terminations = {}
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
  CMD, TCP, H = [], [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    # The PUBLISHED target, not the request: chatter is a claim about what the
    # servo is asked to do, and on a limited arm the request is unbounded.
    CMD.append(robot.data.joint_pos_target[:, qids].clone())
    TCP.append(robot.data.site_pos_w[:, tcp].clone())
    H.append(
      (
        cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - geo.surface_z
      ).clone()
    )
  out = dict(
    names=names,
    dt=env.step_dt,
    cmd=torch.stack(CMD).cpu().numpy(),
    tcp=torch.stack(TCP).cpu().numpy(),
    h=torch.stack(H).cpu().numpy(),
  )
  env.close()
  return out


def _derive(r: dict) -> dict:
  dt = r["dt"]
  vel = np.diff(r["cmd"], axis=0) / dt  # (T-1, N, dim), SIGNED
  acc = np.diff(vel, axis=0) / dt  # (T-2, N, dim)
  tcp_jerk = np.linalg.norm(np.diff(r["tcp"], n=3, axis=0), axis=-1) / dt**3

  # A reversal is a sign change between consecutive samples where BOTH are
  # moving. Requiring both is what stops a joint that merely stops and later
  # restarts in the other direction from being counted as chatter.
  a, b = vel[:-1], vel[1:]
  moving = (np.abs(a) > MOVING) & (np.abs(b) > MOVING)
  rev = (np.sign(a) != np.sign(b)) & moving  # (T-2, N, dim)

  rise = r["h"] - r["h"][0]
  carry_full = np.maximum.accumulate(rise > LIFTED, axis=0)

  # Each difference order costs a sample, so these come out at four different
  # lengths. They all END on the same step, so trimming from the front to the
  # shortest aligns them; the phase mask is monotone, so the sub-step offset
  # this leaves between columns cannot move a step into the wrong phase.
  n = min(len(rev), len(acc), len(tcp_jerk))
  return dict(
    dt=dt,
    rev=rev[-n:],
    acc=np.abs(acc[-n:]),
    vel=np.abs(vel[-n:]),
    tcp_jerk=tcp_jerk[-n:],
    carry=carry_full[-n:],
    arm=[i for i, n in enumerate(r["names"]) if n.startswith("joint")],
    names=r["names"],
  )


def _phase(d: dict, mask: np.ndarray, cap: float | None) -> dict:
  """Reduce every column over the steps/envs selected by `mask`."""
  if not mask.any():
    return {}
  arm = d["arm"]
  m3 = mask[..., None]
  rev_arm = d["rev"][..., arm]
  acc_arm = d["acc"][..., arm]
  # Per-joint reversal RATE: events in the phase over seconds in the phase.
  seconds = mask.sum() * d["dt"]
  rev_s = (rev_arm & m3).sum() / (seconds * len(arm))
  vel_arm = d["vel"][..., arm]
  sel = np.broadcast_to(m3, vel_arm.shape)
  speed = vel_arm[sel].mean()
  out = {
    "rev_s": rev_s,
    "rev_worst": max((rev_arm[..., j] & mask).sum() / seconds for j in range(len(arm))),
    "acc": acc_arm[np.broadcast_to(m3, acc_arm.shape)].mean(),
    "jerk": d["tcp_jerk"][mask].mean(),
    # THE COLUMN THAT MATCHES THE EYE. Reversal rate alone ranks a fast small
    # jitter above a slow large sway, and gets the answer backwards: measured
    # here, the UNLIMITED student reverses 2.4x more often than the limited one
    # and is the visibly steadier of the two. What a person sees is how far the
    # joint travels between reversals, and a rate limiter raises exactly that --
    # it pins the speed at the cap, so every half-cycle sweeps the full cap
    # times the half-period. Degrees, because that is the unit the swing is
    # judged in.
    "swing": np.degrees(speed / rev_s) if rev_s > 0 else 0.0,
  }
  if cap is not None:
    still = d["vel"][..., arm] < 0.05 * cap
    out["dwell"] = still[np.broadcast_to(m3, still.shape)].mean()
  return out


def report(name: str, r: dict, cap: float | None) -> None:
  d = _derive(r)
  approach, carry = ~d["carry"], d["carry"]
  print(f"\n=== {name}")
  hdr = (
    f"{'phase':>10} {'swing deg':>10} {'rev/s':>8} {'rev/s max':>10} "
    f"{'|acc|':>10} {'tcp jerk':>10}"
  )
  if cap is not None:
    hdr += f" {'dwell':>7}"
  print(hdr)
  for label, mask in (("approach", approach), ("carry", carry), ("all", None)):
    m = np.ones_like(d["carry"]) if mask is None else mask
    s = _phase(d, m, cap)
    if not s:
      print(f"{label:>10}   (no steps in this phase)")
      continue
    line = (
      f"{label:>10} {s['swing']:10.2f} {s['rev_s']:8.2f} {s['rev_worst']:10.2f} "
      f"{s['acc']:10.1f} {s['jerk']:10.0f}"
    )
    if cap is not None:
      line += f" {s['dwell']:7.2f}"
    print(line)


def main() -> None:
  specs, num_envs, cap = [], 128, None
  for arg in sys.argv[1:]:
    if arg.startswith("envs="):
      num_envs = int(arg.split("=", 1)[1])
      continue
    if arg.startswith("cap="):
      # The arm's slew cap in rad/s, only used to size the `dwell` threshold.
      cap = float(arg.split("=", 1)[1])
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
    report(name, rollout(task, ckpt, role, num_envs), cap)

  print(
    f"\n{num_envs} envs x {STEPS} steps, deterministic policy, terminations off."
    "\nrev/s is averaged over the six arm joints; rev/s max is the worst single"
    "\njoint, which is where a shake is usually concentrated. A smooth reach"
    "\nshould read 1-2 on approach and near 0 on carry."
  )


if __name__ == "__main__":
  main()
