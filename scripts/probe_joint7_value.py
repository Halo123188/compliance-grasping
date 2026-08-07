"""How much is unfreezing joint7 actually worth? Measure before training anything.

  uv run python scripts/probe_joint7_value.py TASK CKPT

joint7 is frozen in every variant so far -- parked with a 3-4 deg spread across
128 envs whose cube yaws span the full circle, action std collapsed to 0.0083.
Four reward-side attempts to unfreeze it all lost to the baseline. Before trying
an exploration-side fix, establish the ceiling: if perfect wrist alignment is
worth +3 points the whole line is not worth running.

Two passes, no training:

  P1  uniform cube yaw (the training distribution). Bucket success rate by jaw
      yaw error. The spread across buckets IS the headroom.
  P2  cube yaw locked to whatever the frozen wrist is already square to, so the
      policy gets a perfect wrist for free. This is a SUBSET of the training
      distribution, not out of it, so the number is trustworthy. Success here is
      the ceiling that unfreezing joint7 could approach.

Jaw is 180-deg symmetric and the cube is a square prism, so yaw error folds into
[0, 45] deg and uniform yaw gives a mean error of 22.5 deg.
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  LIFT_HEIGHT,
  TABLE_H,
)
from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, STEPS, DEV = 128, 300, "cuda:0"
HOLD = 50


def run(yaw_range=None):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  if yaw_range is not None:
    cmd = cfg.commands["lift_height"]
    assert isinstance(cmd, LiftingCommandCfg)
    cmd.object_pose_range.yaw = yaw_range
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
  runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)

  robot, cube = env.scene["robot"], env.scene["cube"]
  sn = list(robot.site_names)
  li, ri = sn.index("left_pad"), sn.index("right_pad")
  j7 = list(robot.joint_names).index("joint7")

  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  hs, jaws, cys, j7s = [], [], [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    hs.append(
      (cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone()
    )
    pads = robot.data.site_pos_w[:, [li, ri]]
    ax = pads[:, 1] - pads[:, 0]
    jaws.append(torch.atan2(ax[:, 1], ax[:, 0]).clone())
    q = cube.data.root_link_quat_w
    cys.append(
      torch.atan2(
        2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
      ).clone()
    )
    j7s.append(robot.data.joint_pos[:, j7].clone())
  env.close()

  h = torch.stack(hs).cpu()
  above = h > LIFT_HEIGHT
  hold = torch.zeros(N, dtype=torch.long)
  cur = torch.zeros(N, dtype=torch.long)
  for t in range(STEPS):
    cur = torch.where(above[t], cur + 1, torch.zeros_like(cur))
    hold = torch.maximum(hold, cur)
  succ = (hold >= HOLD).numpy()

  # Read posture over the approach window, before the lift reorients anything.
  win = slice(20, 60)
  jaw = torch.stack(jaws).cpu()[win].mean(0).numpy()
  cy = torch.stack(cys).cpu()[win].mean(0).numpy()
  err = np.degrees(jaw - cy) % 90
  err = np.minimum(err, 90 - err)
  return dict(
    succ=succ,
    peak=h.max(0).values.numpy(),
    err=err,
    jaw=np.degrees(jaw) % 90,
    j7=np.degrees(torch.stack(j7s).cpu()[win].mean(0).numpy()),
  )


print("\n=== P1: uniform cube yaw (training distribution) ===")
p1 = run()
print(
  f"success {p1['succ'].mean() * 100:.1f}%   mean jaw yaw error"
  f" {p1['err'].mean():.1f} deg   joint7 {p1['j7'].mean():+.1f} +- {p1['j7'].std():.1f}"
)

edges = [0, 11.25, 22.5, 33.75, 45.0]
print(f"\n{'yaw error bucket':>18} {'n':>4} {'success':>9} {'median peak':>12}")
for lo, hi in zip(edges[:-1], edges[1:], strict=True):
  m = (p1["err"] >= lo) & (p1["err"] < hi)
  if not m.any():
    continue
  print(
    f"{f'{lo:.1f}-{hi:.1f} deg':>18} {int(m.sum()):4d}"
    f" {p1['succ'][m].mean() * 100:8.1f}%"
    f" {np.median(p1['peak'][m]) * 1000:11.1f}mm"
  )

# Where the frozen jaw actually parks, folded into one 90 deg cell.
park = float(np.median(p1["jaw"]))
print(f"\nfrozen jaw parks at {park:.1f} deg (mod 90)")

lo = np.radians(park - 5.0)
hi = np.radians(park + 5.0)
print(f"\n=== P2: cube yaw locked to {park:.1f} +- 5 deg (free perfect wrist) ===")
p2 = run(yaw_range=(float(lo), float(hi)))
print(
  f"success {p2['succ'].mean() * 100:.1f}%   mean jaw yaw error {p2['err'].mean():.1f} deg"
)

print(
  f"\nCEILING: {p1['succ'].mean() * 100:.1f}% (frozen wrist, random yaw)"
  f" -> {p2['succ'].mean() * 100:.1f}% (effectively perfect alignment)"
  f"   headroom {(p2['succ'].mean() - p1['succ'].mean()) * 100:+.1f} points"
)
print("If headroom is small, unfreezing joint7 cannot pay for itself.")
