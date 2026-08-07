"""Is the policy using joint7 (wrist roll) to square the jaw, or the whole arm?

  uv run python scripts/diag_joint7_use.py TASK CKPT

Cube yaw is uniform on [-pi, pi] and the jaw has 180 deg symmetry, so squaring up
needs at most +-45 deg of wrist roll -- which joint7 alone can supply without
moving the arm at all. If the policy is instead contorting the arm to get the jaw
around, that costs reach, and lost reach is exactly the 7-9 mm the short-lift
failures come up short by.

The learned per-dimension action std says it may not be using joint7 at all:
0.0633 on the v8 checkpoint and 0.0083 on v10, the lowest of any joint, i.e.
+-0.5 deg of exploration. This checks the behaviour rather than the parameter:
per env, does joint7 actually track the cube's yaw?

Reports the correlation between cube yaw and joint7, how much of the jaw's final
orientation each source explains, and how far the wrist ends up from the arm's
own reach-efficient posture.
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, STEPS, DEV = 64, 250, "cuda:0"

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
agent = load_rl_cfg(TASK)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(wrapped, asdict(agent), device=DEV)
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

robot, cube = env.scene["robot"], env.scene["cube"]
jn = list(robot.joint_names)
j7 = jn.index("joint7")
sn = list(robot.site_names)
li, ri = sn.index("left_pad"), sn.index("right_pad")

torch.manual_seed(0)
obs = wrapped.reset()[0]
rec = []
for _t in range(STEPS):
  with torch.inference_mode():
    act = policy(obs)
  obs, _, _, _ = wrapped.step(act)
  pads = robot.data.site_pos_w[:, [li, ri]]
  ax = pads[:, 1] - pads[:, 0]
  jaw = torch.atan2(ax[:, 1], ax[:, 0])
  q = cube.data.root_link_quat_w
  cy = torch.atan2(
    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
  )
  base = robot.data.root_link_pos_w
  ee = pads.mean(1)
  rec.append(
    torch.stack(
      [
        robot.data.joint_pos[:, j7],
        cy,
        jaw,
        torch.norm((ee - base)[:, :2], dim=-1),  # horizontal reach
        cube.data.root_link_pos_w[:, 2],
      ],
      dim=-1,
    ).clone()
  )
R = torch.stack(rec).cpu().numpy()  # [T, N, 5]
env.close()

# Steady state: average over the second half, once the grasp is established.
half = R[STEPS // 2 :]
j7v = half[..., 0].mean(0)
cyv = half[..., 1].mean(0)
jawv = half[..., 2].mean(0)
reach = half[..., 3].mean(0)
peak = R[..., 4].max(0)


def fold(a):
  d = np.degrees(a % (np.pi / 2))
  return np.minimum(d, 90 - d)


yaw_err = fold(jawv - cyv)
# Fold both onto the jaw's 90 deg period so they are comparable.
cy_f = np.degrees(cyv % (np.pi / 2))
j7_f = np.degrees(j7v % (np.pi / 2))

print(f"{N} envs, steady state over the last {STEPS // 2} steps\n")
print(
  f"joint7 angle:  mean {np.degrees(j7v).mean():+7.1f} deg   "
  f"std {np.degrees(j7v).std():6.1f} deg   "
  f"range {np.degrees(j7v).min():+.1f} .. {np.degrees(j7v).max():+.1f}"
)
print(
  f"cube yaw:      std {np.degrees(cyv).std():6.1f} deg   "
  f"range {np.degrees(cyv).min():+.1f} .. {np.degrees(cyv).max():+.1f}"
)
c = np.corrcoef(cy_f, j7_f)[0, 1]
print(f"\ncorr(cube yaw, joint7), both folded to the jaw's 90 deg period: {c:+.3f}")
print("  ~+1 = joint7 is doing the aligning;  ~0 = joint7 ignores the cube")

print("\nhorizontal reach (pad midpoint from arm base):")
print(
  f"  mean {reach.mean() * 1000:6.1f} mm   std {reach.std() * 1000:5.1f} mm"
  f"   range {reach.min() * 1000:.0f} .. {reach.max() * 1000:.0f}"
)
print(f"corr(yaw error, reach):        {np.corrcoef(yaw_err, reach)[0, 1]:+.3f}")
print(f"corr(reach, peak cube height): {np.corrcoef(reach, peak)[0, 1]:+.3f}")
print(f"corr(yaw error, peak height):  {np.corrcoef(yaw_err, peak)[0, 1]:+.3f}")

lo = reach < np.median(reach)
print(
  f"\npeak height, near half  (reach < {np.median(reach) * 1000:.0f} mm):"
  f" {peak[lo].mean() * 1000:6.1f} mm"
)
print(
  f"peak height, far half   (reach >= {np.median(reach) * 1000:.0f} mm):"
  f" {peak[~lo].mean() * 1000:6.1f} mm"
)
