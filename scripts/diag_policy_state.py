"""What is the trained policy's hand ACTUALLY doing, step by step?

  uv run python scripts/diag_policy_state.py TASK CKPT

Run 46846 stalled with pad_touch at 0.22 and grasp at exactly 0, and every
armchair explanation has now failed a measurement:

  - "closing would knock the cube away" -- no: closing onto a cube resting on
    the table moves it 0.1 mm and makes 2 contacts, at every closing speed
  - "closing does not pay at the height it parks at" -- no: closing is worth
    +0.23 to +0.81 of pad_touch at every height from 0 to 50 mm too high
  - "exploration collapsed" -- no: action std holds at 0.94 and the close needs
    1.08 sigma

So stop guessing and read the state. Logs, per step, the distal joint angle the
policy commands, how far each pad site is from the cube surface, the hand height
over the cube, and the cube's yaw relative to the jaw -- the last because every
calculation so far assumed a jaw squared to a cube face, and cube yaw is
randomised over the full circle.
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, STEPS, DEV = 64, 200, "cuda:0"
HALF = 0.025

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
agent_cfg = load_rl_cfg(TASK)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=DEV)
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

robot, cube = env.scene["robot"], env.scene["cube"]
site_names = list(robot.site_names)
li, ri = site_names.index("left_pad"), site_names.index("right_pad")
jn = list(robot.joint_names)
jl1, jl2 = jn.index("left_1"), jn.index("left_2")


def box_sdf(pts, c, q):
  """Distance from world points to the cube surface, in the cube's own frame."""
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  R = torch.stack(
    [
      1 - 2 * (y * y + z * z),
      2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z - w * x),
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
    ],
    dim=-1,
  ).reshape(-1, 3, 3)
  loc = torch.einsum("bij,bj->bi", R.transpose(1, 2), pts - c)
  d = loc.abs() - HALF
  return torch.norm(d.clamp(min=0), dim=-1) + d.max(dim=-1).values.clamp(max=0)


torch.manual_seed(0)
obs = wrapped.reset()[0]
rows = []
for t in range(STEPS):
  with torch.inference_mode():
    act = policy(obs)
  obs, _, _, _ = wrapped.step(act)
  pads = robot.data.site_pos_w[:, [li, ri]]
  cpos, cquat = cube.data.root_link_pos_w, cube.data.root_link_quat_w
  dl = box_sdf(pads[:, 0], cpos, cquat)
  dr = box_sdf(pads[:, 1], cpos, cquat)
  q = robot.data.joint_pos
  # Jaw yaw vs cube yaw, folded into [0,45] deg by the cube's 4-fold symmetry.
  ax = pads[:, 1] - pads[:, 0]
  jaw = torch.atan2(ax[:, 1], ax[:, 0])
  w, x, y, z = cquat[:, 0], cquat[:, 1], cquat[:, 2], cquat[:, 3]
  cy = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  mis = torch.rad2deg(((jaw - cy) % (np.pi / 2)))
  mis = torch.minimum(mis, 90.0 - mis)
  rows.append(
    (
      t,
      q[:, jl1].mean().item(),
      q[:, jl2].mean().item(),
      q[:, jl2].min().item(),
      torch.norm(pads[:, 1] - pads[:, 0], dim=-1).mean().item() * 1000,
      dl.mean().item() * 1000,
      dr.mean().item() * 1000,
      (pads[:, :, 2].mean(1) - cpos[:, 2]).mean().item() * 1000,
      mis.mean().item(),
    )
  )

print(f"{N} envs, {STEPS} steps.  distal hover default = -0.40, pinch = -1.05\n")
print(
  f"{'step':>5} {'prox':>7} {'distal':>7} {'dist.min':>9} {'tip sep':>8}"
  f" {'dL':>7} {'dR':>7} {'tip-cube z':>11} {'yaw err':>8}"
)
for r in rows[:: max(1, STEPS // 12)]:
  print(
    f"{r[0]:5d} {r[1]:+7.3f} {r[2]:+7.3f} {r[3]:+9.3f} {r[4]:8.1f}"
    f" {r[5]:7.1f} {r[6]:7.1f} {r[7]:+11.1f} {r[8]:8.1f}"
  )
last = rows[-1]
print(
  f"\nfinal: distal {last[2]:+.3f} (needs -1.05), tip separation {last[4]:.1f} mm"
  f" (cube is 50), pad-to-surface {last[5]:.1f}/{last[6]:.1f} mm,"
  f" tips {last[7]:+.1f} mm above cube centre, jaw {last[8]:.1f} deg off a face"
)
