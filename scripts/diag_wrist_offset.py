"""Does rolling joint7 move the jaw off the grasp site?

  uv run python scripts/diag_wrist_offset.py

Run 49073 (SolvedWrist) plateaued with `pad_touch` at 0.55 against the baseline's
0.88 -- it never learned to put the pads on the cube, even though the solver
demonstrably squares the jaw (2.8 deg mean error) and does not chatter (0 slams).

The remaining suspect is geometric: if the wrist roll axis does not pass through
the jaw centre, then rolling to square the jaw also TRANSLATES it, and the arm
has to chase that translation with a lateral move that depends on cube yaw. The
baseline never had to -- its wrist was frozen, and a 100 mm jaw straddles a
50 mm cube's 70.7 mm diagonal at any yaw, so `pad_touch` was reachable without
aligning at all.

Measures the pad midpoint and the grasp site against joint7, arm otherwise at
the default pose.
"""

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalmAlign"
DEV = "cuda:0"

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
env.reset()

robot = env.scene["robot"]
j7 = list(robot.joint_names).index("joint7")
sn = list(robot.site_names)
li, ri, gi = sn.index("left_pad"), sn.index("right_pad"), sn.index("grasp_site")
default = robot.data.default_joint_pos
assert default is not None
base = float(default[0, j7])


def probe(q7: float):
  pos = default.clone()
  pos[:, j7] = q7
  robot.write_joint_position_to_sim(pos)
  env.sim.forward()
  p = robot.data.site_pos_w[0]
  mid = ((p[li] + p[ri]) / 2).cpu().numpy()
  return mid, p[gi].cpu().numpy(), float(torch.norm(p[ri] - p[li]))


print(f"joint7 default = {np.degrees(base):+.1f} deg, arm otherwise at default\n")
print(
  f"{'joint7':>8} {'pad midpoint (mm, rel to roll=0)':>34}"
  f" {'|mid - grasp_site| mm':>22} {'jaw mm':>8}"
)
ref = None
rows = []
for d in (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0):
  mid, gs, jaw = probe(base + np.radians(d))
  if ref is None and d == -45.0:
    pass
  rows.append((d, mid, gs, jaw))
ref = [r for r in rows if r[0] == 0.0][0][1]
for d, mid, gs, jaw in rows:
  dm = (mid - ref) * 1000
  print(
    f"{d:+7.0f}d   ({dm[0]:+7.1f},{dm[1]:+7.1f},{dm[2]:+7.1f})"
    f" {np.linalg.norm(mid - gs) * 1000:21.1f} {jaw * 1000:7.1f}"
  )

span = max(np.linalg.norm((r[1] - ref)[:2]) for r in rows) * 1000
print(f"\nmax lateral swing of the jaw centre over +-45 deg of roll: {span:.1f} mm")
print("The cube is 50 mm wide and pad_touch has std 30 mm, so a swing of that")
print("order means squaring the jaw and reaching the cube are competing moves.")
env.close()
