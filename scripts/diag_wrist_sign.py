"""Which way does joint7 turn the jaw, and by how much per radian?

  uv run python scripts/diag_wrist_sign.py

A closed-form wrist bias needs two facts that are not open to guessing: the SIGN
of d(jaw yaw)/d(joint7), and whether the ratio is 1.0. Getting the sign wrong
turns a corrective feedback term into a divergent one, so this measures both from
the model rather than reasoning about which way the tool frame points.

Jaw yaw is the world-frame heading of the left_pad -> right_pad vector, i.e. the
same quantity every alignment diagnostic in this project uses.
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
li, ri = sn.index("left_pad"), sn.index("right_pad")
default = robot.data.default_joint_pos
assert default is not None
base = float(default[0, j7])


def jaw_yaw_at(q7: float) -> float:
  pos = default.clone()
  pos[:, j7] = q7
  robot.write_joint_position_to_sim(pos)
  env.sim.forward()
  pads = robot.data.site_pos_w[:, [li, ri]]
  ax = pads[:, 1] - pads[:, 0]
  return float(torch.atan2(ax[:, 1], ax[:, 0])[0])


print(f"joint7 default = {np.degrees(base):+.2f} deg\n")
print(f"{'joint7 (deg)':>13} {'jaw yaw (deg)':>14} {'d(jaw)/d(j7)':>13}")
qs = base + np.radians([-40.0, -20.0, 0.0, 20.0, 40.0])
prev = None
slopes = []
for q in qs:
  y = jaw_yaw_at(float(q))
  s = ""
  if prev is not None:
    k = (np.unwrap([prev[1], y])[1] - prev[1]) / (q - prev[0])
    slopes.append(k)
    s = f"{k:+13.3f}"
  print(f"{np.degrees(q):+12.1f} {np.degrees(y):+13.1f} {s}")
  prev = (q, y)

k = float(np.mean(slopes))
print(f"\nd(jaw yaw)/d(joint7) = {k:+.3f} rad/rad")
print("bias for a jaw-yaw error e is  -e / k  applied to joint7's position target")
env.close()
