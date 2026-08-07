"""How hard can you pull the cube out of the grip, at each closing depth?

  uv run python scripts/diag_pullout.py

The whole "grip harder" question hangs on a number nobody has measured for the
current geometry: the grip FORCE is known (47-61 N per side at deep curl) but
force is not what holds an object -- friction is, and friction is what the
pull-out test measures. If a modest closing depth already holds many times the
cube's 0.49 N weight, then the reason the policy tips and drags the cube is not
that it cannot squeeze hard enough, and chasing grip force is the wrong fix.

Method: close on the cube, then ramp a downward force on it until it moves more
than 5 mm relative to where it settled. The last force it survived is the
pull-out force. The arm is absent, so this is the gripper alone.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  _GRASP_SITE_POS,
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

CUBE, MASS = 0.050, 0.05
HOVER = {"left_1": 0.50, "left_2": -0.40, "right_1": -0.50, "right_2": 0.40}
CASES = ((0.10, -0.40), (0.00, -0.60), (-0.10, -0.90), (-0.10, -1.20), (-0.20, -1.50))
SLIP_MM = 5.0

spec = mujoco.MjSpec.from_file(str(HAND_XML))
b = spec.worldbody.add_body(name="cube", pos=list(_GRASP_SITE_POS))
b.add_freejoint(name="cube_joint")
g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3, mass=MASS)
g.name = "cube_geom"
for a, b_ in (
  ("base_link", "left_1"),
  ("base_link", "right_1"),
  ("left_1", "left_2"),
  ("right_1", "right_2"),
):
  e = spec.add_exclude()
  e.bodyname1, e.bodyname2 = a, b_

model = spec.compile()
for n in HOVER:
  a = model.actuator(n)
  model.actuator_gainprm[a.id, 0] = FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 1] = -FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 2] = -FINGER_DAMPING
  model.actuator_forcerange[a.id] = (-FINGER_EFFORT_LIMIT, FINGER_EFFORT_LIMIT)
# No gravity: the pull is applied explicitly, so weight would just be an
# uncontrolled offset on top of it.
model.opt.gravity[:] = 0.0
cube_bid = model.body("cube").id
cube_gid = model.geom("cube_geom").id
data = mujoco.MjData(model)

print(f"cube {CUBE * 1000:.0f} mm, {MASS * 1000:.0f} g -> weight {MASS * 9.81:.2f} N")
print(f"slip = {SLIP_MM:.0f} mm of movement relative to the settled grip\n")
print(
  f"{'cmd p/d':>14} {'settled p/d':>16} {'Fn/side':>9} {'pull-out':>10} {'x weight':>9}"
)

for p_cmd, d_cmd in CASES:
  mujoco.mj_resetData(model, data)
  for n, v in HOVER.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  data.qpos[model.jnt_qposadr[model.joint("cube_joint").id] :][:3] = _GRASP_SITE_POS
  for n, v in (
    ("left_1", p_cmd),
    ("left_2", d_cmd),
    ("right_1", -p_cmd),
    ("right_2", -d_cmd),
  ):
    data.ctrl[model.actuator(n).id] = v
  for _ in range(4000):
    mujoco.mj_step(model, data)

  fn = 0.0
  for i in range(data.ncon):
    c = data.contact[i]
    gs = {c.geom1, c.geom2}
    if cube_gid not in gs:
      continue
    if model.body(model.geom_bodyid[(gs - {cube_gid}).pop()]).name != "left_2":
      continue
    f = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f)
    fn += abs(f[0])
  if fn == 0.0:
    print(f"{p_cmd:+6.2f}/{d_cmd:+6.2f}   (no grip)")
    continue

  settled = data.xpos[cube_bid].copy()
  # Control: how far does the cube drift under NO pull? If this already exceeds
  # the slip threshold the test is measuring the grip settling, not pull-out.
  for _ in range(400):
    mujoco.mj_step(model, data)
  drift = np.linalg.norm(data.xpos[cube_bid] - settled) * 1000
  hold = 0.0
  for pull in np.arange(0.5, 60.0, 0.5):
    data.xfrc_applied[cube_bid] = [0, 0, -pull, 0, 0, 0]
    for _ in range(400):
      mujoco.mj_step(model, data)
    if np.linalg.norm(data.xpos[cube_bid] - settled) * 1000 > SLIP_MM:
      break
    hold = pull
  data.xfrc_applied[cube_bid] = 0.0
  q = [data.qpos[model.joint(n).qposadr[0]] for n in ("left_1", "left_2")]
  print(
    f"{p_cmd:+6.2f}/{d_cmd:+6.2f}  {q[0]:+7.3f}/{q[1]:+7.3f}  {fn:8.1f}N "
    f"{hold:9.1f}N {hold / (MASS * 9.81):8.0f}x   drift@0N {drift:5.2f} mm"
  )
