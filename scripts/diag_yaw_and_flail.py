"""Two measurements: why yaw matters, and what the finger flailing actually costs.

  uv run python scripts/diag_yaw_and_flail.py

(1) YAW. Close on a cube resting on a table at several jaw-vs-cube yaw angles and
    measure whether the grip holds (pull-out force) and whether the cube is
    displaced/rotated by the close. The claim to test is that an off-square jaw
    lands on a corner, where the two contact normals no longer oppose each other
    and the cube squirts out instead of being trapped.

(2) FLAILING. The trained policy holds the finger action std at 1.95 (arm dims
    are 0.06-0.61), i.e. a +-1.17 rad random command on each finger every control
    step. Replay exactly that -- Gaussian finger commands at the measured std,
    mean at the hover default -- and measure how far the cube is knocked. That is
    the cost the closing reward has to beat, and it is NOT the action-rate
    penalty: knocking the cube away destroys `lift`, which is worth 1.24 against
    pad_touch's 0.22.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

CUBE, MASS = 0.050, 0.05
WEIGHT = MASS * 9.81
TZ = -0.125  # table top, so the cube centre lands at the pinch height
HOVER_P, HOVER_D = 0.50, -0.40
PINCH_D = -1.05
SCALE = 0.6  # finger action scale
FINGER_STD = 1.95  # measured on the v8 checkpoint


def build():
  spec = mujoco.MjSpec.from_file(str(HAND_XML))
  t = spec.worldbody.add_body(name="table", pos=[0, 0.015, TZ - 0.05])
  t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.3, 0.05])
  b = spec.worldbody.add_body(name="cube", pos=[0, 0.015, TZ + CUBE / 2])
  b.add_freejoint(name="cj")
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
  for n in ("left_1", "left_2", "right_1", "right_2"):
    a = model.actuator(n)
    model.actuator_gainprm[a.id, 0] = FINGER_STIFFNESS
    model.actuator_biasprm[a.id, 1] = -FINGER_STIFFNESS
    model.actuator_biasprm[a.id, 2] = -FINGER_DAMPING
    model.actuator_forcerange[a.id] = (-FINGER_EFFORT_LIMIT, FINGER_EFFORT_LIMIT)
  return model, mujoco.MjData(model)


model, data = build()
cb, cg = model.body("cube").id, model.geom("cube_geom").id
START = np.array([0.0, 0.015, TZ + CUBE / 2])


def reset(yaw=0.0):
  mujoco.mj_resetData(model, data)
  for n, v in (
    ("left_1", HOVER_P),
    ("left_2", HOVER_D),
    ("right_1", -HOVER_P),
    ("right_2", -HOVER_D),
  ):
    data.qpos[model.joint(n).qposadr[0]] = v
  adr = model.jnt_qposadr[model.joint("cj").id]
  data.qpos[adr : adr + 3] = START
  data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def cmd(p, d):
  for n, v in (("left_1", p), ("left_2", d), ("right_1", -p), ("right_2", -d)):
    data.ctrl[model.actuator(n).id] = v


def contacts():
  n = 0
  for i in range(data.ncon):
    c = data.contact[i]
    gs = {c.geom1, c.geom2}
    if cg in gs and model.body(model.geom_bodyid[(gs - {cg}).pop()]).name in (
      "left_2",
      "right_2",
    ):
      n += 1
  return n


def pullout():
  s = data.xpos[cb].copy()
  hold = 0.0
  for pull in np.arange(0.25, 20.0, 0.25):
    data.xfrc_applied[cb] = [0, 0, -pull, 0, 0, 0]
    for _ in range(300):
      mujoco.mj_step(model, data)
    if np.linalg.norm(data.xpos[cb] - s) * 1000 > 5.0:
      break
    hold = pull
  data.xfrc_applied[cb] = 0.0
  return hold


print("(1) closing squarely vs off-square, cube resting on the table\n")
print(
  f"{'yaw':>6} {'presented':>10} {'contacts':>9} {'cube moved':>11}"
  f" {'cube rotated':>13} {'pull-out':>9} {'holds?':>7}"
)
for deg in (0, 10, 22.5, 30, 45):
  yaw = np.radians(deg)
  reset(yaw)
  for k in range(4000):
    a = min(1.0, k / 1500)
    cmd(HOVER_P, HOVER_D + a * (PINCH_D - HOVER_D))
    mujoco.mj_step(model, data)
  moved = np.linalg.norm(data.xpos[cb] - START) * 1000
  q = data.qpos[model.jnt_qposadr[model.joint("cj").id] + 3 :][:4]
  rot = np.degrees(2 * np.arccos(np.clip(abs(q[0]), 0, 1))) - deg
  nc = contacts()
  hold = pullout() if nc else 0.0
  print(
    f"{deg:6.1f} {50 * (np.cos(yaw) + np.sin(yaw)):10.1f} {nc:9d} {moved:11.1f}"
    f" {rot:+13.1f} {hold:8.2f}N {'YES' if hold > WEIGHT else 'no':>7}"
  )

print("\n(2) replaying the policy's measured finger flailing (std 1.95)\n")
print(
  f"{'std':>6} {'rad jitter':>11} {'cube moved mm':>14} {'cube still on table?':>21}"
)
rng = np.random.default_rng(0)
for std in (0.0, 0.3, 0.6, 1.0, 1.95):
  reset(0.0)
  for _ in range(400):  # 400 control steps
    d = HOVER_D + SCALE * std * rng.standard_normal()
    p = HOVER_P + SCALE * std * rng.standard_normal()
    cmd(p, d)
    for _ in range(20):
      mujoco.mj_step(model, data)
  moved = np.linalg.norm(data.xpos[cb] - START) * 1000
  onz = data.xpos[cb][2] > TZ
  print(
    f"{std:6.2f} {SCALE * std:11.2f} {moved:14.1f} {'yes' if onz and moved < 60 else 'NO -- gone':>21}"
  )
