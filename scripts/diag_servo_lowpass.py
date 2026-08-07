"""Can the finger actually follow the command the policy is sending it?

  uv run python scripts/diag_servo_lowpass.py

The claim to check: at the learned action std of 1.95 the policy hands the finger
a brand-new random target every control step (20 ms), and the position servo is
slow enough that the joint never arrives anywhere -- it just dithers near the
mean, so no single step's action changes the outcome and no gradient forms.

Three measurements, all on the real actuator settings (kp = FINGER_STIFFNESS,
kd = FINGER_DAMPING, torque capped at FINGER_EFFORT_LIMIT):

  A. step response -- hold the pinch command and time how long the joint takes
     to get there. That is the servo's own speed, with nothing else going on.
  B. tracking -- drive it with iid Gaussian targets at several stds and compare
     the spread of the COMMAND with the spread of the JOINT that results.
  C. dwell -- what fraction of the time the joint is actually within 0.1 rad of
     the pinch angle, which is the only state that earns the closing reward.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

DT, DECIM = 0.005, 4  # 50 Hz control, from lift_cube_env_cfg
CTRL_DT = DT * DECIM
HOVER_D, PINCH_D = -0.40, -1.05
LO, HI = -1.60, -0.40  # the one-way box on the distal
SCALE = 0.6

spec = mujoco.MjSpec.from_file(str(HAND_XML))
model = spec.compile()
model.opt.timestep = DT
for n in ("left_1", "left_2", "right_1", "right_2"):
  a = model.actuator(n)
  model.actuator_gainprm[a.id, 0] = FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 1] = -FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 2] = -FINGER_DAMPING
  model.actuator_forcerange[a.id] = (-FINGER_EFFORT_LIMIT, FINGER_EFFORT_LIMIT)
data = mujoco.MjData(model)
adr = model.joint("left_2").qposadr[0]
aid = model.actuator("left_2").id


def reset():
  mujoco.mj_resetData(model, data)
  for n, v in (
    ("left_1", 0.50),
    ("left_2", HOVER_D),
    ("right_1", -0.50),
    ("right_2", -HOVER_D),
  ):
    data.qpos[model.joint(n).qposadr[0]] = v
    data.ctrl[model.actuator(n).id] = v


print(
  f"control step {CTRL_DT * 1000:.0f} ms; kp={FINGER_STIFFNESS}, "
  f"kd={FINGER_DAMPING}, torque limit {FINGER_EFFORT_LIMIT} N.m\n"
)

# --- A. step response -------------------------------------------------------
reset()
data.ctrl[aid] = PINCH_D
target_move = abs(PINCH_D - HOVER_D)
t63 = t95 = None
for k in range(4000):
  mujoco.mj_step(model, data)
  frac = abs(data.qpos[adr] - HOVER_D) / target_move
  t = (k + 1) * DT
  if t63 is None and frac >= 0.632:
    t63 = t
  if t95 is None and frac >= 0.95:
    t95 = t
    break
print("A. step response: hold the pinch command and wait")
print(
  f"   63% of the way there: {t63 * 1000:6.0f} ms  = {t63 / CTRL_DT:5.1f} control steps"
)
print(
  f"   95% of the way there: {t95 * 1000:6.0f} ms  = {t95 / CTRL_DT:5.1f} control steps"
)
print(
  f"   -> the joint needs ~{t95 / CTRL_DT:.0f} steps of a CONSISTENT command to arrive.\n"
)

# --- B/C. tracking under iid noise ------------------------------------------
print("B. tracking: iid Gaussian action each control step, mean 0 (= stay at hover)")
print(
  f"{'action std':>11} {'cmd rad std':>12} {'JOINT rad std':>14} {'joint mean':>11}"
  f" {'attenuation':>12} {'dwell at pinch':>15}"
)
rng = np.random.default_rng(0)
for std in (0.3, 0.6, 1.0, 1.95):
  reset()
  cmds, qs = [], []
  for _ in range(1500):  # 30 s of control
    tgt = float(np.clip(HOVER_D + SCALE * std * rng.standard_normal(), LO, HI))
    data.ctrl[aid] = tgt
    for _ in range(DECIM):
      mujoco.mj_step(model, data)
    cmds.append(tgt)
    qs.append(data.qpos[adr])
  cmds, qs = np.array(cmds), np.array(qs)
  dwell = float(np.mean(np.abs(qs - PINCH_D) < 0.10)) * 100
  print(
    f"{std:11.2f} {cmds.std():12.3f} {qs.std():14.3f} {qs.mean():+11.3f}"
    f" {cmds.std() / max(qs.std(), 1e-9):11.1f}x {dwell:14.1f}%"
  )

reset()
for _ in range(1500):
  data.ctrl[aid] = PINCH_D
  for _ in range(DECIM):
    mujoco.mj_step(model, data)
print(
  f"{'sustained':>11} {0.0:12.3f} {0.0:14.3f} {data.qpos[adr]:+11.3f}"
  f" {'--':>11}  {100.0:14.1f}%"
)
print("\nC. 'dwell at pinch' = fraction of time the joint is within 0.1 rad of -1.05,")
print("   which is the only state that collects the closing reward.")
