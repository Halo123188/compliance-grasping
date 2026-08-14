"""Every number the deployed policy depends on, and where it came from.

Split into three blocks by PROVENANCE, because that is what decides what you may
change:

  FROM THE TRAINED POLICY  -- read out of the ONNX metadata at load time and
    asserted against the copies here. Changing one of these silently makes the
    policy wrong in a way nothing will report. They are duplicated here only so
    a mismatch raises instead of drifting.

  FROM THE SIMULATED SCENE -- geometry the policy was trained against. These are
    claims about the REAL bench, and the real bench has to match them or the
    policy is being asked a different question than it was trained on.

  MEASURED ON HARDWARE     -- cannot be derived from either repo. Left as None
    so that importing this module and forgetting to calibrate raises, rather
    than deploying with a plausible-looking guess.
"""

from __future__ import annotations

# --- FROM THE TRAINED POLICY -------------------------------------------------
# Action order == observation joint order == MuJoCo model joint order. Verified
# by compiling the model: all three are this list. The real robot must report
# and accept joint vectors in exactly this order.
JOINT_NAMES: tuple[str, ...] = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
  "left_1",
  "left_2",
  "right_1",
  "right_2",
)
ARM_SLICE = slice(0, 7)
HAND_SLICE = slice(7, 11)

# q_target = DEFAULT_JOINT_POS + ACTION_SCALE * action. There is no clipping
# anywhere in the trained pipeline (clip_actions is None), so the network's raw
# output is the command.
# FULL PRECISION, from TWOFINGER_ARM_HOME -- not from the ONNX metadata, which
# `list_to_csv_str` writes at 3 decimals. Reading them off the metadata makes
# every joint offset wrong by up to 5e-4 rad, which is small enough to look like
# nothing and is exactly what a parity check against the env catches.
DEFAULT_JOINT_POS: tuple[float, ...] = (
  0.0,
  -0.3685,
  0.217,
  2.3382,
  -0.1801,
  1.1223,
  1.1535,
  0.2746,
  0.0,
  -0.2746,
  0.0,
)
ACTION_SCALE: tuple[float, ...] = (
  0.40,
  0.40,
  0.40,
  0.40,
  0.40,
  0.40,
  1.10,
  0.35,
  0.35,
  0.35,
  0.35,
)

# Joint ranges off the compiled model, in JOINT_NAMES order. The trained
# pipeline never clips -- soft_joint_pos_limit_factor shapes a reward, it does
# not bound the command -- so on hardware these are the last line between a
# blown-up action and the arm's own hard stop.
JOINT_LIMITS: tuple[tuple[float, float], ...] = (
  (-2.8798, +2.8798),  # joint1
  (-2.3562, +2.3562),  # joint2
  (-3.0543, +3.0543),  # joint3
  (-1.9548, +2.7751),  # joint4
  (-3.0543, +3.0543),  # joint5
  (-1.4835, +4.6251),  # joint6
  (-3.0543, +3.0543),  # joint7
  (-1.6000, +1.6000),  # left_1
  (-1.6000, +1.6000),  # left_2
  (-1.6000, +1.6000),  # right_1
  (-1.6000, +1.6000),  # right_2
)

# Abort threshold on the RAW network output, not on the target. Trained actions
# run to about |5|; the out-of-distribution blow-up this catches was |600|. It
# has to be the raw action rather than "how far the target is from the measured
# angle", because a position servo legitimately lags its target during a reach
# and that distance is a measure of intent, not of malfunction.
MAX_ABS_ACTION = 15.0

# Sim timestep 5 ms x decimation 4. The gripper firmware runs its own 200 Hz
# loop underneath and the binary protocol drops torque after 500 ms of silence,
# so 50 Hz sits comfortably inside both.
CONTROL_HZ = 50.0

# Command slew limit, rad/s, in JOINT_NAMES order -- the deployment half of the
# training env's `SmoothedJointPositionAction`. None means no limiting, which is
# what every checkpoint trained before the speed work expects.
#
# THIS MUST MATCH THE CHECKPOINT'S OWN TRAINING SETTING and nothing checks it
# for you: the ONNX metadata carries default_joint_pos and action_scale but not
# this. Set it from the task id the checkpoint came from:
#
#   ...-Success-Dr, -Slow          None      (no limiter in the action term)
#   ...-Success-Dr-Slew, -SlewCurr (1.0 x6, 1.5, 3.0 x4)   <- the value below
#   ...-Success-Dr-SlewTight       (0.5 x6, 1.0, 2.0 x4)
#
# GETTING THIS WRONG IS NOT A DEGRADATION, IT IS A CRASH. A policy trained with
# a limiter learns to lean on it: measured on the -SlewCurr teacher
# (scripts/wide_speed_audit.py, job 61221) the target it ASKS for moves 16.7x
# faster than the one the limiter publishes, with a peak request of 135 rad/s
# against a 1.5 rad/s cap -- and -SlewTight asks for 292 rad/s. Publish that
# unfiltered and the arm gets a full-speed command into the bench on the first
# step. The reverse mistake is merely bad: limiting a policy trained without one
# makes it lag its own plan.
#
# The sim integrates this at its 200 Hz physics step and the robot host at
# CONTROL_HZ. Both bound the same rad/s, so the guarantee is identical; only the
# fine shape of the ramp inside one control period differs.
RATE_LIMIT: tuple[float, ...] | None = (
  1.0,  # joint1
  1.0,  # joint2
  1.0,  # joint3
  1.0,  # joint4
  1.0,  # joint5
  1.0,  # joint6
  1.5,  # joint7, the wrist roll -- needs +-45 deg of travel before contact
  3.0,  # left_1
  3.0,  # left_2
  3.0,  # right_1
  3.0,  # right_2
)

# First-order low-pass time constant on the command, seconds, or None. The
# deployment half of `SmoothedJointPositionAction`'s `ema_tau`; the ...-Ema arm
# trains with 0.09. Same matching rule as RATE_LIMIT.
EMA_TAU: float | None = None

# Depth observation, exactly as `manipulation_mdp.camera_depth` builds it:
#   clamp(metres, MIN, CUTOFF) / CUTOFF   -> float32 in [0, 1], shape (1,120,160)
DEPTH_CUTOFF_M = 3.0
DEPTH_MIN_M = 0.01
DEPTH_HW = (120, 160)

# The D435 recipe the sim camera was deliberately sized for (see the comment on
# CameraSensorCfg in env_cfgs.py): the intrinsics are the 848x480 depth
# profile's, and a 4:3 render centre-crops that to exactly 640x480, so
#     848x480  ->  centre-crop 640x480  ->  resize 160x120
# reproduces the 73.53 x 58.53 deg field the student learned. Streaming 16:9
# instead squashes 89.42 deg into pixels it learned as 73.53.
D435_STREAM_WH = (848, 480)
D435_CROP_WH = (640, 480)

# --- FROM THE SIMULATED SCENE ------------------------------------------------
# Heights are metres above the FLOOR, which is the frame `goal_height` is in.
WORK_SURFACE_Z = 0.400  # top of the foam; cube height is measured from here
RESTING_Z = 0.425  # centre of a 50 mm cube sitting on it
ARM_BASE_Z = 0.365  # top of the arm's mounting plate

# goal_height is the one non-privileged part of the lift command: the operator
# supplies it. Trained over WORK_SURFACE_Z + 0.10 .. 0.20; success is scored at
# +0.10. In sim it RESAMPLES every 5 s -- for deployment hold it fixed.
GOAL_HEIGHT_M = WORK_SURFACE_Z + 0.15
GOAL_HEIGHT_RANGE = (WORK_SURFACE_Z + 0.10, WORK_SURFACE_Z + 0.20)

# Pad separation is affine in the proximal joint angle, fitted off the compiled
# model (residual < 0.25 mm over q in -0.15..0.40):
#     sep(q) = APERTURE_A + APERTURE_B * q     [mm]
# The open default (+0.2746) is 86.9 mm; full close (-0.0754) is 40.0 mm, i.e.
# a ~10 mm squeeze on a 50 mm cube rather than 52 mm of drive-through.
APERTURE_A_MM = 50.1
APERTURE_B_MM = 134.0

# Grip force cap. THE SIM'S FINGER ACTUATOR IS STRONGER THAN THE REAL MOTOR.
# Sim: position servo, kp 20, kv 4, forcerange +-2.0 N.m per finger joint (the
# URDF's effort field). Real: XC330-M181-T, stall 1.8 A at 5.0 V, and the
# firmware ships DEFAULT_CURRENT_CAP_A = 0.8 (~44% of stall) with a hard ceiling
# of 1.5 A. Those are different quantities in different units and NOTHING in
# either repo relates them. Until someone measures pad force vs cap on the real
# gripper, start at the firmware default and treat any "the policy grips harder
# in sim" observation as expected rather than as a bug.
GRIP_CURRENT_CAP_A = 0.8

# --- MEASURED ON HARDWARE ----------------------------------------------------
# None of the following can be derived. `deploy/calibrate_hand.py` measures them.
#
# The sim's finger joints and the real gripper's motors are both "4 joints, 2
# per finger", but they do not share a zero, a sign, or a scale:
#
#   sim   left_1 = +0.2746 rad is OPEN (86.9 mm), -0.0754 rad is CLOSED (40 mm);
#         left_2 / right_2 (the distal curl) default to 0.
#   real  homing offset was zeroed at the mechanical OPEN limit, so 0 counts is
#         fully open on EVERY motor, and the closing direction is per-motor
#         mixed sign (Config.cpp: POS_MIN/POS_MAX are
#         {-3337, 0, -1016, -1070} / {1074, 1017, 0, 1832}).
#
# So the map is affine per motor, q_rad -> counts:  counts = COUNTS_PER_RAD * q
# + COUNTS_AT_ZERO_RAD, and both coefficients must be MEASURED. Fill these in
# from the calibration run; leaving them None makes hand.py refuse to arm.
#
# Motor order is the protocol's global order f1j1, f1j2, f2j1, f2j2 = ids
# 12, 11, 21, 22, which must be matched to (left_1, left_2, right_1, right_2) --
# WHICH PHYSICAL FINGER IS "left" IS ALSO A MEASUREMENT, not a convention. Get it
# backwards and the policy's asymmetric grasps mirror silently.
COUNTS_PER_RAD: tuple[float, float, float, float] | None = None
COUNTS_AT_ZERO_RAD: tuple[float, float, float, float] | None = None

# Camera extrinsic, and the single most expensive number in the whole pipeline.
# Ablation on the trained student (scripts/wide_dr_ablate.py, one checkpoint
# re-scored under narrower envs) attributed the camera-DR cost per axis:
#
#   depth bias   1.4 mm -> 88.3% success        2.6 mm -> 67.2%
#
# i.e. error ALONG THE OPTICAL AXIS is a threshold, not a proportionality, and
# the knee sits between those two. Error perpendicular to it is nearly free
# (90.6%). Depth is metric, so sliding the camera along its own view direction
# offsets every reading and the policy has no way to know.
#
# Calibrate the camera's position along its optical axis to better than ~1.5 mm.
# If you can measure the residual offset, subtract it here (metres, positive =
# camera is FURTHER from the scene than the model says) rather than hoping the
# policy absorbs it -- a constant you can measure is not something a policy
# should be spending capacity on.
DEPTH_AXIS_OFFSET_M = 0.0
