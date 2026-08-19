"""Every number the deployed BENCH depends on, and where it came from.

What is NOT here: anything that changes with the checkpoint. The observation
layout, the action scale, the finger travel and the command slew limit are
properties of the policy rather than of the robot, and they live in
`deploy/profiles.py` keyed by the run that produced the weights. Everything in
this file is the same whichever checkpoint is flown.

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

# q_target = DEFAULT_JOINT_POS + profile.action_scale * action. There is no
# clipping anywhere in the trained pipeline (clip_actions is None), so the
# network's raw output is the command. The scale is per-checkpoint (round 2
# moved the fingers from 0.35 to 0.45) and lives in profiles.py; the home pose
# has not moved since the claw was fitted.
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

# ARM joint ranges off the compiled model, in JOINT_NAMES order. The trained
# pipeline never clips -- soft_joint_pos_limit_factor shapes a reward, it does
# not bound the command -- so on hardware these are the last line between a
# blown-up action and the arm's own hard stop.
#
# THE FINGERS ARE NOT HERE, and that is the fix rather than an omission. Their
# range is a TASK parameter the training env sets per arm (`_WIDE_DIRECTIONAL`,
# and round 2's wider replacement), not a property of the mechanism -- the URDF
# ships +-1.6 rad on all four, which is far more travel than the linkage wants
# on either side. In sim the configured range is a physical stop; on hardware
# nothing stops the command, so the clamp has to be the checkpoint's own range.
# `profiles.Profile.joint_limits` joins these seven to that checkpoint's four.
ARM_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
  (-2.8798, +2.8798),  # joint1
  (-2.3562, +2.3562),  # joint2
  (-3.0543, +3.0543),  # joint3
  (-1.9548, +2.7751),  # joint4
  (-3.0543, +3.0543),  # joint5
  (-1.4835, +4.6251),  # joint6
  (-3.0543, +3.0543),  # joint7
)

# Abort threshold on the RAW network output, not on the target. Trained actions
# run to about |5|; the out-of-distribution blow-up this catches was |600|. It
# has to be the raw action rather than "how far the target is from the measured
# angle", because a position servo legitimately lags its target during a reach
# and that distance is a measure of intent, not of malfunction.
#
# UNDER A SLEW LIMIT THIS IS THE ONLY GUARD THAT STILL SEES A BLOW-UP. Once
# the profile's rate limit is on the published target moves at most `rate / HZ` per
# step, so a 600 rad action and a 3 rad one produce the SAME first command and
# `--max-jump` cannot tell them apart. It only diverges after the limiter has
# spent several steps ramping in the wrong direction.
MAX_ABS_ACTION = 15.0

# The sim's joint servo, read off the compiled model (`actuator_gainprm[:, 0]`
# and `-actuator_biasprm[:, 2]`, which is what the exporter writes). Unlike
# DEFAULT_JOINT_POS these ARE safe to take from the ONNX metadata -- every value
# is an integer or 9.9, so the 3-decimal render is lossless -- and `policy.py`
# asserts them equal at load anyway.
#
# The ARM half is a command: `arm.FlexivArm` pushes it into RDK's
# SetJointImpedance, because that is the closest the real arm gets to the
# position servo the policy was trained against. The HAND half is informational
# only -- the gripper firmware runs its own servo and is tuned by a current cap
# (GRIP_CURRENT_CAP_A), not by a gain.
JOINT_STIFFNESS: tuple[float, ...] = (
  289.0,
  673.0,
  224.0,
  373.0,
  237.0,
  232.0,
  186.0,
  20.0,
  20.0,
  20.0,
  20.0,
)
JOINT_DAMPING: tuple[float, ...] = (
  61.0,
  143.0,
  36.0,
  59.0,
  13.0,
  12.0,
  9.9,
  4.0,
  4.0,
  4.0,
  4.0,
)

# Sim timestep 5 ms x decimation 4. The gripper firmware runs its own 200 Hz
# loop underneath and the binary protocol drops torque after 500 ms of silence,
# so 50 Hz sits comfortably inside both.
CONTROL_HZ = 50.0

# WHERE THE SLEW LIMIT WENT. `SmoothedJointPositionAction`'s rate_limit and
# ema_tau used to live here as one global setting an operator edited between
# checkpoints. They are `profiles.Profile.rate_limit` / `.ema_tau` now, because
# they are the checkpoint's property and not the bench's, and because the edit
# they replaced is the one nobody remembers to make. Read the value off the
# run's `params/env.yaml` under `actions.joint_pos.rate_limit`, and remember the
# curriculum scales it -- `Curriculum/rate_limit_schedule/rate_scale` in the
# run's tensorboard is the multiplier the FINAL checkpoint trained under. Every
# checkpoint that has a limiter configures the same one and logs rate_scale 1.0
# from its first iteration to its last, so the configured limits ARE the trained
# ones.
#
# The sim integrates the limit at its 200 Hz physics step and the robot host at
# CONTROL_HZ. Both bound the same rad/s, so the guarantee is identical; only the
# fine shape of the ramp inside one control period differs.

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

# The cube on the bench, EDGE in mm, measured with a caliper rather than taken
# from the scene. Round 2's students are told the object's size (as a HALF
# extent, in metres) and it is an observation like any other, so a wrong value
# here is an out-of-distribution input rather than a mis-set preference -- see
# `run.py --cube-mm`, which defaults to this and range-checks it against the
# checkpoint's own trained range.
CUBE_EDGE_MM = 50.0

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
#   real  homing offset is zeroed at the STRAIGHT / NEUTRAL pose -- mid-travel,
#         deliberately NOT a mechanical limit (gripper firmware/GUIDE.md 3a).
#         So 0 counts is roughly HALFWAY between open and closed on every motor,
#         and the closing direction is per-finger mixed sign.
#
# Do not read the poses out of the gripper's Config.cpp. `zero`, `setopen` and
# `setclose` write to the Teensy's EEPROM and `save` persists them, so Config.cpp
# is only the compiled fallback for an empty EEPROM -- and it is currently stale
# by a wide margin (it says POS_MIN/POS_MAX {-3337, 0, -1016, -1070} /
# {1074, 1017, 0, 1832}; the live values are about +-1200 on all four). Read them
# live with the `limits` CLI command instead.
#
# So the map is affine per motor, q_rad -> counts:  counts = COUNTS_PER_RAD * q
# + COUNTS_AT_ZERO_RAD, and both coefficients must be MEASURED. Fill these in
# from the calibration run; leaving them None makes hand.py refuse to arm.
#
# Motor order is the protocol's global order f1j1, f1j2, f2j1, f2j2 = ids
# 12, 11, 21, 22, which must be matched to (left_1, left_2, right_1, right_2) --
# WHICH PHYSICAL FINGER IS "left" IS ALSO A MEASUREMENT, not a convention. Get it
# backwards and the policy's asymmetric grasps mirror silently.
# PROVISIONAL. Good enough to fly the policy and see it move; NOT good enough to
# trust a grasp width or to conclude anything about why a grasp failed. What is
# solid and what is not:
#
#   OFFSET -- solid, and derived rather than measured. The gripper's homing zero
#     is set at the STRAIGHT / NEUTRAL pose, and in the compiled model that pose
#     IS q = 0: at q1 = 0 both proximal links sit at exactly 0.00 deg to
#     `hand_base_tf`, and the proximal->distal frame angle simply IS q2, so
#     q2 = 0 is exactly collinear. Real 0 counts and sim 0 rad are the same pose.
#
#   SIGN -- solid. Matching the taught poses against the sim's angles agrees on
#     all four: f1j1 OPEN +221 vs left_1 open +0.2746; f1j2 CLOSED -214 vs
#     left_2 curling negative; f2j1 OPEN -219 vs right_1 open -0.2746; f2j2
#     CLOSED +235 vs right_2 curling positive. All four scales are POSITIVE, and
#     the same comparison settles the finger identity: finger 1 is LEFT.
#
#   SCALE -- MEASURED 2026-08-11, and the 1:1 guess was badly wrong. It used to
#     be 651.9 = 4096 / 2pi, i.e. the assumption that the joint turns with the
#     motor's output shaft. `deploy/calibrate_finger_scale.py --method caliper`
#     swept the two proximals in mirror and read the jaw's outside width against
#     the compiled model's own width-vs-angle curve:
#
#       1028 +- 9 counts/rad   (n=5, residual rms 1.3 counts, max 2.1)
#
#     So there IS a linkage, at 1.58 : 1 from motor to joint, and the residual
#     says it is affine over the swept range rather than a four-bar needing a
#     table. Applied to all four motors on the operator's statement that the two
#     joints share one linkage design; only the proximals were swept.
#
#     WHAT THE OLD VALUE COST. Under 651.9 a commanded angle reached only 63% of
#     itself and the encoder read back 58% high, both making the claw NARROWER
#     than the policy believed. At the home pose the policy thinks it is holding
#     a 71.8 mm jaw open; it was actually holding 64.2 mm. A 50 mm cube presents
#     70.7 mm across its diagonal and the trained yaw is the full circle, so the
#     cube physically could not enter the jaw for 55% of placements -- which is
#     the 2/20 hardware success rate of the 2026-08-11 runs, not a policy fault.
#
#     The taught-endpoint evidence pointed the right way and understated it:
#     f1j1's taught open of +221 counts against the 1:1 prediction of 179 implies
#     805, where the direct measurement says 1028.
#
# Also measured that day, and NOT a calibration constant: the assembled jaw is
# 3.12 mm narrower than the model at every angle (inner gap 46.88 against 50.00,
# outside width 86.88 against 90.00, one finger exactly the CAD's 20.00 mm
# thick). `outer - inner` being exactly twice the thickness only holds when the
# fingers are parallel, so the zero below is confirmed rather than merely
# assumed, and the two fingers simply sit 1.56 mm inboard of where
# `two_finger_hand.xml` puts their knuckles. That belongs in the asset, not here.
HAND_CALIB_IS_PROVISIONAL = False
COUNTS_PER_RAD: tuple[float, float, float, float] | None = (
  1028.0,
  1028.0,
  1028.0,
  1028.0,
)
COUNTS_AT_ZERO_RAD: tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 0.0)

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
