"""Domain randomization for the wide claw on the foam bench.

Split into two halves, because they cost different things to add and they are
consumed by different policies:

  PHYSICS (`add_physics_dr`)     mass, inertia, friction, servo gains, joint
                                 friction/damping, encoder zeros, bench height,
                                 start pose. The teacher sees all of it, so
                                 turning it on means retraining the teacher.
  PERCEPTION (`add_percept_dr`)  camera pose, field of view, depth noise and
                                 dropout. The teacher reads privileged state
                                 and is blind to every one of these, so this
                                 half can be added at distillation time against
                                 an ALREADY-TRAINED teacher.

That asymmetry is the whole reason for the split: the expensive half is the one
that does not have to be repeated.

WHAT WAS ALREADY ON before any of this, and is left alone: cube spawn pose
(+-80/+-100 mm and full yaw), fingertip friction at startup, and observation
noise on joint_pos / joint_vel / the teacher's privileged terms.

Ranges are the ones measured upstream in cubegrasp-env unless a comment says
otherwise; three of them are deliberately WIDER than upstream and each says why.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  ARM_ACTUATOR_GROUP,
  FINGER_ACTUATOR_GROUP,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

# Bodies whose inertial properties are CAD estimates of printed parts rather
# than weighed values -- and stale ones at that, since the servo horns were cut
# off after the CAD was exported.
HAND_BODIES = (r"hand_base_tf", r"(left|right)_[12]_tf")
ARM_JOINT_PATTERN = r"joint[1-7]"

# The cube's compile-time mass, from scene.get_cube_spec. pseudo_inertia scales
# mass by exp(2*alpha), so a mass RANGE has to be converted into an alpha range
# rather than written directly -- and it must be done against this number.
# Command latency destabilizes a position servo that was tuned without it, so it
# gets its own switch: a run that goes unstable has to be attributable to this
# rather than to "the physics DR" as a lump. Read before `add_physics_dr` uses it.
CMD_LATENCY_DR = (os.environ.get("CG_WIDE_CMD_LATENCY_DR") or "1") != "0"

_CUBE_MASS = 0.06
_CUBE_MASS_RANGE = (0.03, 0.15)  # kg, upstream `dr_cube_mass`
_CUBE_SCALE_RANGE = (0.9, 1.1)  # multiplier on the 50 mm half-edge

# Scale bands, as multipliers, converted to alpha the same way.
_HAND_MASS_SCALE = (0.6, 1.4)  # upstream `dr_hand_mass`


def _alpha(lo: float, hi: float) -> tuple[float, float]:
  """Mass-scale band -> pseudo_inertia alpha band. mass' = mass * exp(2a)."""
  return (0.5 * math.log(lo), 0.5 * math.log(hi))


def add_physics_dr(
  cfg: ManagerBasedRlEnvCfg,
  cube_mass_range: tuple[float, float] | None = None,
  cube_scale_range: tuple[float, float] | None = None,
) -> None:
  """Randomize what the arm and the object are made of. Mutates ``cfg``.

  ``cube_mass_range`` / ``cube_scale_range`` override the defaults below. They
  are parameters rather than module constants so a task can widen them and have
  the widening recorded in its task id -- the same reason the collision set is
  threaded rather than read from the environment.
  """
  mass_range = cube_mass_range or _CUBE_MASS_RANGE
  scale_range = cube_scale_range or _CUBE_SCALE_RANGE
  # --- the object -----------------------------------------------------------
  # `reset` rather than `startup`: a real trial can present a different object,
  # so the mass and the friction should differ episode to episode. Everything
  # below that describes the ROBOT is `startup` instead, because the real robot
  # has one unknown value for each of those, not a fresh draw per grasp.
  #
  # pseudo_inertia, not body_mass. dr.body_mass changes ONLY the mass and leaves
  # the inertia tensor at its 0.06 kg value -- across a 5x mass range that is
  # not a heavier cube, it is a cube whose rotational and translational inertia
  # disagree, and the grasp is decided by exactly that coupling when the jaw
  # tips it.
  cfg.events["dr_cube_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
      "alpha_range": _alpha(mass_range[0] / _CUBE_MASS, mass_range[1] / _CUBE_MASS),
    },
  )

  # SIZE, and it must come after the inertia draw above: `object_scale`
  # multiplies whatever `body_inertia` currently holds, and `pseudo_inertia`
  # rewrites that field absolutely from defaults on every reset.
  #
  # +-10% is a chosen band, not a measured one. It is bounded by the reward
  # constants rather than by the claw: the aperture law `sep(q) = 50.1 + 134.0q`
  # spans 40.0-86.9 mm, so the hand could hold a far wider spread, but
  # PAD_SEP_AT_GRASP and CONTACT_DIST were both tuned against a 50 mm cube and
  # start mis-measuring the grasp before the claw runs out of travel.
  #
  # Mass and size are drawn INDEPENDENTLY, which is the point. Coupling them
  # through a density would teach the policy that a bigger cube is a heavier
  # one; the bench can hand it a 55 mm foam block and a 45 mm steel one, and the
  # only safe thing for it to learn is that the two are unrelated.
  #
  # Two couplings this does not correct, both small and both real: a bigger cube
  # rests with its centre higher (at +-10%, +-2.5 mm), and `cube_lifted` measures
  # against an absolute bar, so a big cube starts marginally closer to it. And
  # the spawn height is already raised to clear the highest bench draw, which
  # covers the largest cube too.
  cfg.events["dr_cube_scale"] = EventTermCfg(
    func=manipulation_mdp.object_scale,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("cube", body_names=("cube",), geom_names=("cube",)),
      "scale_range": scale_range,
    },
  )

  # The cube's geom carries `priority = 1`, so MuJoCo takes ITS friction for
  # every pair it is in and discards the other geom's. That makes this the only
  # live friction knob for the grasp: the three `fingertip_friction_*` events
  # inherited from the base cfg are silently overridden on every finger-cube
  # contact and only ever affect finger-vs-bench. The 0.5-1.2 band is upstream's
  # and it was measured under the same priority setup, so it already means
  # "effective pad-object friction" rather than "the cube's half of it".
  cfg.events["dr_cube_friction"] = EventTermCfg(
    func=dr.geom_friction,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("cube", geom_names=("cube",)),
      "operation": "abs",
      "distribution": "uniform",
      "axes": [0],
      "ranges": (0.5, 1.2),
    },
  )

  # --- the hand -------------------------------------------------------------
  # The widest band in the file. These five links are printed parts whose mass
  # and inertia come from CAD, and the CAD is stale: the servo horns were cut
  # off the fingers afterwards. +-40% is not caution, it is the actual size of
  # the disagreement.
  cfg.events["dr_hand_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODIES),
      "alpha_range": _alpha(*_HAND_MASS_SCALE),
    },
  )

  # --- servo gains ----------------------------------------------------------
  # Two bands, not one, which is what the split actuator groups in arm_cfg.py
  # exist for. The arm's servos are Menagerie's published position gains; the
  # fingers' are estimates for a DC15-A01-class servo, and kv alone separated a
  # 61%-hold rollout from a 100%-hold one. Randomizing the well-known and the
  # guessed-at over the same +-20% would understate the second by roughly the
  # margin that matters.
  cfg.events["dr_arm_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[ARM_ACTUATOR_GROUP]),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
    },
  )
  cfg.events["dr_finger_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[FINGER_ACTUATOR_GROUP]),
      "kp_range": (0.6, 1.4),
      "kd_range": (0.6, 1.4),
      "operation": "scale",
    },
  )

  # --- arm joints -----------------------------------------------------------
  # +-20%, and here that IS caution rather than ignorance: damping, frictionloss
  # and armature on joints 1-7 are the identified values from
  # robot-actuator-model, so the band only has to cover fit residual and wear.
  for name, func in (
    ("dr_arm_friction", dr.joint_friction),
    ("dr_arm_damping", dr.joint_damping),
  ):
    cfg.events[name] = EventTermCfg(
      func=func,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(ARM_JOINT_PATTERN,)),
        "operation": "scale",
        "distribution": "uniform",
        "ranges": (0.8, 1.2),
      },
    )

  # Encoder zeros. Not modelled at all before this: sim reads joint angles that
  # are exact by construction, while the real arm reports each joint through a
  # zeroing that was done once, by hand. +-0.005 rad is ~0.29 deg, which at the
  # 0.6 m working radius is about 3 mm of tool error per joint.
  cfg.events["dr_encoder_bias"] = EventTermCfg(
    func=dr.encoder_bias,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(ARM_JOINT_PATTERN,)),
      "bias_range": (-0.005, 0.005),
    },
  )
  # ...and make the policy actually live with it. mjlab already subtracts the
  # bias from the position TARGET, so without this the joint lands offset while
  # the observation still reports the truth -- an encoder error the policy can
  # see and correct, which is not what an encoder error is. Reading
  # `joint_pos_biased` puts observation and command in the same (wrong) frame,
  # which is the real robot's situation: everything it reports and everything
  # it is told agree with each other and disagree with the world.
  #
  # The critic shares these term objects with the actor by construction, so it
  # is biased too. That is acceptable -- its privilege is the cube pose, and a
  # critic told the true joint angles while the actor is not would be estimating
  # the value of a state the actor cannot identify.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params["biased"] = True

  # --- the bench ------------------------------------------------------------
  # The foam is 50 mm nominal and compresses; its uncompressed height is the one
  # number in scene.py that was taken from the spec sheet rather than measured,
  # and the arm is bolted to the TABLE while the cube rests on the FOAM, so the
  # error lands directly on the descent the policy has to judge. Moving the
  # table body moves the surface without moving the arm, which is the right
  # coupling.
  #
  # The cube's spawn height is raised to clear the highest bench draw (see
  # `physics_dr` in env_cfgs), so it always starts in free space and settles.
  # The success bar is absolute (`cube_lifted` compares against a fixed
  # `table_z + height`), so a higher bench genuinely makes the task harder here
  # rather than shifting the goalposts with it.
  cfg.events["dr_bench_height"] = EventTermCfg(
    func=dr.body_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("table", body_names=("table",)),
      "operation": "add",
      "distribution": "uniform",
      "axes": [2],
      "ranges": (-0.004, 0.004),
    },
  )

  # --- start pose -----------------------------------------------------------
  # Was (0.0, 0.0): every episode began from the identical joint vector, so the
  # policy never had to be right about anything except the cube. +-0.03 rad.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.03, 0.03)

  # --- command latency ------------------------------------------------------
  # THIS HALF IS PHYSICS, so it lands here rather than with the sensor latency
  # in `add_percept_dr`, and turning it on means RETRAINING THE TEACHER: a
  # command that arrives late changes the dynamics the teacher's labels are
  # correct for, which observation latency does not.
  #
  # Lags are in PHYSICS timesteps (5 ms at the 200 Hz sim), not control steps.
  # Both bands are ESTIMATES from the hardware, not measurements -- nothing in
  # `deploy/` times the round trip yet, and that measurement should replace
  # these before either is used to explain a sim-to-real gap.
  #
  #   arm     1-3 steps (5-15 ms).  Flexiv RDK writes joint targets over its own
  #           real-time link and the drive closes its loop far above 200 Hz, so
  #           the delay is transport, not control.
  #   fingers 4-10 steps (20-50 ms). A serial servo bus in front of a
  #           DC15-A01-class hobby servo, an order of magnitude slower than the
  #           arm, which is the same asymmetry `dr_finger_gains` exists for.
  #
  # `delay_hold_prob` keeps the lag from resampling every step. Bus latency is
  # correlated in time -- a link that is running late stays late for a while --
  # and a per-step redraw is zero-mean jitter the policy averages away, which is
  # the same trap `scale_err` is written to avoid on the depth side.
  if not CMD_LATENCY_DR:
    return
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  for group, (lo, hi) in (
    (ARM_ACTUATOR_GROUP, (1, 3)),
    (FINGER_ACTUATOR_GROUP, (4, 10)),
  ):
    act = articulation.actuators[group]
    act.delay_min_lag = lo
    act.delay_max_lag = hi
    act.delay_hold_prob = 0.8


# Camera extrinsic jitter. Upstream uses +-0.5 deg on all three angles, on the
# argument that the D435i is bolted down once and not remounted between trials.
# That argument is sound for the two angles that were MEASURED -- fitting the
# table plane in the point cloud pins pitch and the height along the normal --
# and it is not sound for yaw, which is still the CAD value with nothing
# checking it. Three of the six extrinsic DoF are measured; the other three are
# assumptions, and the as-built optical axis already came out 1.044 deg off the
# CAD chain, i.e. twice upstream's whole band.
#
# So: keep 0.5 deg where there is a measurement, and give yaw 2.5 deg.
#
# MEASURED 2026-08-06, and it inverts the paragraph above. `wide_dr_ablate.py`
# re-scored ONE student checkpoint under narrower envs (128 envs, +-2.5 pt
# sampling noise from a repeated config):
#
#   full DR                          62.5%
#   yaw 2.5 -> 0.5 deg               59.4% / 64.1%   <- two draws, ZERO effect
#   depth-sensor DR off              64.1%           <- zero effect
#   camera ROTATION off (pos on)     68.8%           <- +6
#   camera TRANSLATION off (rot on)  88.3%           <- +26
#   camera pose DR off entirely      92.2%
#
# So of the 30 points perception DR costs, 26 are the +-3 mm TRANSLATION and 6
# are all three rotations combined. "Rotation is the axis, translation is nearly
# free" is upstream's finding on upstream's setup; here it is backwards. The
# yaw widening this file argues for so hard buys and costs nothing.
#
# Splitting the translation per axis pins the mechanism. scene_cam's view
# direction in its parent body frame is [-0.002, -0.879, -0.477], so Y is the
# optical axis and X is very nearly perpendicular to it:
#
#   axis   along view   depth bias at 3 mm   lateral   success
#   X        0.002           0.006 mm         1.000     90.6%
#   Z        0.477           1.4  mm          0.879     88.3%
#   Y        0.879           2.6  mm          0.477     67.2%   <- all of it
#
# That rules out apparent image motion: Z has MORE lateral component than Y and
# costs nothing. Cost tracks the DEPTH BIAS alone, because depth here is metric
# -- sliding the camera along its optical axis offsets every reading, and a
# student inferring the cube's world position from those readings cannot know
# the camera moved, so it eats a constant bias for the whole episode. Rotation
# moves where things appear; translation moves what they measure.
#
# It is also a THRESHOLD, not a proportionality: 1.4 mm of depth bias is free
# and 2.6 mm costs 26 points. The real-robot spec that follows is specific --
# calibrate the camera's position ALONG ITS OPTICAL AXIS to better than ~1.5 mm
# and this student recovers to ~88-90% with no retraining. The other two axes
# are nearly free at this magnitude.
#
# Do NOT shrink _CAM_POS_JITTER to buy the 26 points back. It is a claim about
# how well the real D435 extrinsic translation can be calibrated; shrinking it
# without that measurement just hides the error until the real robot finds it.
_CAM_ANGLE_MEASURED = math.radians(0.5)
_CAM_ANGLE_YAW = math.radians(float(os.environ.get("CG_WIDE_CAM_YAW_DEG") or "2.5"))
_CAM_POS_JITTER = 0.003  # m, bolt-hole slop -- the expensive number, see above

# ABLATION SWITCHES, for attributing a score drop to a specific knob rather than
# to "perception DR" as a lump. Set to 0 to disable that half at BUILD time,
# which means an existing checkpoint can be re-scored under a narrower env
# without retraining -- the student's weights do not care which events the env
# registers. Defaults keep everything on; these exist to be flipped by
# `scripts/wide_dr_ablate.py`, not to be left off.
#
# Needed because three of the perception ranges have no measurement behind them:
# the 2.5 deg camera yaw (upstream uses 0.5 and this file widens it on the
# argument that yaw was never calibrated), and both dropout rates (upstream
# leaves them at 0 and never measured them). A 39-point drop attributed to "the
# perception DR" is not actionable; attributed to one of those three, it is.
#
# The pose half splits again into translation and rotation, because the first
# ablation showed the lump is not where the argument above predicted: narrowing
# yaw from 2.5 to 0.5 deg recovered NOTHING (59.4% vs a 62.5% baseline), while
# dropping camera-pose DR outright recovered 30 points. Whatever costs those 30
# points is in the three DoF this file calls measured, so the pos/rot split is
# the next thing that has to be separated before any range is touched.
CAM_POSE_DR = (os.environ.get("CG_WIDE_CAM_DR") or "1") != "0"
CAM_POS_DR = (os.environ.get("CG_WIDE_CAM_POS_DR") or "1") != "0"
# Which of the three cam_pos axes to jitter, as a comma-separated list. These are
# PARENT-BODY axes, not the optical frame, so map them through the camera's
# rotation before calling one of them "the view axis". Exists to test whether the
# 26-point translation cost is a metric-depth bias (one axis) or apparent motion
# (spread across axes) -- the two imply different fixes.
CAM_POS_AXES = [
  int(a) for a in (os.environ.get("CG_WIDE_CAM_POS_AXES") or "0,1,2").split(",")
]
CAM_ROT_DR = (os.environ.get("CG_WIDE_CAM_ROT_DR") or "1") != "0"
DEPTH_SENSOR_DR = (os.environ.get("CG_WIDE_DEPTH_DR") or "1") != "0"
SENSOR_LATENCY_DR = (os.environ.get("CG_WIDE_LATENCY_DR") or "1") != "0"

# Sensor latency, in CONTROL steps. The sim runs 200 Hz physics with decimation
# 4, so one control step is 20 ms and `deploy/run.py` closes its loop at the
# matching 50 Hz.
#
# CAMERA 1-3 steps (20-60 ms). Two contributions, neither of them measured here:
# the D435's own depth pipeline (exposure, the D4 ASIC's stereo, USB3, and
# librealsense) is tens of milliseconds, and the stream runs at 90 fps against a
# 50 Hz loop, so the newest frame is already up to 11 ms old when it is read.
# `deploy/` does not time this yet; the band should be replaced by a measurement
# before it is used to explain anything.
#
# PROPRIOCEPTION 0-1 steps. The Flexiv RDK streams joint state at 1 kHz and the
# loop reads the latest sample, so the error is sub-control-step -- it is here to
# stop the policy treating joint angles as simultaneous with the image, not
# because the arm is slow.
#
# `delay_hold_prob` because pipeline latency is CORRELATED IN TIME. A per-step
# redraw is zero-mean jitter that a policy averages out over a few frames, i.e.
# training against nothing -- the same failure the depth DR's per-episode holds
# exist to avoid. Holding the lag models a pipeline that runs at one latency and
# occasionally shifts to another, which is what a real one does.
_CAM_LAG = (1, 3)
_PROPRIO_LAG = (0, 1)
_LAG_HOLD_PROB = 0.9


def add_sensor_latency(cfg: ManagerBasedRlEnvCfg) -> None:
  """Make the student's observations arrive late. Mutates ``cfg``.

  Perception, not physics: the command still reaches the motor when it did, so
  the teacher's labels stay correct and this needs no teacher retrain. The
  COMMAND half lives in `add_physics_dr` and does need one.

  Only the SENSED terms are delayed. `actions` is the policy's own previous
  output and `goal_height` is a commanded number -- both are known exactly and
  on time on the real robot, and delaying them would model a latency that does
  not exist.

  And only the DEPLOYED policy's group. On the distillation task "actor" is the
  TEACHER, reading privileged state through an already-trained checkpoint;
  lagging its inputs would degrade the labels the student is being fitted to
  rather than toughening the student.
  """
  if not SENSOR_LATENCY_DR:
    return
  policy = "student" if "student" in cfg.observations else "actor"
  lagged = [("camera", "d435_depth", _CAM_LAG)] if "camera" in cfg.observations else []
  lagged += [
    (policy, term, _PROPRIO_LAG)
    for term in ("joint_pos", "joint_vel")
    if term in cfg.observations[policy].terms
  ]
  for group, term, (lo, hi) in lagged:
    # DEEPCOPY, because the groups SHARE TERM OBJECTS. `student.joint_pos` is
    # the same object as `actor.joint_pos` and `critic.joint_pos` (the same
    # aliasing the `biased` note in `add_physics_dr` relies on), so assigning to
    # it in place lags the teacher and the critic as well -- silently, and in
    # the one direction that makes the labels worse instead of the student
    # tougher.
    t = deepcopy(cfg.observations[group].terms[term])
    t.delay_min_lag = lo
    t.delay_max_lag = hi
    t.delay_hold_prob = _LAG_HOLD_PROB
    cfg.observations[group].terms[term] = t


def add_percept_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Randomize what the camera sees and how well. Mutates ``cfg``."""
  if not CAM_POSE_DR:
    return
  cam = SceneEntityCfg("robot", camera_names=("scene_cam",))

  if CAM_POS_DR:
    cfg.events["dr_cam_pos"] = EventTermCfg(
      func=dr.cam_pos,
      mode="startup",
      params={
        "asset_cfg": cam,
        "operation": "add",
        "distribution": "uniform",
        "ranges": (-_CAM_POS_JITTER, _CAM_POS_JITTER),
        "axes": CAM_POS_AXES,
      },
    )
  if CAM_ROT_DR:
    cfg.events["dr_cam_quat"] = EventTermCfg(
      func=dr.cam_quat,
      mode="startup",
      params={
        "asset_cfg": cam,
        "roll_range": (-_CAM_ANGLE_MEASURED, _CAM_ANGLE_MEASURED),
        "pitch_range": (-_CAM_ANGLE_MEASURED, _CAM_ANGLE_MEASURED),
        "yaw_range": (-_CAM_ANGLE_YAW, _CAM_ANGLE_YAW),
      },
    )
  # Intrinsics: the residual of the factory calibration, not an unknown.
  cfg.events["dr_cam_fovy"] = EventTermCfg(
    func=dr.cam_fovy,
    mode="startup",
    params={
      "asset_cfg": cam,
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (0.995, 1.005),
    },
  )
