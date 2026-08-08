"""Tabletop grasp tasks for the Flexiv Rizon4S + the WIDE two-finger claw.

Same reward stack as ``flexiv_two_finger``, on a different robot and a different
bench. Nothing about the shaping was re-derived: every reward term, curriculum,
gate and penalty below is the one that produced v11_align (80.5% deployment
success) on the 56 mm claw. What changed is everything it MEASURES AGAINST --

  robot   the fitted 70 mm-knuckle claw, read from CAD at build time, on the
          sys-id'd arm with gravity compensation and the calibrated D435i mount
          (mjlab.asset_zoo.robots.twofinger_wide). Every name gains a ``_tf``
          suffix and the palm body is ``hand_base_tf``, not ``base_link``.
  bench   the MEASURED 63 x 31.75 in table with 50 mm of foam and the wall it
          now stands flush against (``scene.py``). The cube rests on the FOAM at
          z = 0.425, not on the table top at 0.400, and the arm is bolted to the
          table at 0.365 -- three heights where there used to be one.

Constants that were carried over unexamined the last time a claw changed and
were all wrong are re-measured here by ``scripts/wide_calibrate.py``; each one
below cites what that script reported.

Two tasks share one scene (arm + claw on the bench, a cube, and the fixed
D435i in front of the arm base):

  * state-based  -- obs = joint state + cube pose + gripper; grasp with privileged
    object pose.
  * vision-based -- obs = D435 RGB or depth + proprioception; no cube pose.
  * distillation -- both at once: the state observation for a frozen teacher and
    the vision observation for the student being fitted to it (DAgger).

Both drive the whole arm plus the four finger joints in joint space with a
single position action (offsets from a hover start pose), matching the stock
lift-cube recipe where grasping emerges from exploration. Success = lift the
cube >= 10 cm off the table and hold it there for 1 s.
"""

from copy import deepcopy
from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  ARM_BASE_Z,
  TWOFINGER_ARM_HOME,
  get_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_root_state_uniform
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.sensor.camera_sensor import CameraDataType
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .dr_cfg import (
  DEPTH_SENSOR_DR,
  add_percept_dr,
  add_physics_dr,
  add_sensor_latency,
)
from .scene import (
  CUBE_HALF,
  RESTING_Z,
  SPAWN_X_JITTER,
  SPAWN_Y_JITTER,
  TABLE_X,
  WORK_SURFACE_Z,
  get_cube_spec,
  get_table_spec,
)

# Three heights where the old scene had one, because the bench carries 50 mm of
# foam cut around the arm's hardware. Getting these confused does not raise --
# it silently puts the cube inside the foam or the arm inside the table.
#
#   ARM_BASE_Z       0.365   the arm's own mount, on top of its 15 mm base plate
#   WORK_SURFACE_Z   0.400   foam top; what the cube rests on and lifts from
#   RESTING_Z        0.425   cube CENTRE at rest
#
# `TABLE_H` is deliberately not defined in this module. Every use in the old
# file meant one of the three and they are no longer the same number.

# Cube spawn region: +-80 mm in x, +-100 mm in y about the manipulation centre.
# Derived from the bench rather than written out -- the jitter is clamped so a
# 50 mm cube can never be spawned overhanging an edge, which on this table binds
# in y (403 mm half-width) and not in x (the slab runs to 1.343 m).
#
# The box is the one cubegrasp-env swept against the D435i's cone and its 200 mm
# minimum range (scripts/calib/aim_camera.py): 0.00% of the 22869-pose envelope
# is blind or out of frame, with the nearest cube corner at 205.3 mm and the
# worst frame coordinate at 0.646 of half-frame. That 5 mm of range margin is the
# tightest constraint in the scene and it is what pins TABLE_X at 0.46.
CUBE_REGION = dict(
  x=(TABLE_X - SPAWN_X_JITTER, TABLE_X + SPAWN_X_JITTER),
  y=(-SPAWN_Y_JITTER, +SPAWN_Y_JITTER),
)
LIFT_HEIGHT = 0.10  # m above the WORK surface the cube must reach
HOLD_S = 1.0

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
FINGER_JOINTS = ("left_1_tf", "left_2_tf", "right_1_tf", "right_2_tf")
# The distal link's colliders are a 4-part COACD decomposition, so a single
# `_col` suffix no longer names them -- the old `(left_2|right_2)_col` pattern
# matches nothing on this claw, and a contact sensor that matches nothing reports
# no contact rather than an error.
FINGERTIP_GEOMS = r"(left_2|right_2)_col\d_tf"
LEFT_FINGERTIP_GEOMS = r"left_2_col\d_tf"
RIGHT_FINGERTIP_GEOMS = r"right_2_col\d_tf"
# The FINGERTIP reference sites, placed where the old claw's were -- at the tip,
# not at the centroid of the gripping face 22 mm up the finger. Carried from the
# tuned point and re-checked against this claw's mesh: 0.404 mm off its surface,
# against 0.387 mm for the old claw's own tuned point.
PAD_SITES = ("left_pad_tf", "right_pad_tf")
GRASP_SITE = "grasp_site_tf"
PALM_BODY = "hand_base_tf"

# Separation of the two PAD_SITES while the claw is actually GRIPPING the 50 mm
# cube -- driven onto it and stopped BY it, not commanded into free space. Those
# are different numbers and the free-space one is the wrong one.
#
# MEASURED, scripts/wide_calibrate.py section 2: commanded to the full-close
# angle the fingers settle at left_1_tf = -0.0023 with the pad markers 49.9 mm
# apart (flush on the cube's 50 mm faces) and the tip SITES 67.41 mm apart, i.e.
# 8.71 mm outboard of each face. The old claw's equivalent was 64.4 mm / 7.2 mm.
#
# Targeting the cube's 50 mm instead would put `antipodal_pinch`'s maximum at a
# separation the claw can only reach by closing PAST the cube, which is the
# over-closing loophole the term exists to remove.
PAD_SEP_AT_GRASP = 0.0674

# What the hand actually lands on when it comes down. Still one name, and that
# is load-bearing rather than incidental: mjlab's ContactSensor requires its
# SECONDARY match to resolve to exactly ONE element, and under the default
# `secondary_policy="first"` a pattern matching several silently keeps one of
# them. The bench is therefore built as a single `table` body carrying the slab,
# the four foam strips and the wall as geoms (see scene.py) -- so this covers the
# whole work surface, which the fingertips now reach 50 mm above the table top.
#
# Wired to "table" while the surface was four separate `foam_*` bodies, both
# crash guards reported no contact for the whole of run 55060 while the
# fingertips were pressed into foam. Neither raised.
WORK_SURFACE_BODIES = "table"

# (proximal, distal) limits for the LEFT finger; mirrored onto the right.
FingerRange = tuple[tuple[float, float], tuple[float, float]]


def _robot_cfg(finger_range: FingerRange | None) -> EntityCfg:
  """The arm + claw, optionally with the finger joints limited to real travel.

  The URDF ships +-1.6 rad on all four finger joints, which is far more than the
  mechanism wants on either side. On THIS claw pad separation is affine in the
  proximal angle over the whole useful band -- sep(q) = 50.1 + 134.0 q mm,
  measured off the compiled model with a fit residual under 0.25 mm -- so unlike
  the old claw there is no scissoring region where the map goes two-valued. What
  the limits are for here is drive-through: at q = -0.0754 the pads are 40 mm
  apart, a 10 mm squeeze on the cube, and anything below that is the servo
  saturating against interpenetration.

  Ranges are given for the LEFT finger and mirrored onto the right, because
  closing is left-negative / right-positive: a single symmetric range cannot
  express the limit.

  Setting this ALSO scopes the stock `joint_pos_limits` penalty to the arm (see
  `_apply_common`), which is the point -- these limits are a design constraint
  and pressing against them is intended behaviour, not a fault.
  """
  cfg = get_robot_cfg()
  if finger_range is None:
    return cfg
  prox, dist = finger_range
  inner = cfg.spec_fn
  assert inner is not None

  def spec_fn() -> mujoco.MjSpec:
    spec = inner()
    for j in spec.joints:
      lohi = prox if j.name in ("left_1_tf", "right_1_tf") else dist
      if j.name in ("left_1_tf", "left_2_tf"):
        j.range = [lohi[0], lohi[1]]
      elif j.name in ("right_1_tf", "right_2_tf"):
        j.range = [-lohi[1], -lohi[0]]
    return spec

  cfg.spec_fn = spec_fn
  return cfg


def _apply_common(
  cfg: ManagerBasedRlEnvCfg,
  joint_vel_penalty_final: float,
  goal_mode: Literal["dynamic", "above_object"],
  bringing_std: float,
  grasp_shaping: Literal["proximity", "antipodal"] = "proximity",
  symmetry_penalty: float = 0.0,
  finger_range: FingerRange | None = None,
  reach_from: Literal["palm", "pads"] = "palm",
  align_weight: float = 0.0,
  pinch_z_std: float = 0.06,
  touch_weight: float = 0.0,
  success_reward: float = 0.0,
  align_gate: float = 0.0,
  align_std: float = 0.436,
  align_on: tuple[str, ...] = ("grasp",),
  align_curriculum: bool = False,
  free_wrist_penalties: bool = False,
  wrist_reset_range: float = 0.0,
  solve_wrist: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Shared setup: robot + table + cube, 6-DOF IK + gripper action, success."""
  cfg.scene.entities = {
    "robot": _robot_cfg(finger_range),
    # One entity carrying the table slab, the four foam strips and the wall --
    # all static, all welded to the world, so MuJoCo filters every pair among
    # them and none of it enters the dynamics.
    "table": EntityCfg(spec_fn=get_table_spec),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # Contact headroom for the bigger bench. The stock 55 contacts was sized for a
  # 48 x 32 in slab and a 5-geom hand; this scene adds the foam strips, the wall,
  # the base plate, the camera mount and a 4-part COACD decomposition on each
  # distal link, so the arm links and the plate are candidates against the table
  # on top of the hand/cube contacts.
  #
  # njmax stays at the stock 600 rather than the 1100 upstream uses, and the
  # reason is now just "600 is plenty", not the NVRTC failure this originally
  # worked around. 1100 broke `update_gradient_JTDAJ_dense_tiled` until the real
  # cause -- NVRTC precompiled headers, see mjlab/__init__.py -- was found; both
  # values compile now (scripts/wide_diag_workarounds.py).
  #
  # 600 is kept because it is measured, not guessed: the constraint count is 11
  # joint limits plus condim (4) rows per contact, and scripts/wide_check.py
  # reports a peak of 59 rows and 12 contacts per world for a full grasp-and-lift
  # -- an order of magnitude of headroom either way. Both are PER WORLD; `nacon`
  # read back from the sim is the global total across worlds, which is how a
  # comfortable 4-of-150 first read as a 70% overflow.
  cfg.sim.nconmax = 150
  cfg.sim.njmax = 600

  # The stock `joint_pos_limits` penalty covers ".*" at weight -10. That is right
  # for the ARM, whose limits are the hardware's and should never be leaned on,
  # and wrong for the fingers once `finger_range` is set: those limits are a
  # deliberate design constraint, and pressing against them is the intended
  # behaviour rather than a fault.
  #
  # It is not a small effect. soft_joint_pos_limit_factor=0.9 puts the soft band
  # at the middle 90% of each range, so a one-way box whose boundary IS the hover
  # pose charges from the very first step:
  #
  #   proximal [+0.50,+1.60] -> soft [+0.555,+1.545]; hover AND grasp at +0.50
  #                             are 0.055 outside it, permanently
  #   distal   [-1.60,-0.40] -> soft [-1.540,-0.460]; hover at -0.40 is 0.060 out
  #
  # summed over two mirrored fingers at weight -10 that is -2.30 per step at the
  # start pose and -1.10 per step at the ideal grasp, against `lift` at +1.25 --
  # a penalty actively pushing the hand OFF the pose the task is trying to learn.
  if finger_range is not None:
    cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=ARM_JOINTS
    )

  # The hover reset pose. NOT hand-written and not IK-only: solved through the
  # physics by cubegrasp-env's pose solver (kinematic IK, then the static
  # equilibrium map inverted by fixed-point iteration) for THIS claw over the
  # 50 mm foam, and re-verified here.
  #
  # It is no longer droop-compensated. The arm now carries gravity compensation
  # the way the real controller does (measured sag 2.735 deg -> 0.000), so the
  # command and the pose it settles at have collapsed to the same numbers. Every
  # pre-gravcomp checkpoint is geometrically invalid against this.
  #
  # MEASURED at this pose, scripts/wide_calibrate.py section 3:
  #   pad-site separation      115.44 mm   (86.9 mm at the gripping faces)
  #   pad midpoint - cube      +49.5 mm in z, +7.5 mm in x
  #   fingertip clears foam by  57.13 mm
  # so the approach is a ~50 mm descent with the jaw already open wide enough to
  # clear the cube's 70.7 mm diagonal by 8 mm per side.
  #
  # The fingers start STRAIGHT (distal = 0), unlike the old claw's pre-curled
  # hover. This claw does not need the curl: its pad separation is affine in the
  # proximal angle alone and the scripted expert holds 22.4 N with the distals at
  # zero. Curling here would only spend fingertip clearance.
  cfg.scene.entities["robot"].init_state = EntityCfg.InitialStateCfg(
    # Fixed base -> this pos overwrites the spec's base.pos, so it must carry the
    # arm's own mount height (table top PLUS the 15 mm base plate) or the arm
    # ends up standing on the floor, passing through the table
    # (see Entity._add_initial_state_keyframe). This is ARM_BASE_Z and NOT the
    # work surface: the foam is cut away around the plate.
    pos=(0.0, 0.0, ARM_BASE_Z),
    joint_pos=dict(TWOFINGER_ARM_HOME.joint_pos or {}),
    joint_vel={".*": 0.0},
  )

  # Envs are laid out on a grid at this spacing; the stock 1 m would overlap the
  # 48 x 32 in tables of neighbouring envs (harmless physically -- worlds are
  # independent -- but it makes multi-env renders unreadable).
  cfg.scene.env_spacing = 2.5

  # The table is a fixed-base (mocap) entity, and a fixed entity only lands on
  # its env_origin if some reset event writes its mocap pose (the stock cfg only
  # does this for the robot, via "reset_base"). Without this the table would stay
  # at the world origin while each env's arm + cube sit at their own origin --
  # i.e. the arm floating next to the table instead of standing on it.
  cfg.events["reset_table"] = EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  # --- Action: joint-space position control of the whole arm + gripper -------
  # The stock lift-cube recipe drives all actuators in joint space with a single
  # position action and lets grasping emerge from exploration (the staged reward
  # only pays out once the cube rises, which requires closing the fingers).
  # Cartesian IK decouples the arm from the gripper and its Gaussian action noise
  # does not readily explore the coordinated close+lift, so we use joint-space
  # control here (offsets from the hover default pose).
  #
  # All FOUR finger joints are driven independently -- left_1_tf/left_2_tf and
  # right_1_tf/right_2_tf are separate actuators with no mimic or tendon
  # coupling, so the policy commands each knuckle. Action dim = 7 + 4 = 11.
  #
  # ACTION REACH. With use_default_offset the command is
  # `default + scale * action`, so the action needed to reach a pose is
  # (target - default) / scale -- and anything past ~1.5 sigma is a pose
  # Gaussian exploration will never stumble into, which means no gradient ever
  # forms toward it. This is what killed four earlier runs on the old claw, and
  # its symptom is a reward term reading exactly 0.0000 for a whole run rather
  # than any kind of error.
  #
  # With `solve_wrist`, joint7 is not learned at all: it is driven to the angle
  # that squares the jaw, re-solved from measured state every substep, and the
  # policy keeps only a small trim on top. See
  # WristAlignedJointPositionActionCfg for why -- six RL variants failed to learn
  # a one-line geometry problem. The trim scale is 0.15 rad (8.6 deg), inside the
  # 0-22.5 deg band where alignment provably does not affect success, so the
  # policy cannot undo the solution but still has a real gradient on the joint.
  #
  # THE SCALES ARE THIS ROBOT'S, NOT THE OLD CLAW'S. All three were re-measured
  # for the plate-mounted arm on this bench (cubegrasp-env
  # scripts/check/check_grasp_reachable.py, end-to-end in full physics rather
  # than by kinematics, which understates it because sag grows with reach):
  #
  #  * joints 1-6 (0.40). On the old FLOOR mount 0.15-0.3 was enough only
  #    because gravity sag happened to park the tool 34 mm from the cube.
  #    Plate-mounted the arm reaches out level and the cube jitters over
  #    +-59/+-100 mm, where the minimum-norm action to put the pad midpoint on
  #    the cube measures max|a| = 0.81..1.64 -- above 1 is outside the envelope
  #    entirely, with nothing left for descent, closure and lift.
  #  * joint7 (1.1), the wrist roll that squares the jaw to a cube face. Cube
  #    yaw is uniform on [-pi, pi] and the jaw is 180-deg symmetric, so
  #    alignment needs a full pi of travel.
  #  * fingers (0.35), set by ONE requirement: full close must land a ~10 mm
  #    squeeze on the cube, not drive 52 mm through it. With the open default at
  #    +0.2746 the most-closed command is q = -0.0754 = 40 mm of pad separation.
  #    At 0.6 the policy could command a NEGATIVE separation, which saturates the
  #    2 N.m servo the instant it touches -- forced-closure rollouts spiked the
  #    touch sensor to 459 N and threw the cube 607 mm off the table.
  #
  # These are coupled to `init_std` and cannot be tuned alone: the noise actually
  # injected per joint is scale * std against a validated 0.15 rad budget, so
  # 0.40 * 0.4 = 0.16 rad. Raising the scale without dropping std multiplies
  # exploration noise, and upstream lost a run that way (the arm thrashed into
  # the table hard enough that the penalties buried the sparse grasp reward and
  # the policy learned to RETREAT). See rl_cfg.py.
  #
  # ACTION REACH under these scales, scripts/wide_calibrate.py section 4 -- every
  # pose the task has to reach is inside 1 sigma:
  #   full close (40 mm)          -1.00     distal curl -0.3   -0.86
  #   grip as the cube stops it   -0.79     hover -> pre-grasp, worst joint +0.47
  wrist_scale = 0.15 if solve_wrist else 1.1
  action_scale = {
    r"joint[1-6]": 0.40,
    r"joint7": wrist_scale,
    r"(left|right)_[12]_tf": 0.35,
  }
  cfg.actions = {
    "joint_pos": manipulation_mdp.WristAlignedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(*ARM_JOINTS, *FINGER_JOINTS),
      scale=action_scale,
      use_default_offset=True,
      object_name="cube",
      pad_sites=PAD_SITES,
    )
    if solve_wrist
    else JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(*ARM_JOINTS, *FINGER_JOINTS),
      scale=action_scale,
      use_default_offset=True,
    ),
  }

  # Extra solver headroom: the self-colliding finger hulls need more Newton /
  # line-search iterations to resolve cleanly and avoid contact blow-ups.
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.ls_iterations = 30

  # --- Observations / rewards reference the hand's grasp_site ---
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    GRASP_SITE,
  )

  # --- The cube's ORIENTATION, which nothing else in the observation carries ---
  # `ee_to_cube` is a position vector; until now the actor could see where the
  # cube was but not which way it was turned. For an antipodal gripper on a
  # yaw-randomized box that is fatal: the grasp requires squaring the jaw to a
  # face, and the angle to square to was simply unobservable. Every state-based
  # run so far has been blind to it.
  cube_quat = ObservationTermCfg(
    func=manipulation_mdp.object_orientation,
    params={"object_name": "cube"},
    noise=Unoise(n_min=-0.01, n_max=0.01),
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["cube_quat"] = cube_quat
  cfg.rewards["lift"].params["asset_cfg"].site_names = (GRASP_SITE,)

  # --- Get the PADS onto the cube, not just the palm site -------------------
  # The stock `lift` reaching term measures grasp_site, a fixed point on the
  # palm, and the policy found the loophole: park that point on the cube while
  # splaying the fingers as far away as possible (splaying is free, and it also
  # keeps the fingertips clear of the table-contact penalty). Run 45123 at 3000
  # iterations: site 8.5 mm from the cube centre, pads 81.6 mm away, jaws
  # trained open to left_1 +0.98, cube never touched, grasp reward 0.0000.
  #
  # This is added ALONGSIDE the wide reaching kernel rather than replacing it.
  # Swapping the kernel over to the pads and narrowing it collapsed the whole
  # `lift` reward to ~0.005 across the spawn region (measured, smoke 45443) --
  # it fixed the endgame and destroyed the approach gradient that already works.
  cfg.rewards["pad_pinch"] = RewardTermCfg(
    func=manipulation_mdp.fingertip_proximity,
    weight=1.0,
    params={
      "object_name": "cube",
      # Tight across the closing direction, where the contact geometry is sharp
      # (0.3 mm spread in the measured contact points); loose along the finger,
      # where it is not (3.9-12.0 mm, and the two fingers disagree by 26 mm). A
      # 26 mm site error still scores 0.645 against 0.78 for a perfect grasp, so
      # the site's placement along the finger does not have to be got right --
      # while the 90 mm hover the horizontal-only version settled into scores
      # 0.099.
      "std": 0.05,
      "z_std": pinch_z_std,
      "asset_cfg": SceneEntityCfg("robot", site_names=PAD_SITES),
    },
  )

  # The reaching half of `lift` measures from grasp_site, a point on the PALM --
  # the hand can hold that on the cube while splaying the fingers away, which is
  # exactly the loophole fingertip_proximity was bolted on to patch. Now that the
  # pad sites are calibrated the main reward can measure from between the
  # fingertips instead. reaching_std stays at 0.20: an earlier attempt swapped
  # the kernel to the pads AND narrowed it, and the whole lift reward collapsed
  # to ~0.005 across the spawn region (smoke 45443). The narrowing was the cause.
  if reach_from == "pads":
    cfg.rewards["lift"].params["asset_cfg"].site_names = PAD_SITES

  if touch_weight:
    # Score the fingertips landing ON the cube, rather than any of the proxies
    # for it. Every proxy tried so far (separation, midpoint, jaw yaw, pad tilt)
    # turned out to be satisfiable with the tips in mid-air; see the note on
    # pad_site_touch.
    #
    # std = 30 mm, and the first attempt at 15 mm is why. Being non-zero is not
    # enough: with the jaw at the hover width the tips are 27.4 mm off the cube
    # face, and at std=15 that scores 0.0013 per side, 0.0005 in training. Beside
    # `lift` at 1.27 that is 0.04% of the return -- invisible to PPO, and run
    # 46789 sat at 0.0005 for 558 iterations without moving. The relative
    # gradient was steep (5 mm closer was worth 9x) but the ABSOLUTE swing was
    # nothing, and it is the absolute one that enters the advantage.
    #
    #   std     hover    pinch    over-closed
    #   15 mm   0.0013   0.9996   0.554
    #   30 mm   0.1885   0.9999   0.863
    #
    # 30 mm makes closing worth ~0.81 of return against lift's 1.27, while still
    # paying 5x more for touching than for hovering.
    cfg.rewards["pad_touch"] = RewardTermCfg(
      func=manipulation_mdp.pad_site_touch,
      weight=touch_weight,
      params={
        "object_name": "cube",
        "half_extent": CUBE_HALF,
        "std": 0.030,
        "asset_cfg": SceneEntityCfg("robot", site_names=PAD_SITES),
      },
    )

    # `pad_pinch` has to GO, not sit alongside this. The two terms want opposite
    # things and they are not a little bit different, they are opposed:
    #
    #   pad_pinch  maximal when both pad sites are at the cube's CENTRE, i.e. a
    #              jaw gap of ZERO -- the tips driven through the cube
    #   pad_touch  maximal when both pad sites are on the cube's SURFACE, i.e. a
    #              jaw gap of 50 mm -- one tip on each face
    #
    # At the correct grasp each tip is 25 mm off centre, so pad_pinch scores 0.78
    # against the 1.0 it would pay for collapsing them together: it bids 0.22 for
    # over-closing on every step. pad_touch drops from 1.00 to 0.86 over the same
    # move, so the pair very nearly cancels and the net signal is noise.
    #
    # This is the same loophole `antipodal_pinch` was written to close ("two pads
    # cannot both be at the centre"), so keeping the term it replaces defeats the
    # point. pad_touch already does pad_pinch's job -- both pull the tips onto the
    # cube -- without the collapse-to-centre optimum.
    cfg.rewards.pop("pad_pinch", None)

  if align_weight:
    cfg.rewards["jaw_alignment"] = RewardTermCfg(
      func=manipulation_mdp.jaw_face_alignment,
      weight=align_weight,
      params={
        "object_name": "cube",
        "std": 0.436,  # 25 deg; see the note in jaw_face_alignment
        "asset_cfg": SceneEntityCfg("robot", site_names=PAD_SITES),
      },
    )

  if grasp_shaping == "antipodal":
    # Same slot and same weight, so this is a clean swap of the shaping term
    # rather than an extra reward stacked on top -- otherwise the proximity
    # term's pull toward a zero jaw gap would still be there, competing.
    cfg.rewards["pad_pinch"] = RewardTermCfg(
      func=manipulation_mdp.antipodal_pinch,
      weight=1.0,
      params={
        "object_name": "cube",
        # NOT 2*CUBE_HALF. The pad sites sit 8.6 mm behind each gripping face,
        # so while the cube is held the pair is 64.4 mm apart, not the cube's
        # 50 mm. Targeting 50 would put the reward's maximum at a separation the
        # hand can only reach by closing PAST the cube, which is the very
        # over-closing loophole this term exists to remove.
        "width": PAD_SEP_AT_GRASP,
        # These two were 0.02 / 0.04 in run 45593 and the term read exactly
        # 0.0000 for all 3000 iterations -- the signature of a reward the policy
        # can never reach, not of a broken sensor. At the hover the separation is
        # 68.5 mm off target (1.75 sigma at 0.02) and the midpoint is 100 mm off
        # (2.5 sigma at 0.04), and because the two terms MULTIPLY the product was
        # ~9e-5 everywhere the policy actually starts. No gradient, no learning.
        #
        # Sized from the hover now, not from the grasp: the term has to be worth
        # something at the pose training begins in. At 0.04 / 0.08 the hover
        # scores 0.053 * 0.21 = 0.011 and the grasp ~1.0, so there is a climb.
        # Anything tighter must be checked the same way before it is used.
        "sep_std": 0.04,
        "mid_std": 0.08,
        "asset_cfg": SceneEntityCfg("robot", site_names=PAD_SITES),
      },
    )

  if symmetry_penalty:
    cfg.rewards["finger_symmetry"] = RewardTermCfg(
      func=manipulation_mdp.mirror_asymmetry_penalty,
      weight=symmetry_penalty,
      params={
        "joint_pairs": (("left_1_tf", "right_1_tf"), ("left_2_tf", "right_2_tf")),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )

  # --- Cube spawn: reachable region on the table ---
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=CUBE_REGION["x"],
    y=CUBE_REGION["y"],
    z=(RESTING_Z, RESTING_Z),  # cube CENTRE resting on the foam
    yaw=(-3.14, 3.14),
  )
  # Lift goal height above the table top.
  #
  # goal_mode="above_object" puts the goal directly over wherever the cube
  # spawned, so the reward's position error IS the lift height. The stock
  # "dynamic" mode samples the goal x/y independently of the cube, which
  # leaves ~0.11 m of horizontal error after a perfect vertical lift -- that
  # dilutes the lift gradient and, worse, puts `lift_precise` (a Gaussian with
  # std 0.05, worth a full 1.0) permanently out of reach. Meanwhile the
  # success criterion `cube_lifted` only ever measures height, so the reward
  # was pulling toward something the task does not score.
  lift_cmd.difficulty = goal_mode
  cfg.rewards["lift"].params["bringing_std"] = bringing_std
  lift_cmd.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=CUBE_REGION["x"],
    y=CUBE_REGION["y"],
    z=(WORK_SURFACE_Z + 0.10, WORK_SURFACE_Z + 0.20),
  )

  # --- Fingertip friction DR on the distal finger colliders ---
  for ev in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[ev].params["asset_cfg"].geom_names = FINGERTIP_GEOMS

  # --- Crash guard: the PALM (not the fingers) slamming into the bench ---
  # A grasp needs the fingertips to reach the work surface, so scope this to the
  # palm's own geom ("body" mode on hand_base_tf) -- the ~11 cm fingers keep the
  # palm well above the surface during a grasp, so it fires only on a genuine
  # arm-into-bench crash.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.mode = "body"
      sensor.primary.pattern = PALM_BODY
      sensor.secondary = ContactMatch(
        mode="body", pattern=WORK_SURFACE_BODIES, entity="table"
      )

  # --- Soft penalty on fingertips mashing the work surface ------------------
  # Light contact is unavoidable when grasping a low object, so rather than
  # terminate we measure fingertip-vs-surface contact force and penalize the hard
  # part of it, and the policy learns to grip the cube cleanly instead of driving
  # the tips into the bench.
  #
  # The surface is COMPLIANT here (CG_FOAM_SOFT), which is what lets the grasp be
  # centred at all: this claw's gripping face sits 23-33 mm above the fingertip
  # while the cube centre is only 25 mm above the surface, so gripping on the
  # cube's centre line needs the tips a few mm BELOW it. Against a rigid box that
  # is simply blocked (the grip lands 9.3 mm high); against foam it is what
  # happens. So this penalty is now charging for something the task genuinely
  # needs a little of, and the threshold is what keeps it a penalty on mashing
  # rather than on touching.
  fingertip_table_contact = ContactSensorCfg(
    name="fingertip_table_contact",
    primary=ContactMatch(mode="geom", pattern=FINGERTIP_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=WORK_SURFACE_BODIES, entity="table"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, fingertip_table_contact)
  cfg.rewards["fingertip_table_contact"] = RewardTermCfg(
    func=manipulation_mdp.contact_force_penalty,
    weight=-0.02,
    params={"sensor_name": "fingertip_table_contact", "force_threshold": 1.0},
  )

  # --- Grasp bootstrap: reward a genuine two-finger pinch on the cube ---------
  # The stock staged reward only pays for lifting, which this custom hand never
  # discovers by chance (the fingers reach the cube but never learn to close).
  # Per-side fingertip-vs-cube contact sensors let us reward BILATERAL contact
  # (left AND right tip on the cube = a pinch, not a one-sided shove), which the
  # policy *can* stumble into while reaching. Once it pinches, the cube becomes
  # controllable and the existing lift/bring rewards become climbable.
  left_cube_contact = ContactSensorCfg(
    name="left_cube_contact",
    primary=ContactMatch(mode="geom", pattern=LEFT_FINGERTIP_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=1,
  )
  right_cube_contact = ContactSensorCfg(
    name="right_cube_contact",
    primary=ContactMatch(mode="geom", pattern=RIGHT_FINGERTIP_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=1,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, left_cube_contact, right_cube_contact)
  cfg.rewards["grasp"] = RewardTermCfg(
    func=manipulation_mdp.bilateral_grasp,
    weight=1.0,
    params={
      "left_sensor": "left_cube_contact",
      "right_sensor": "right_cube_contact",
      "force_threshold": 0.1,
      "firm_scale": 5.0,
    },
  )

  # Alignment as a MULTIPLIER on the grasp reward, never as its own term. Adding
  # it failed at both ends -- weight 1.0 was hacked (jaw_alignment 0.946 while
  # grasp went to 0), weight 0.15 was ignored (0.074, chance) -- because an
  # additive term can be collected without grasping. Multiplied it cannot be
  # banked alone and cannot be ignored. `floor` keeps it a modulation rather than
  # an annihilator; see jaw_alignment_gate for why a bare product is the
  # 0.0000-forever trap.
  #
  # WHICH terms it gates matters more than the kernel width, because of a
  # chicken-and-egg measured on run 46984 (scripts/diag_joint7_use.py): joint7,
  # the wrist roll, ends up parked at +19.1 deg with a 4.3 deg spread across 64
  # envs whose cube yaws span the full circle, and corr(cube yaw, joint7) is
  # -0.089. It is not aligning anything. Its learned action std had already
  # collapsed to 0.0083 (+-0.5 deg) -- the lowest of any joint -- because
  # `grasp` reads 0.0000 for the first ~2000 iterations, so a gate multiplied
  # onto `grasp` supplies NO alignment gradient during exactly the window in
  # which action_rate_l2 and joint_vel_hinge are squeezing an unrewarded joint
  # to a standstill. By the time grasping works the wrist can no longer explore
  # its way to the 45 deg it needs.
  #
  # Gating a term that is non-zero from iteration 0 (`pad_touch` sits at ~0.2
  # early) gives alignment a gradient before the wrist freezes. Gating `lift`
  # additionally makes the carry care about posture, which nothing currently
  # does.
  if align_gate:
    align_params = {
      "object_name": "cube",
      "align_std": align_std,
      "align_floor": align_gate,
      "asset_cfg": SceneEntityCfg("robot", site_names=PAD_SITES),
    }
    if "grasp" in align_on:
      cfg.rewards["grasp"] = RewardTermCfg(
        func=manipulation_mdp.grasp_aligned,
        weight=1.0,
        params={
          **align_params,
          "left_sensor": "left_cube_contact",
          "right_sensor": "right_cube_contact",
          "force_threshold": 0.1,
          "firm_scale": 5.0,
        },
      )
    if "touch" in align_on:
      base = cfg.rewards["pad_touch"]
      cfg.rewards["pad_touch"] = RewardTermCfg(
        func=manipulation_mdp.gated_by_alignment,
        weight=base.weight,
        params={"inner": base, **align_params},
      )
    if "lift" in align_on:
      base = cfg.rewards["lift"]
      cfg.rewards["lift"] = RewardTermCfg(
        func=manipulation_mdp.gated_by_alignment,
        weight=base.weight,
        params={"inner": base, **align_params},
      )

    # Tightening the kernel is right; doing it from step 0 is not. Run 47063 held
    # align_std at 0.26 (15 deg) for the whole run and scored 22.5% against the
    # 0.436 baseline's 36.0% -- a 13.5 point LOSS. The reason is the same one that
    # killed the 15 mm pad_touch kernel: early on the jaw is ~22 deg off by
    # default, and at std 0.26 that collects 0.06 of the gate, so a tight kernel
    # multiplies the already-sparse grasp reward by ~0.1 during exactly the window
    # where grasping has to be discovered at all.
    #
    # As a schedule it costs nothing. Measured on run 46934, `grasp` is still
    # 0.0000 at iteration 600, reaches 0.15 by ~750 and 0.57 by ~900, and success
    # does not leave zero until ~2000. So hold the kernel wide until grasping is
    # established, tighten once there is a grasp to sharpen, and be at the tight
    # value with ~1000 iterations left to exploit it.
    if align_curriculum:
      cfg.curriculum["align_std_schedule"] = CurriculumTermCfg(
        func=manipulation_mdp.reward_curriculum,
        params={
          "reward_name": "grasp",
          "stages": [
            {"step": 0, "params": {"align_std": 0.436}},  # 25 deg
            {"step": 1000 * 24, "params": {"align_std": 0.35}},  # 20 deg
            {"step": 2000 * 24, "params": {"align_std": 0.26}},  # 15 deg
          ],
        },
      )

  # Strong early to bootstrap closing, then decayed so the policy is pushed from
  # "pinch and hold" toward actually lifting (where lift/lift_precise dominate).
  cfg.curriculum["grasp_weight"] = CurriculumTermCfg(
    func=manipulation_mdp.reward_curriculum,
    params={
      "reward_name": "grasp",
      "stages": [
        {"step": 0, "weight": 1.0},
        {"step": 1500 * 24, "weight": 0.6},
        {"step": 3000 * 24, "weight": 0.3},
      ],
    },
  )

  # --- Joint-velocity penalty ramp -------------------------------------------
  # The stock recipe ramps joint_vel_hinge -0.01 -> -0.1 (iter 500) -> -1.0
  # (iter 1000). On the simple parallel gripper that just smooths a policy that
  # already lifts; here the last state run started producing brief lifts and then
  # regressed into "pinch and press down on the table" right as the ramp hit
  # -1.0 -- a 100x velocity penalty makes standing still cheaper than the
  # transient velocity a lift costs. JOINT_VEL_PENALTY_FINAL caps the ramp.
  vel_curr = cfg.curriculum["joint_vel_hinge_weight"]
  vel_curr.params["stages"] = [
    {"step": 0, "weight": -0.01},
    {"step": 500 * 24, "weight": max(-0.1, joint_vel_penalty_final)},
    {"step": 1000 * 24, "weight": joint_vel_penalty_final},
  ]

  # --- Two ways to unfreeze the wrist roll (joint7) ---------------------------
  # Measured ceiling (scripts/probe_joint7_value.py on run 46984's final policy,
  # 128 envs): success by jaw-yaw-error quartile is 85.1% / 86.2% / 60.0% / 50.0%
  # over 0-11 / 11-22 / 22-34 / 34-45 deg, and locking the cube yaw so the frozen
  # wrist is already square takes success 78.1% -> 89.1%. So the payoff is +10.9
  # points and it is concentrated entirely in the worst quarter of envs -- the
  # response is a cliff at 22.5 deg, not a slope. Note the mean error is 14.1 deg,
  # not the 22.5 deg a fully frozen jaw under uniform yaw would give: the fingers
  # already rotate the cube most of the way square on contact. What is missing is
  # only the tail that they cannot rotate.
  #
  # Four reward-side attempts (AlignTight/Lift/Touch/Curr) all lost to the
  # baseline, so both of these attack EXPLORATION instead:
  #
  #   free_wrist_penalties  stop charging joint7 for moving. Removes the force
  #                         that crushed its action std to 0.0083 rad during the
  #                         ~2000 iterations when `grasp` still read 0.0000.
  #   wrist_reset_range     randomize joint7's reset angle. Does not merely permit
  #                         variance, it injects it, so the critic sees the
  #                         yaw-error/return relationship from iteration 0 whether
  #                         or not the actor ever explores its way there.
  if free_wrist_penalties:
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
      func=manipulation_mdp.action_rate_l2_except,
      weight=cfg.rewards["action_rate_l2"].weight,
      params={"exclude_joints": ("joint7",)},
    )
    vel_cfg = cfg.rewards["joint_vel_hinge"].params["asset_cfg"]
    assert isinstance(vel_cfg, SceneEntityCfg)
    vel_cfg.joint_names = tuple(j for j in ARM_JOINTS if j != "joint7") + (
      r"(left|right)_[12]",
    )

  if wrist_reset_range:
    cfg.events["reset_wrist_roll"] = EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-wrist_reset_range, wrist_reset_range),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("joint7",)),
      },
    )

  # --- Safety net: reset (not crash) on a rare mujoco-warp NaN blow-up ---
  cfg.terminations["nonfinite_state"] = TerminationTermCfg(
    func=manipulation_mdp.nonfinite_state,
    params={"asset_names": ("robot", "cube")},
  )

  # --- Success: cube lifted >= LIFT_HEIGHT and held HOLD_S ---
  cfg.terminations["cube_lifted"] = TerminationTermCfg(
    func=manipulation_mdp.cube_lifted,
    params={
      "object_name": "cube",
      "height": LIFT_HEIGHT,
      "hold_time": HOLD_S,
      "table_z": WORK_SURFACE_Z,
    },
  )

  # Succeeding currently costs half the return and is paid nothing for it.
  # `cube_lifted` above is a plain TerminationTermCfg, and `time_out` defaults
  # False, so it lands in `terminated` rather than `time_outs` -- the wrapper
  # only forwards `truncated` as time_outs -- and rsl_rl zeroes the bootstrap
  # value. Measured on run 46934: 14.4% of episode endings are cube_lifted, mean
  # episode length 933.8 of 1000, which puts successful episodes at step ~540; at
  # the measured ~0.056 return per step that forfeits ~26 of a ~52 return. The
  # dense stack meanwhile pays ~5/s indefinitely for hovering near the cube, so
  # "lift to 9 cm and stay there" strictly dominates "lift to 10 cm and hold".
  #
  # The policy still reached 26%, i.e. it is succeeding IN SPITE of the pay cut
  # and has not yet found the exploit (park under the line, or drop and re-catch
  # to reset the hold counter). That is a ceiling now and a regression later.
  #
  # Both halves are needed and neither works alone: flagging the termination as a
  # time_out restores the bootstrap but still pays nothing for success, while a
  # success reward that still terminates still forfeits the remaining steps.
  if success_reward:
    cfg.terminations["cube_lifted"].time_out = True
    cfg.rewards["lift_hold"] = RewardTermCfg(
      func=manipulation_mdp.lift_hold_bonus,
      weight=success_reward,
      params={
        "object_name": "cube",
        "height": LIFT_HEIGHT,
        "hold_time": HOLD_S,
        "table_z": WORK_SURFACE_Z,
        "left_sensor": "left_cube_contact",
        "right_sensor": "right_cube_contact",
      },
    )

  cfg.viewer.body_name = "link7"
  return cfg


def flexiv_two_finger_grasp_env_cfg(
  play: bool = False,
  joint_vel_penalty_final: float = -0.1,
  goal_mode: Literal["dynamic", "above_object"] = "dynamic",
  bringing_std: float = 0.3,
  grasp_shaping: Literal["proximity", "antipodal"] = "proximity",
  symmetry_penalty: float = 0.0,
  finger_range: FingerRange | None = None,
  reach_from: Literal["palm", "pads"] = "palm",
  align_weight: float = 0.0,
  pinch_z_std: float = 0.06,
  touch_weight: float = 0.0,
  success_reward: float = 0.0,
  align_gate: float = 0.0,
  align_std: float = 0.436,
  align_on: tuple[str, ...] = ("grasp",),
  align_curriculum: bool = False,
  free_wrist_penalties: bool = False,
  wrist_reset_range: float = 0.0,
  solve_wrist: bool = False,
  physics_dr: bool = False,
) -> ManagerBasedRlEnvCfg:
  """State-based grasp: privileged cube pose in the observation."""
  cfg = _apply_common(
    make_lift_cube_env_cfg(),
    joint_vel_penalty_final,
    goal_mode,
    bringing_std,
    grasp_shaping,
    symmetry_penalty,
    finger_range,
    reach_from,
    align_weight,
    pinch_z_std,
    touch_weight,
    success_reward,
    align_gate,
    align_std,
    align_on,
    align_curriculum,
    free_wrist_penalties,
    wrist_reset_range,
    solve_wrist,
  )

  # NOTE: there is deliberately no cube-colour randomization. The scene is a
  # fixed high-contrast pairing (white cube on a black table, see CUBE_RGBA /
  # TABLE_RGBA) so the cube is trivially separable in the D435 image. Re-add a
  # dr.geom_rgba reset event here if the vision policy needs colour robustness.

  if physics_dr:
    add_physics_dr(cfg)
    # Spawn the cube clear of the highest bench draw so it is always in free
    # space and settles, rather than starting interpenetrated with a foam that
    # moved up under it -- HARD_SOLREF would launch it. The drop is at most
    # 14 mm, which settles in well under one control step.
    lift_cmd = cfg.commands["lift_height"]
    assert isinstance(lift_cmd, LiftingCommandCfg)
    assert lift_cmd.object_pose_range is not None
    lift_cmd.object_pose_range.z = (RESTING_Z + 0.005, RESTING_Z + 0.010)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)
  return cfg


# Depth sensor DR, in the units the observation is in (metric depth divided by
# the 3.0 m cutoff, so 0.01 is 30 mm) except where a comment says metres.
#
#   RANGE NOISE is not optional and not a detail. Measured upstream (job 42184):
#   a student trained noise-free and evaluated noise-free scores 67-69%, and the
#   SAME checkpoint evaluated under +-0.01 scores ZERO. Rendered depth is exact,
#   and a CNN fitted to exact depth learns to read absolute values that a real
#   D435 simply does not report.
#
#   THE INVALID PIXELS ARE NOT SALT AND PEPPER. What the bench produces is two
#   structured failures with geometry behind them -- an occlusion shadow beside
#   every depth step and blind patches ON the surfaces themselves -- plus a
#   small speckle residue. Both are modelled below from that geometry. The old
#   `patch_prob` (random 8x8 blocks, position independent of the scene and
#   resampled every frame) stood in for them and is now removed: its position
#   carried no information a policy could learn against, and its per-frame
#   resampling was averageable in a way a real dropout is not.
_DEPTH_NOISE = 0.01
# Speckle residue only. Measured blob p50 is 2 px, so there IS a genuine
# salt-and-pepper component -- it is just not 2% of the frame, which is what the
# two structured terms below were being asked to hide inside.
_DEPTH_DROPOUT = (0.005, 0.01)
_DEPTH_SCALE_ERR = 0.01

# --- occlusion shadow -------------------------------------------------------- #
# Horizontal focal length of the RENDERED frame, in pixels. Not a free
# parameter: the 160x120 render subtends 73.53 deg horizontally (see the
# CameraSensorCfg comment above), so f = 80 / tan(73.53/2 deg) = 107.08. It has
# to be the render's f and not the D435's, because the shadow is measured in
# the pixels the student actually sees.
_DEPTH_SHADOW_FOCAL_PX = 107.08
# Stereo baseline, metres. NOT fitted -- inverting the measured shadow widths
# back through w = f*B*(1/z_fg - 1/z_bg) gives 50 +- 6 mm, which is the D435's
# nominal baseline. The band is that measurement's spread, not a guess.
_DEPTH_SHADOW_BASELINE = (0.044, 0.056)
# How much of a shadow band actually comes back invalid.
_DEPTH_SHADOW_FILL = (0.70, 0.95)
# Metric step counting as a background -> foreground edge, metres.
_DEPTH_SHADOW_STEP = 0.01

# --- surface blinding -------------------------------------------------------- #
# Fraction of each surface that returns nothing, drawn per episode.
#
# The claw number is measured: 4.9-22.7% of the visible claw comes back invalid
# on the bench, and it swings by 4x with viewing angle, which is why the band is
# wide rather than centred on a mean.
_DEPTH_BLIND_GRIPPER = (0.05, 0.25)
# THE CUBE NUMBER IS THE ONE THING HERE THAT WAS NOT MEASURED. A matte white
# cube returns better than printed plastic, so it is set lower, but the band is
# an estimate and should be replaced by a bench measurement before it is used to
# explain anything.
_DEPTH_BLIND_OBJECT = (0.02, 0.15)
# The claw's four finger-link visual meshes, which carry the pads: the pad
# marker bodies are massless references with no geometry, so the pad surface a
# camera sees is part of the distal link mesh.
_CLAW_VIS_GEOMS = (r"(left|right)_[12]_(left|right)_[12]_vis_tf",)


def _add_d435_camera_group(
  cfg: ManagerBasedRlEnvCfg,
  cam_type: Literal["rgb", "depth", "rgbd"],
  percept_dr: bool = False,
) -> None:
  """Attach the D435 sensor and a "camera" observation group, in place."""
  modalities: tuple[CameraDataType, ...] = (
    ("rgb", "depth") if cam_type == "rgbd" else (cam_type,)
  )

  # Surface blinding needs to know which pixels are claw and which are cube, so
  # it renders segmentation as well -- but segmentation is an INPUT to the depth
  # DR, not an observation. It is added to `data_types` and deliberately not to
  # `modalities`, so no term is built for it and the student never sees it.
  blind_on = percept_dr and DEPTH_SENSOR_DR and "depth" in modalities
  data_types: tuple[CameraDataType, ...] = (
    (*modalities, "segmentation") if blind_on else modalities
  )

  # One D435 sensor producing every requested modality (rgb + aligned depth).
  cam_cfg = CameraSensorCfg(
    name="d435",
    camera_name="robot/scene_cam",
    # 160x120, NOT the old 128x72. The D435i intrinsics in arm_cfg.py are
    # calibrated for the 848x480 depth profile, and mujoco-warp's renderer
    # shrinks whichever sensor dimension is too large for the render aspect,
    # so a 4:3 render crops the 16:9 sensor to EXACTLY a 640x480 centre-crop
    # (3.60/4.77 * 848 = 640.0000 px) and the student sees 73.53 x 58.53 deg.
    # That makes the deployment recipe one line and no retrain:
    #   848x480 -> centre-crop to 640x480 -> resize to 160x120.
    # Rendering 16:9 instead would squash 89.42 deg into pixels the student
    # learned as 73.53 deg, and would also cost resolution on the one small
    # thing that has to be resolved: 2.176 px/deg here against 1.789 at 16:9.
    height=120,
    width=160,
    data_types=data_types,
    # (0, 1) AND NOT THE DEFAULT (0, 1, 2). On this model:
    #
    #   group 0  the Rizon's own link visuals, plus the bench -- table, foam,
    #            wall and cube all default to it
    #   group 1  the claw, the base plate and the camera mount
    #   group 2  THE D435i'S OWN 9 VISUAL MESHES, and nothing else
    #   group 3  every collider; rendering it would draw this claw as its
    #            collision hulls, which is what the old (0, 1, 3) did
    #
    # Group 2 is dropped because a camera cannot see its own housing, and on
    # this model it very much could. The lens sits INSIDE that housing: ray-cast
    # from the camera at the home pose, the D435i's own mesh is 0.21 mm away
    # along the optical axis and within 1.23-7.8 mm in every other direction.
    # Backface culling hides it at the nominal pose, which is why this was
    # invisible -- but `dr_cam_pos` jitters the lens by +-3 mm, fourteen times
    # that clearance, and 5 of 64 environments then render a frame that is
    # ENTIRELY the inside of the housing (depth mean 0.0007-0.0086 against a
    # healthy 0.48). With the jitter off it is 0 of 64. Those environments hand
    # the student a blank near-clip image while the teacher labels them
    # normally, so DAgger fits the teacher's action to no scene at all.
    #
    # The clearance is NEW: before the stage-B extrinsic correction moved the
    # camera 19.9 mm (arm_cfg.CALIB_CAM_OFFSET) the optical axis left the
    # housing without hitting anything. The correction is a MEASUREMENT and
    # stays; what has to change is that the housing is drawn at all.
    enabled_geom_groups=(0, 1),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  terms: dict[str, ObservationTermCfg] = {}
  for mod in modalities:
    params: dict[str, Any] = {"sensor_name": "d435"}
    if mod == "depth":
      params["cutoff_distance"] = 3.0  # D435 usable range
      func = manipulation_mdp.camera_depth
      if percept_dr and DEPTH_SENSOR_DR:
        params["range_noise"] = _DEPTH_NOISE
        params["dropout_prob"] = _DEPTH_DROPOUT
        params["scale_err"] = _DEPTH_SCALE_ERR
        params["shadow_focal_px"] = _DEPTH_SHADOW_FOCAL_PX
        params["shadow_baseline"] = _DEPTH_SHADOW_BASELINE
        params["shadow_fill"] = _DEPTH_SHADOW_FILL
        params["shadow_step"] = _DEPTH_SHADOW_STEP
        params["blind_gripper_cfg"] = SceneEntityCfg(
          "robot", geom_names=_CLAW_VIS_GEOMS
        )
        params["blind_gripper_prob"] = _DEPTH_BLIND_GRIPPER
        params["blind_object_cfg"] = SceneEntityCfg("cube", geom_names=("cube",))
        params["blind_object_prob"] = _DEPTH_BLIND_OBJECT
    else:
      func = manipulation_mdp.camera_rgb
    terms[f"d435_{mod}"] = ObservationTermCfg(func=func, params=params)
  # Camera terms are (B, C, H, W); concatenate along the per-env channel dim
  # (index 0 without the batch axis; the manager shifts it to runtime dim 1) so
  # RGB (3ch) + depth (1ch) stack into a single 4-channel RGB-D image the CNN
  # encoder ingests. For a single modality this is a no-op.
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )
  if percept_dr:
    add_percept_dr(cfg)


# Observation terms carrying cube pose, i.e. exactly what the camera has to
# replace. Everything else in the actor group is proprioception the real robot
# reports directly.
_PRIVILEGED_TERMS = ("ee_to_cube", "cube_to_goal", "cube_quat")


def flexiv_two_finger_vision_env_cfg(
  cam_type: Literal["rgb", "depth", "rgbd"],
  play: bool = False,
  joint_vel_penalty_final: float = -0.1,
) -> ManagerBasedRlEnvCfg:
  """Vision-based grasp: D435 RGB, depth, or both + proprioception, no cube pose.

  ``cam_type="rgbd"`` feeds the policy the RGB image *and* the aligned depth
  image from the same D435 (two observation terms concatenated), mirroring the
  real sensor's registered colour+depth streams.
  """
  cfg = flexiv_two_finger_grasp_env_cfg(
    play=play, joint_vel_penalty_final=joint_vel_penalty_final
  )
  _add_d435_camera_group(cfg, cam_type)

  # Drop privileged cube info from the actor observation. cube_quat is included:
  # the vision policy has to read the cube's yaw off the image, which is the
  # whole point -- leaving it in would hand it the answer. The critic keeps it.
  actor = cfg.observations["actor"]
  for term in _PRIVILEGED_TERMS:
    actor.terms.pop(term, None)
  return cfg


def flexiv_two_finger_distill_env_cfg(
  cam_type: Literal["rgb", "depth", "rgbd"],
  play: bool = False,
  percept_dr: bool = False,
  **teacher_kwargs: Any,
) -> ManagerBasedRlEnvCfg:
  """Teacher + student observations side by side, for DAgger distillation.

  Three observation groups come out of this env:

    actor    the state teacher's observation, UNTOUCHED. The teacher's weights
             were fitted to this exact vector in this exact order, so nothing
             here may be added, removed or reordered -- pass the teacher run's
             own kwargs through ``teacher_kwargs`` and the group is identical
             by construction.
    student  proprioception only: joint positions, velocities, last action, and
             the commanded goal height. No cube pose in any form.
    camera   the D435 image, which is where the cube has to come from instead.

  ``critic`` is left in place but unused: distillation regresses onto the
  teacher's actions and never estimates a value.

  Note the student sees the goal HEIGHT (``goal_height``) where the teacher sees
  a full goal-minus-cube vector. The commanded lift height is sampled per
  episode over a 10 cm range, so a student without it is being asked to guess
  how high to lift; it is a task input, not privileged state.
  """
  cfg = flexiv_two_finger_grasp_env_cfg(play=play, **teacher_kwargs)
  _add_d435_camera_group(cfg, cam_type, percept_dr=percept_dr)

  student_terms = {
    name: deepcopy(term)
    for name, term in cfg.observations["actor"].terms.items()
    if name not in _PRIVILEGED_TERMS
  }
  student_terms["goal_height"] = ObservationTermCfg(
    func=manipulation_mdp.goal_height,
    params={"command_name": "lift_height"},
  )
  cfg.observations["student"] = ObservationGroupCfg(
    terms=student_terms,
    # Same sensor noise the teacher trained under, so the student is not
    # distilled on a cleaner signal than it will ever see.
    enable_corruption=not play,
  )

  # AFTER the student group exists, and that ordering is the whole point: the
  # student is a deepcopy of the teacher's terms, so lagging them any earlier
  # lags the TEACHER and the student merely inherits it. The teacher would then
  # be labelling from stale joint angles, which makes the labels worse rather
  # than the student tougher.
  if percept_dr:
    add_sensor_latency(cfg)

  # End the episode once the cube is off the table. The state task runs without
  # this and merely wastes the remaining steps; under DAgger every one of those
  # steps becomes a labelled training sample, and they are all "what would the
  # teacher do about a cube on the floor". Measured on run 50426: the student
  # knocks the cube off in the first few steps, `object_height` then reads 0.197
  # against a 0.400 table top, `Mean episode length` stays pinned at the full
  # 1000, and the behaviour loss never leaves 1.0. Deliberately NOT added to the
  # state task -- that would change the env the teacher was trained in.
  if not play:
    cfg.terminations["cube_dropped"] = TerminationTermCfg(
      func=manipulation_mdp.object_dropped,
      params={"object_name": "cube", "table_z": WORK_SURFACE_Z},
    )

  # Report the success the task is actually scored on. The stock metric is the
  # 3D distance to a goal sampled in [10, 20] cm, while `cube_lifted` ends the
  # episode at 10 cm -- so above a 15 cm draw the metric can never latch, and
  # the same policy reads 84.0% or 24.5% purely on the goal draw. Only the
  # distill tasks are switched: changing it on the state task would make
  # v11_align's logged history incomparable, and it costs nothing there because
  # `eval_deploy.py` is the number of record either way.
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  lift_cmd.success_mode = "lift_height"
  lift_cmd.success_height = LIFT_HEIGHT
  lift_cmd.success_table_z = WORK_SURFACE_Z
  return cfg
