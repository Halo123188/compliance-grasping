"""Tabletop grasp tasks for the Flexiv Rizon4S + custom two-finger hand.

Two tasks share one scene (arm + hand on a 32x48 in table, a cube, and a fixed
D435 RGB-D camera in front of the arm base):

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

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  TABLE_HEIGHT,
  get_two_finger_hand_robot_cfg,
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
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

IN = 0.0254
TABLE_LEN_X = 48 * IN
TABLE_WID_Y = 32 * IN
TABLE_H = TABLE_HEIGHT  # table-top height above the floor (arm is mounted on it)
TABLE_BACK_X = -0.10
CUBE_HALF = 0.025  # 5 cm cube -- matches the real object being grasped
# Cube spawn region, trimmed to what the D435 can actually see. The camera sits
# at x = 0.106 on the base X axis with a ~89 deg horizontal FOV, so a point is in
# frame only while |y| < 0.985 * (x - 0.106). The old (0.32, +-0.28) rectangle
# put its near corners outside that cone -- roughly 3% of spawns started with the
# cube not in the image at all, which is unlearnable for the vision task. This
# box clears the cone by ~17% at its worst corner (x=0.35: limit 0.240 vs 0.215
# for the far edge of the cube).
CUBE_REGION = dict(x=(0.35, 0.60), y=(-0.20, 0.20))
LIFT_HEIGHT = 0.10  # m above the table top the cube must reach
HOLD_S = 1.0

# High-contrast work surface: a white cube on a matte black table, so the cube
# is the brightest thing in the D435 frame by a wide margin. Not quite 0/1 --
# pure black kills all shading cues on the table and pure white clips.
TABLE_RGBA = (0.04, 0.04, 0.045, 1.0)
CUBE_RGBA = (0.93, 0.93, 0.93, 1.0)

# Hard contact impedance for the work surfaces. MuJoCo's default
# solimp=(0.9, 0.95, ...) is soft enough that the gripper pressing down (~50 N)
# buries the cube 5 mm into the table -- a third of its half-height, and clearly
# visible as clipping. Measured penetration at 50 N: default 5.05 mm, this
# 0.63 mm. solref is deliberately left at the default 0.02 s: the minimum safe
# value is 2x the 0.005 s timestep, and going to that bound has previously made
# this scene unstable. Contact params are mixed between the two geoms in a pair,
# so this must be set on the table, the cube AND the hand colliders to take full
# effect.
HARD_SOLIMP = (0.98, 0.999, 0.001, 0.5, 2.0)

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
FINGER_JOINTS = ("left_1", "left_2", "right_1", "right_2")
FINGERTIP_GEOMS = r"(left_2|right_2)_col"
LEFT_FINGERTIP_GEOMS = r"left_2_col"
RIGHT_FINGERTIP_GEOMS = r"right_2_col"
# Sites on the REAL inner gripping faces (see two_finger_hand.xml). The
# collision geoms' frame origins sit 10-26 mm behind these, by a margin that
# changes with the jaw angle, so they cannot stand in for the contact surface.
PAD_SITES = ("left_pad", "right_pad")

# Separation of the two PAD_SITES while the hand is actually GRIPPING the 50 mm
# cube -- fingers driven to the pull-out-test pose and stopped by the cube at
# left_1 = +0.234, not commanded into free space. Those are different numbers and
# the free-space one is the wrong one: unloaded, the fingers close to a 36.3 mm
# site separation, but with the cube in the way they stop at 64.4 mm.
#
# It is not the cube's 50 mm either. The sites sit 8.6 mm behind each gripping
# face, so the pair straddles the cube 7.2 mm clear on each side -- see the note
# on left_pad in two_finger_hand.xml.
PAD_SEP_AT_GRASP = 0.0644

# (proximal, distal) limits for the LEFT finger; mirrored onto the right.
FingerRange = tuple[tuple[float, float], tuple[float, float]]


def get_cube_spec(size: float = CUBE_HALF) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(size,) * 3,
    mass=0.05,
    rgba=CUBE_RGBA,
    solimp=HARD_SOLIMP,
  )
  return spec


def get_table_spec() -> mujoco.MjSpec:
  """A 32 x 48 in solid table: a box from the floor (z=0) up to the top surface
  at z=TABLE_H. It is the physics work surface (the arm is bolted to it and the
  cube rests on it); the terrain plane at z=0 is the floor the table stands on.
  """
  spec = mujoco.MjSpec()
  cx = TABLE_BACK_X + TABLE_LEN_X / 2
  body = spec.worldbody.add_body(name="table", pos=(cx, 0.0, TABLE_H / 2))
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(TABLE_LEN_X / 2, TABLE_WID_Y / 2, TABLE_H / 2),
    rgba=TABLE_RGBA,
    solimp=HARD_SOLIMP,
  )
  return spec


def _robot_cfg(finger_range: FingerRange | None) -> EntityCfg:
  """The hand, optionally with the finger joints limited to their real travel.

  The XML ships +-1.6 rad on all four finger joints. The mechanism is done long
  before that: mirrored, the jaw is shut at left_1 = -0.20 (9.9 mm gap) and past
  it the fingers cross like scissors and the gap REOPENS -- 37.6 mm at -0.60,
  70.3 mm at -1.00 -- so angle -> gap is two-valued and a policy exploring there
  gets a gradient that means nothing. ``soft_joint_pos_limit_factor`` puts the
  ``joint_pos_limits`` penalty at +-1.44, which no useful pose comes near, so
  nothing charges for going there either.

  Ranges are given for the LEFT finger and mirrored onto the right, because
  closing is left-negative / right-positive: a single symmetric range cannot
  express the limit.

  The proximal and distal limits are NOT the same number. The distal has to curl
  well inward to grip -- the pull-out-test pose is left_2 = -0.40, and the
  distals only meet near -0.85 -- so reusing the proximal's -0.20 here would put
  the one grasp this hand is known to hold with outside the action space.
  Measured by scripts/diag_hand_limits.py.
  """
  cfg = get_two_finger_hand_robot_cfg()
  if finger_range is None:
    return cfg
  prox, dist = finger_range
  inner = cfg.spec_fn

  def spec_fn() -> mujoco.MjSpec:
    spec = inner()
    for j in spec.joints:
      lohi = prox if j.name in ("left_1", "right_1") else dist
      if j.name in ("left_1", "left_2"):
        j.range = [lohi[0], lohi[1]]
      elif j.name in ("right_1", "right_2"):
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
    "table": EntityCfg(spec_fn=get_table_spec),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

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

  # Start the hand hovering 10 cm above the centre of the cube region, pointing
  # straight down with the gripper open (IK-solved; grasp_site = (0.45, 0, 0.10)
  # in the base frame, i.e. 10 cm over the table top). The default HOME_KEYFRAME
  # folds the arm up at z~=1.25 m, forcing the policy to learn the whole 1.2 m
  # descent before it can even attempt a grasp; hovering over the workspace lets
  # it focus on the approach + close + lift.
  #
  # 10 cm rather than the earlier 20 cm because the D435 sits only 20.5 cm above
  # the table (the real mount height): a gripper hovering at the same height as
  # the camera lands on the very top edge of the frame. At 10 cm the whole hand
  # is comfortably inside the image.
  cfg.scene.entities["robot"].init_state = EntityCfg.InitialStateCfg(
    # Fixed base -> this pos overwrites the spec's base.pos, so it must carry the
    # table height or the arm ends up standing on the floor, passing through the
    # table (see Entity._add_initial_state_keyframe).
    pos=(0.0, 0.0, TABLE_H),
    joint_pos={
      "joint1": -0.1104,
      "joint2": -0.4672,
      "joint3": 0.2889,
      "joint4": 2.3750,
      "joint5": -0.4049,
      "joint6": 1.2378,
      "joint7": 0.6380,
      # Start ALREADY SHAPED like the grasp: distals curled inward, proximals
      # open only as far as clearance demands. The previous hover had the
      # distals dead straight, which left the policy to discover the inward
      # curl on its own -- and that curl is the whole grasp: with it the
      # pull-out force is 5.2-6.1 N, without it 1.3-3.1 N on a cube that weighs
      # 0.49 N. Run 45530 never found it, and settled with both distals curled
      # the WRONG way (left_2 = +1.05).
      #
      # Sizing is set by the cube's DIAGONAL, not its face: yaw is randomized
      # over the full circle, so at 45 deg the cube presents 50*sqrt(2) =
      # 70.7 mm. Jaw gap at curl -0.40, measured on the real mesh colliders by
      # scripts/diag_hover_pose.py:
      #   +0.20 -> 41.6 | +0.30 -> 56.2 | +0.40 -> 70.7 | +0.50 -> 85.1
      #   +0.60 -> 99.2 | +0.70 -> 112.9 | +0.90 -> 138.4
      # +0.50 clears the diagonal by 14.4 mm (7.2 mm per side). +0.40 is exactly
      # the diagonal, i.e. zero clearance, and the old +0.40/straight pose
      # (74.7 mm, 2 mm clearance) demonstrably hit corners and knocked the cube
      # away (object_height fell to 0.37 in runs 45029/45030).
      #
      # This also shortens ACTION REACH (see cfg.actions), which is what killed
      # four earlier runs. Closing to the grasp pose (+0.10, -0.40) is now
      # -0.67 sigma on the proximal and 0 sigma on the distal, against -1.0 and
      # -0.67 from the old hover.
      #
      # Cost, measured: curling the distals drops the fingertips 22 mm, so at
      # the moment grasp_site reaches the cube centre the tips clear the table
      # by 7.1 mm rather than 29.1 mm. Still clear, but this is the budget that
      # gets spent if the curl is ever deepened here.
      "left_1": 0.50,
      "left_2": -0.40,
      "right_1": -0.50,
      "right_2": 0.40,
    },
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
  # All FOUR finger joints are driven independently -- left_1/left_2 and
  # right_1/right_2 are separate actuators with no mimic or tendon coupling, so
  # the policy commands each knuckle. Action dim = 7 arm + 4 finger = 11.
  #
  # ACTION REACH. With use_default_offset the command is
  # `default + scale * action`, so the action needed to reach a pose is
  # (target - default) / scale -- and anything past ~1.5 sigma is a pose
  # Gaussian exploration will never stumble into, which means no gradient ever
  # forms toward it. Two DOF groups needed their own scale:
  #
  #  * fingers (0.6). Closing from the 0.70 hover onto the cube (left_1 ~ 0.10)
  #    is -0.60 rad = -1.0 sigma; the distal curl that makes the grasp strong
  #    (left_2 ~ -0.50, see below) is -0.83 sigma. At the arm's 0.3 those were
  #    -2.0 and -1.7 sigma.
  #  * joint7 (0.8), the wrist roll that squares the jaw to a cube face. Cube
  #    yaw is uniform on [-pi, pi] and the jaw is 180-deg symmetric, so
  #    alignment needs a full pi of travel, i.e. +-1.57 rad about the hover
  #    default. At 0.3 that is +-5.2 sigma -- the policy physically could not
  #    turn the wrist far enough to grasp most spawns. At 0.8 it is +-2.0.
  #    joint7's limit is +-3.05 rad, so this scale stays well inside it.
  #
  # Joint-space (not Cartesian IK) because IK decouples the arm from the gripper
  # and its Gaussian action noise does not readily explore the coordinated
  # close+lift.
  #
  # With `solve_wrist`, joint7 is not learned at all: it is driven to the angle
  # that squares the jaw, re-solved from measured state every substep, and the
  # policy keeps only a small trim on top. See
  # WristAlignedJointPositionActionCfg for why -- six RL variants failed to learn
  # a one-line geometry problem. The trim scale is 0.15 rad (8.6 deg), inside the
  # 0-22.5 deg band where alignment provably does not affect success, so the
  # policy cannot undo the solution but still has a real gradient on the joint.
  wrist_scale = 0.15 if solve_wrist else 0.8
  action_scale = {
    r"joint[1-6]": 0.3,
    r"joint7": wrist_scale,
    r"(left|right)_[12]": 0.6,
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
    "grasp_site",
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
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

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
        "joint_pairs": (("left_1", "right_1"), ("left_2", "right_2")),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )

  # --- Cube spawn: reachable region on the table ---
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=CUBE_REGION["x"],
    y=CUBE_REGION["y"],
    z=(TABLE_H + CUBE_HALF, TABLE_H + CUBE_HALF),  # resting on the table top
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
    z=(TABLE_H + 0.10, TABLE_H + 0.20),
  )

  # --- Fingertip friction DR on the distal finger colliders ---
  for ev in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[ev].params["asset_cfg"].geom_names = FINGERTIP_GEOMS

  # --- Crash guard: the PALM (not the fingers) slamming into the table ---
  # A grasp needs the fingertips to reach the table top, so scope this to the
  # palm's own geom ("body" mode on base_link) -- the ~11 cm fingers keep the
  # palm well above the surface during a grasp, so it fires only on a genuine
  # arm-into-table crash. Secondary is the table (the work surface).
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.mode = "body"
      sensor.primary.pattern = "base_link"
      sensor.secondary = ContactMatch(mode="body", pattern="table", entity="table")

  # --- Soft penalty on fingertips mashing the table -------------------------
  # The fingers physically collide with the table (a solid box, so they can't
  # penetrate it) and light contact is unavoidable when grasping a low object.
  # Rather than terminate, we measure fingertip-vs-ground contact force and
  # penalize the hard part of it, so the policy learns to grip the cube cleanly
  # instead of driving the tips into the table.
  fingertip_table_contact = ContactSensorCfg(
    name="fingertip_table_contact",
    primary=ContactMatch(mode="geom", pattern=FINGERTIP_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="table", entity="table"),
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
      "table_z": TABLE_H,
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
        "table_z": TABLE_H,
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

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)
  return cfg


def _add_d435_camera_group(
  cfg: ManagerBasedRlEnvCfg,
  cam_type: Literal["rgb", "depth", "rgbd"],
) -> None:
  """Attach the D435 sensor and a "camera" observation group, in place."""
  modalities: tuple[str, ...] = ("rgb", "depth") if cam_type == "rgbd" else (cam_type,)

  # One D435 sensor producing every requested modality (rgb + aligned depth).
  cam_cfg = CameraSensorCfg(
    name="d435",
    camera_name="robot/d435",
    height=72,
    width=128,  # 16:9, downsampled from 1280x720 for RL
    data_types=modalities,
    enabled_geom_groups=(0, 1, 3),
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
  _add_d435_camera_group(cfg, cam_type)

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
      params={"object_name": "cube", "table_z": TABLE_H},
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
  lift_cmd.success_table_z = TABLE_H
  return cfg
