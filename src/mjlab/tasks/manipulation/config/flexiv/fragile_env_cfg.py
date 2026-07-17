"""Flexiv fragile single-cube lift: lift 10 cm and hold for 2 s without crushing.

The object is fragile: total grasp normal force (from the two fingertip FT
sensors) must stay in ``[Fn_min, F_break]``. Exceeding ``F_break`` crushes it
(terminal failure) and a shaping penalty discourages approaching it. Only three
quantities are randomized per episode: mass ``m``, friction ``mu``, and the
band-width factor ``k`` that sets ``F_break = k * Fn_min``.
"""

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import (
  FLEXIV_ACTION_SCALE,
  FT_FORCE_SENSOR_NAMES,
  FT_NORMAL_AXIS,
  FT_TORQUE_SENSOR_NAMES,
  get_flexiv_robot_cfg,
  get_ft_sensor_cfgs,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manipulation.config.flexiv import fragile_mdp
from mjlab.tasks.manipulation.config.flexiv.env_cfgs import get_cube_spec
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# FT sensors register under their entity-prefixed scene names.
_ENTITY = "robot"
FORCE_SENSORS = tuple(f"{_ENTITY}/{n}" for n in FT_FORCE_SENSOR_NAMES)
TORQUE_SENSORS = tuple(f"{_ENTITY}/{n}" for n in FT_TORQUE_SENSOR_NAMES)

# Deterministic single-cube geometry: fixed spawn, target 10 cm above it.
_CUBE_HALF = 0.02
_SPAWN_XYZ = (0.5, 0.0, _CUBE_HALF)
_LIFT_HEIGHT = 0.10
_SUCCESS_TOL = 0.02
_HOLD_TIME_S = 2.0


def flexiv_fragile_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  # ── Scene: robot + single cube, plus the two fingertip FT sensors ──────────
  cfg.scene.entities = {
    "robot": get_flexiv_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }
  assert cfg.scene.sensors is not None
  cfg.scene.sensors = tuple(cfg.scene.sensors) + get_ft_sensor_cfgs(_ENTITY)

  # ── Actions ────────────────────────────────────────────────────────────────
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = FLEXIV_ACTION_SCALE

  # ── Observations ────────────────────────────────────────────────────────────
  # The actor sees noisy sensors (corruption on); the critic sees ground truth
  # (corruption off). Crucially, the crush event and the force penalties are
  # adjudicated on the true contact force (they call the sensors directly, never
  # the noised observation), so sensor noise can never trigger a phantom break.
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  actor.enable_corruption = True
  critic.enable_corruption = False

  actor.terms["ee_to_cube"].params["asset_cfg"].site_names = ("grasp_site",)
  critic.terms["ee_to_cube"].params["asset_cfg"].site_names = ("grasp_site",)

  # Actor sees the noisy 12-D FT wrench; must infer the force budget from it.
  # The same term object feeds the critic, but with corruption off there it is
  # read clean (privileged true wrench).
  ft_term = ObservationTermCfg(
    func=fragile_mdp.ft_wrench,
    params={
      "force_sensor_names": FORCE_SENSORS,
      "torque_sensor_names": TORQUE_SENSORS,
    },
    noise=Unoise(n_min=-0.25, n_max=0.25),  # N; starting FT sensor noise
  )
  actor.terms["ft_wrench"] = ft_term
  critic.terms["ft_wrench"] = ft_term
  # Privileged critic-only obs: episode params [mass, mu, F_break], the
  # precomputed force budget [Fn_min, slack, k] (so the critic need not learn
  # m*g/mu), and the true object velocity.
  critic.terms["grasp_params"] = ObservationTermCfg(
    func=fragile_mdp.privileged_grasp_params
  )
  critic.terms["force_budget"] = ObservationTermCfg(
    func=fragile_mdp.force_budget_features,
    params={
      "force_sensor_names": FORCE_SENSORS,
      "normal_axis": FT_NORMAL_AXIS,
    },
  )
  critic.terms["object_vel"] = ObservationTermCfg(
    func=fragile_mdp.object_velocity, params={"object_name": "cube"}
  )

  # ── Commands: fixed spawn, target exactly _LIFT_HEIGHT above it ─────────────
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  sx, sy, sz = _SPAWN_XYZ
  lift_cmd.difficulty = "dynamic"
  lift_cmd.success_threshold = _SUCCESS_TOL
  lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(sx, sx), y=(sy, sy), z=(sz, sz), yaw=(0.0, 0.0)
  )
  lift_cmd.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(sx, sx), y=(sy, sy), z=(sz + _LIFT_HEIGHT, sz + _LIFT_HEIGHT)
  )

  # ── Randomization: exactly three knobs (mass, mu, k) ───────────────────────
  for key in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events.pop(key, None)
  cfg.events["randomize_grasp"] = EventTermCfg(
    func=fragile_mdp.randomize_grasp,
    mode="reset",
    params={
      "mass_range": (0.1, 1.5),
      "mu_range": (0.4, 1.0),
      "k_range": (1.5, 20.0),
      "object_name": "cube",
      "cube_half_extent": _CUBE_HALF,
      "robot_name": "robot",
      "finger_geom_names": ("left_finger_col", "right_finger_col"),
    },
  )

  # ── Rewards: reuse reach/lift shaping + FT force-margin penalty ─────────────
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)
  cfg.rewards["force_margin"] = RewardTermCfg(
    func=fragile_mdp.force_margin_penalty,
    weight=-1.0,
    params={
      "force_sensor_names": FORCE_SENSORS,
      "ramp": 0.8,
      "normal_axis": FT_NORMAL_AXIS,
    },
  )
  # Independent efficiency penalty anchored at Fn_min (not F_break): pushes the
  # grasp toward the minimum holding force instead of only avoiding the crush
  # limit. Weight is the coefficient c in -c*max(0, Fn/Fn_min - 1); tune it up if
  # the policy still over-grips, down if it under-grips and drops the object.
  cfg.rewards["force_over_min"] = RewardTermCfg(
    func=fragile_mdp.force_over_min_penalty,
    weight=-0.2,
    params={
      "force_sensor_names": FORCE_SENSORS,
      "normal_axis": FT_NORMAL_AXIS,
    },
  )

  # ── Terminations: crush = failure, held-for-2 s = success ──────────────────
  cfg.terminations["object_crushed"] = TerminationTermCfg(
    func=fragile_mdp.object_crushed,
    params={"force_sensor_names": FORCE_SENSORS, "normal_axis": FT_NORMAL_AXIS},
  )
  cfg.terminations["lift_success"] = TerminationTermCfg(
    func=fragile_mdp.lift_hold_success,
    params={
      "command_name": "lift_height",
      "object_name": "cube",
      "height_tol": _SUCCESS_TOL,
      "hold_time_s": _HOLD_TIME_S,
    },
  )

  # Terminate if the gripper subtree slams the ground.
  from mjlab.sensor import ContactSensorCfg

  for sensor in cfg.scene.sensors:
    if isinstance(sensor, ContactSensorCfg) and sensor.name == "ee_ground_collision":
      sensor.primary.pattern = "gripper_base"

  cfg.curriculum = {}
  cfg.viewer.body_name = "link7"
  cfg.episode_length_s = 8.0

  if play:
    cfg.episode_length_s = int(1e9)
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)
    # Evaluate on clean sensors.
    cfg.observations["actor"].enable_corruption = False

  return cfg
