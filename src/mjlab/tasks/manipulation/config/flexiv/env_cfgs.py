"""Pick-and-place environment configurations for Flexiv Rizon4S."""

import mujoco

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import (
  FLEXIV_ACTION_SCALE,
  get_flexiv_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg


def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def flexiv_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flexiv pick-and-place: cube spawns on table at arm's reach, target is in the air."""
  cfg = make_lift_cube_env_cfg()

  # --- Scene entities ---
  cfg.scene.entities = {
    "robot": get_flexiv_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # --- Actions ---
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = FLEXIV_ACTION_SCALE

  # --- Observations ---
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  # --- Commands ---
  # Move cube spawn range further from the robot base (Flexiv has ~0.9 m reach).
  # Target position keeps the default z=(0.2, 0.4) → object must be lifted into the air.
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.40, 0.65),
    y=(-0.15, 0.15),
    z=(0.008, 0.010),
    yaw=(-3.14, 3.14),
  )

  # --- Domain randomization ---
  fingertip_geoms = "left_finger_col|right_finger_col"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  cfg.events["cube_color"] = EventTermCfg(
    func=dr.geom_rgba,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
      "operation": "abs",
      "distribution": "uniform",
      "axes": [0, 1, 2],
      "ranges": (0.2, 1.0),
    },
  )

  # --- Contact sensor: terminate when gripper subtree hits the ground ---
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "gripper_base"

  cfg.viewer.body_name = "link7"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)

  return cfg
