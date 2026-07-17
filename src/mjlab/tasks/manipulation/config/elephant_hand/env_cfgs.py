"""Pick-and-place env config for Flexiv Rizon 4S + mygripper_H100_R elephant hand."""

import mujoco

from mjlab.asset_zoo.robots.elephant_hand.constants import (
  _ARM_HAND_JOINT_POS_COUPLED,
  ELEPHANT_ACTION_SCALE,
  get_elephant_arm_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

# Pre-grasp pose: arm pointing forward and down, grasp_site at ~[0.50, -0.11, 0.15]
# (directly above cube spawn center x=0.40-0.65, z≈0.009).
# resolve_expr uses first-match ordering, so arm joints must precede the ".*" catch-all.
_PREGRASP_JOINT_POS: dict[str, float] = {
  "joint1": -0.65,  # base yaw — compensates j3 lateral drift
  "joint2": -1.20,  # shoulder
  "joint3": 1.00,  # upper-arm roll — rotates palm to face DOWN (palm_z ≈ -0.99)
  "joint4": 1.20,  # elbow
  "joint6": 0.40,  # wrist pitch
  **{k: v for k, v in _ARM_HAND_JOINT_POS_COUPLED.items() if k not in {"joint4", ".*"}},
  ".*": 0.0,  # joint5, joint7 → 0
}


def _get_cube_spec(
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


def elephant_hand_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Elephant-hand pick-and-place: cube spawns at arm reach, target is in the air.

  Physics notes (JOINT-COUPLING variant):
    * The four-bar loops are no longer closed with stiff 6-DOF welds; each loop
      is opened into a tree and the passive joints are slaved to the driven curl
      joint with scalar mjEQ_JOINT polynomial couplings (see constants.py).
      This drops 3 DOF (22→19) and removes the weld stiffness that capped the
      timestep — the couplings stay stable to dt≈0.008 in free fall.
    * Newton solver at dt=0.005, decimation=4 → 50 Hz policy with only 4 physics
      substeps per control step (vs 10 for the weld baseline, 5 for the softened
      weld).  Substep count is the real cost driver on the GPU (mujoco-warp)
      backend.  iterations kept high (cheap on GPU); njmax=384.
    * All gripper geoms have contype/conaffinity=0 (linkage links physically
      overlap), so cube contact only happens on the fingertip pads.  For
      grasping contact to work you may need to re-enable specific pad geoms
      after inspecting the URDF.
  """
  cfg = make_lift_cube_env_cfg()

  # ── Sim: joint-coupled hand, fast Newton solver at 5 ms timestep ─────────
  cfg.sim = SimulationCfg(
    nconmax=55,
    njmax=384,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=50,
      ls_iterations=20,
      impratio=30,
      cone="pyramidal",
    ),
  )
  cfg.decimation = 4

  # ── Scene entities ────────────────────────────────────────────────────────
  # coupling=True: welds replaced by mjEQ_JOINT couplings (loops → tree).
  arm_cfg = get_elephant_arm_cfg(coupling=True)
  arm_cfg.init_state.joint_pos = _PREGRASP_JOINT_POS

  cfg.scene.entities = {
    "robot": arm_cfg,
    "cube": EntityCfg(spec_fn=_get_cube_spec),
  }

  # ── Actions ──────────────────────────────────────────────────────────────
  # JointPositionActionCfg targets all actuators (".*") and scales by the
  # per-joint dict.  The dict uses the same regex keys as ELEPHANT_ACTION_SCALE.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ELEPHANT_ACTION_SCALE

  # ── Observations ─────────────────────────────────────────────────────────
  # grasp_site is on base_step_h (the hand root) after attach().
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site_h",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site_h",)
  cfg.rewards["lift"].params["reaching_std"] = (
    0.2  # matches Flexiv default; effective gradient range ~43cm
  )

  # ── Commands ─────────────────────────────────────────────────────────────
  # Cube spawn range for a ~0.9 m reach arm (same as Flexiv task).
  lift_cmd = cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  lift_cmd.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.45, 0.75),  # centered under pre-grasp pose (grasp_site x≈0.65)
    y=(-0.05, 0.15),  # centered under pre-grasp pose (grasp_site y≈0.04)
    z=(0.008, 0.010),
    yaw=(-3.14, 3.14),
  )

  # ── Reset: arm joints start at pre-grasp pose with small noise ───────────
  # position_range offsets from init_state.joint_pos (= _PREGRASP_JOINT_POS).
  # SceneEntityCfg restricts to arm joints only so hand weld angles stay exact.
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=cfg.events["reset_robot_joints"].func,
    mode="reset",
    params={
      "position_range": (-0.15, 0.15),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("joint[1-7]",)),
    },
  )

  # ── Domain randomisation ──────────────────────────────────────────────────
  # No fingertip friction DR yet (geoms are collision-disabled; add pad geom
  # names here once you re-enable their collision).
  cfg.events.pop("fingertip_friction_slide", None)
  cfg.events.pop("fingertip_friction_spin", None)
  cfg.events.pop("fingertip_friction_roll", None)

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

  # ── Contact sensor ────────────────────────────────────────────────────────
  # Terminate when the gripper base hits the ground.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "base_step_h"

  # Viewer follows the hand root body.
  cfg.viewer.body_name = "link7"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)

  return cfg
