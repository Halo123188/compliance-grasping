"""Flexiv Rizon4S + UMI gripper instantiation of the tracking task.

Everything robot-specific lives here: the arm's torque limits, the home pose,
the cube, and the fingertip force sensors.  The task config itself names no
robot, so a different 7-DoF arm needs only this file replaced.

The home pose and the grasp workspace were solved together offline (damped
least-squares IK against the compiled model).  Both matter more than they look:
orientation is *locked* by the impedance controller to whatever pose the arm
resets in (spec §2.3 — the policy does not control it), so the home pose must
already be the top-down grasp orientation, and every grasp target in the
workspace box must be reachable *at that fixed orientation*.  The box below was
checked corner-to-corner; worst-case IK residual was 0.05 mm.
"""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import (
  FLEXIV_ARM_EFFORT_LIMIT,
  FT_FORCE_SENSOR_NAMES,
  FT_NORMAL_AXIS,
  get_flexiv_torque_robot_cfg,
  get_ft_sensor_cfgs,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.compliance_tracking.mdp import TeacherCommandCfg
from mjlab.tasks.compliance_tracking.tracking_env_cfg import (
  OBJECT_NAME,
  TEACHER,
  Stage,
  make_tracking_env_cfg,
)
from mjlab.tasks.manipulation.config.flexiv.env_cfgs import get_cube_spec

_ENTITY = "robot"
FORCE_SENSORS = tuple(f"{_ENTITY}/{n}" for n in FT_FORCE_SENSOR_NAMES)

# Retracted start: grasp site at (0.38, 0.00, 0.24), i.e. 12 cm back and 22 cm
# above the nominal grasp, with the top-down grasp orientation the impedance
# controller will lock. IK residual 0.04 mm.
_HOME_ARM_POSE: dict[str, float] = {
  "joint1": -0.225705,
  "joint2": 0.003626,
  "joint3": 0.507476,
  "joint4": 2.438249,
  "joint5": 0.002759,
  "joint6": 0.864276,
  "joint7": -1.290931,
}

_CUBE_HALF = 0.02
_CUBE_SPAWN = (0.50, 0.0, _CUBE_HALF)

# Reachable at the locked top-down orientation (verified corner-to-corner).
_GRASP_BOX_MIN = (0.42, -0.10, _CUBE_HALF)
_GRASP_BOX_MAX = (0.58, 0.10, _CUBE_HALF)

# The stock UMI jaw makes only ~7.5 N of squeeze at full close (residual spring
# travel x 200 N/m per finger), which is below the F_grasp the teacher scripts,
# so the finger-force tracking reward would be unreachable by construction.
# Raising the finger gains puts F_grasp comfortably inside the achievable range.
_GRIPPER_STIFFNESS = 1200.0  # N/m (stock 200)
_GRIPPER_DAMPING = 60.0  # N*s/m (stock 20)
_GRIPPER_EFFORT_LIMIT = 40.0  # N (stock 20)
_FINGER_CLOSED = 0.041  # m, full closure


def _strengthen_gripper(robot_cfg: EntityCfg) -> None:
  """Raise the finger actuator gains on a task-local copy of the robot cfg."""
  assert robot_cfg.articulation is not None
  actuators = []
  for act in robot_cfg.articulation.actuators:
    if "left_finger_joint" in act.target_names_expr:
      act = dataclasses.replace(
        act,
        stiffness=_GRIPPER_STIFFNESS,
        damping=_GRIPPER_DAMPING,
        effort_limit=_GRIPPER_EFFORT_LIMIT,
      )
    actuators.append(act)
  robot_cfg.articulation = dataclasses.replace(
    robot_cfg.articulation, actuators=tuple(actuators)
  )


def flexiv_tracking_env_cfg(
  stage: Stage = "A",
  play: bool = False,
  supervise_stiffness: bool = False,
  actor_sees_pull_dir: bool = False,
  torque_action: bool = False,
  aux_force: bool = False,
  smooth_weight: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  with_object = stage in ("B", "C")
  cfg = make_tracking_env_cfg(
    stage=stage,
    force_sensor_names=FORCE_SENSORS if with_object else (),
    normal_axis=FT_NORMAL_AXIS,
    arm_effort_limit=FLEXIV_ARM_EFFORT_LIMIT,
    finger_closed_position=_FINGER_CLOSED,
    supervise_stiffness=supervise_stiffness,
    actor_sees_pull_dir=actor_sees_pull_dir,
    torque_action=torque_action,
    aux_force=aux_force,
    smooth_weight=smooth_weight,
  )

  # ── Robot ──────────────────────────────────────────────────────────────────
  # Replace init_state / articulation with fresh objects rather than mutating:
  # get_flexiv_torque_robot_cfg() returns module-level shared defaults, and an
  # in-place edit would leak into every other Flexiv task.
  robot_cfg = get_flexiv_torque_robot_cfg()
  home = dict(_HOME_ARM_POSE)
  home["left_finger_joint"] = 0.0
  home["right_finger_joint"] = 0.0
  robot_cfg.init_state = dataclasses.replace(robot_cfg.init_state, joint_pos=home)
  if with_object:
    _strengthen_gripper(robot_cfg)

  entities: dict[str, EntityCfg] = {"robot": robot_cfg}
  if with_object:
    entities[OBJECT_NAME] = EntityCfg(
      spec_fn=lambda: get_cube_spec(cube_size=_CUBE_HALF, mass=0.15),
      init_state=EntityCfg.InitialStateCfg(pos=_CUBE_SPAWN),
    )
  cfg.scene.entities = entities

  if with_object:
    existing = tuple(cfg.scene.sensors or ())
    cfg.scene.sensors = existing + get_ft_sensor_cfgs(_ENTITY)

  # ── Teacher geometry ───────────────────────────────────────────────────────
  teacher = cfg.commands[TEACHER]
  assert isinstance(teacher, TeacherCommandCfg)
  teacher.grasp_box_min = _GRASP_BOX_MIN
  teacher.grasp_box_max = _GRASP_BOX_MAX

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
