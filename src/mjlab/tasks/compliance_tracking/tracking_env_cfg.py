"""Robot-agnostic env config for teacher–student compliant grasping.

The three stages of spec §3.3 are the same environment with pieces switched on,
not three environments.  That matters: the point of staging is to add one new
failure mode at a time while everything already verified stays byte-identical.

  Stage A  pre-grasp only.  Reach + perturbations + s-freezing + admittance
           target.  No fingers, no object, no weld.  Verifies the compliance
           behaviour in isolation, where nothing can be blamed on contact.
  Stage B  + scripted grasp, the object, the weld, and post-grasp pushes.
           Verifies that the grasp force is held constant while the arm yields.
  Stage C  + domain randomization for transfer.  Contact randomization is
           deliberately absent and is not an oversight: perturbations enter as a
           wrench and the object is welded, so there is no contact channel left
           for it to act on.

The concrete robot is supplied by the ``config/`` layer; nothing here names one.
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import CartesianImpedanceActionCfg
from mjlab.envs.mdp.actions.arm_torque import ArmTorqueActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.compliance_tracking import mdp
from mjlab.tasks.velocity import mdp as base_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

Stage = Literal["A", "B", "C"]

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
FINGER_JOINT = "left_finger_joint"
EE_SITE = "grasp_site"
EE_BODY = "gripper_base"
ATTACH_BODY = "link7"
OBJECT_NAME = "cube"

TEACHER = "teacher"
IMPEDANCE = "impedance"
TORQUE = "arm_torque"
GRASP = "grasp"


def make_tracking_env_cfg(
  *,
  stage: Stage = "A",
  force_sensor_names: tuple[str, ...] = (),
  normal_axis: int = 0,
  arm_effort_limit: tuple[float, ...] | None = None,
  finger_closed_position: float = 0.041,
  supervise_stiffness: bool = False,
  stiffness_weight: float = 1.0,
  pull_sigma: float = 0.10,
  actor_sees_pull_dir: bool = False,
  torque_action: bool = False,
  torque_weight: float = 1.0,
  aux_force: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the tracking env for one curriculum stage (see module docstring).

  ``supervise_stiffness`` turns on the teacher's target-impedance signal and the
  matching reward term (``mdp.stiffness_tracking``).  This is the escape hatch
  from the pure-tracking ceiling: without it the trajectory reward cannot pull
  the policy off "track by stiffening", but it costs the framework's purity — the
  target impedance profile is hand-authored (anisotropic: soft along the pull,
  stiff perpendicular).  It also relaxes the position tolerance *during a pull*
  to ``pull_sigma`` so softening is not a pure tracking loss the policy refuses
  to pay.  The freeze band is left at its baseline (4 cm) so this is a clean
  single-variable change from the un-supervised task — the earlier round moved
  the band at the same time and could not attribute the regression."""
  with_object = stage in ("B", "C")
  if with_object and not force_sensor_names:
    raise ValueError("Stage B/C needs fingertip force sensors for the grasp reward")

  arm = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
  finger = SceneEntityCfg("robot", joint_names=(FINGER_JOINT,))

  # ── Actions (§2.3) ──────────────────────────────────────────────────────────
  # The policy steers the impedance *reference* rather than offsetting a fixed
  # anchor.  With a fixed anchor the equilibrium can never be more than a few cm
  # from the goal, so the reachable target set could not express "hold back
  # here while the hand pulls" or "dwell during closure" — precisely the parts
  # of x_t the student is meant to reproduce.  Note also what the anchor is
  # *not*: feeding x_t in as the anchor would hand the policy the answer and
  # collapse the problem to "output max stiffness".
  actions: dict[str, ActionTermCfg]
  if torque_action:
    # exp-2: the policy emits joint torque directly (residual over gravity comp)
    # instead of Δx_ref + K. Compliance is supervised by matching the teacher's
    # compliant torque target (mdp.torque_tracking); the impedance controller and
    # its diagonal, unrotatable stiffness are gone entirely.
    actions = {
      TORQUE: ArmTorqueActionCfg(
        entity_name="robot",
        actuator_names=ARM_JOINTS,
        effort_limit=tuple(arm_effort_limit) if arm_effort_limit else (),
        gravity_comp=True,
      )
    }
  else:
    actions = {
      IMPEDANCE: CartesianImpedanceActionCfg(
        entity_name="robot",
        actuator_names=ARM_JOINTS,
        frame_name=EE_SITE,
        reference_mode="integrate",
        # Per-step increment: 1 cm at 100 Hz == a 1 m/s reference slew limit, ~7x
        # the nominal path speed and above the peak of the admittance return
        # transient, without making the action so high-gain that useful commands
        # live in the first 1% of its range.
        delta_pos_scale=0.01,
        leash_radius=0.10,
        stiffness_range=(50.0, 2000.0),
        damping_ratio=0.8,
        effective_mass=2.0,
        rot_stiffness=50.0,
        nullspace_damping=2.0,
        effort_limit=arm_effort_limit,
      )
    }
  if with_object:
    actions[GRASP] = mdp.GraspActionCfg(
      entity_name="robot",
      actuator_name=FINGER_JOINT,
      closed_position=finger_closed_position,
    )

  # ── Observations (§2.1, §2.2) ───────────────────────────────────────────────
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"asset_cfg": arm},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": arm},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_torque": ObservationTermCfg(
      func=mdp.joint_total_torque,
      params={"asset_cfg": arm},
      noise=Gnoise(mean=0.0, std=0.5),
    ),
    "ee_pos": ObservationTermCfg(func=mdp.ee_pos, params={"command_name": TEACHER}),
    "ee_vel": ObservationTermCfg(func=mdp.ee_lin_vel, params={"command_name": TEACHER}),
    "goal_error": ObservationTermCfg(
      func=mdp.goal_error, params={"command_name": TEACHER}
    ),
    "actions": ObservationTermCfg(func=mdp.sanitized_last_action),
  }
  if with_object:
    actor_terms["finger_state"] = ObservationTermCfg(
      func=mdp.finger_state,
      params={
        "asset_cfg": finger,
        "force_sensor_names": force_sensor_names,
        "normal_axis": normal_axis,
      },
      noise=Gnoise(mean=0.0, std=0.25),
    )
    actor_terms["grasp_flag"] = ObservationTermCfg(
      func=mdp.grasp_phase_flag, params={"command_name": TEACHER}
    )

  if actor_sees_pull_dir:
    # DIAGNOSTIC: hand the actor the privileged pull direction to test whether
    # observability (not the diagonal-K action space) is what blocks compliance.
    actor_terms["pull_direction"] = ObservationTermCfg(
      func=mdp.pull_direction, params={"command_name": TEACHER}
    )

  critic_terms = {
    **actor_terms,
    "privileged_teacher": ObservationTermCfg(
      func=mdp.privileged_teacher, params={"command_name": TEACHER}
    ),
    "privileged_perturbation": ObservationTermCfg(
      func=mdp.privileged_perturbation, params={"command_name": TEACHER}
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }
  if aux_force:
    # Auxiliary force-estimation LABEL: a standalone group no model set consumes,
    # so it flows to the rollout storage untouched and AuxPPO reads it as the
    # regression target for the actor's force head (see rl_aux). Never corrupted
    # (it is ground truth) and never fed to the actor (proprioception only).
    observations["ext_force"] = ObservationGroupCfg(
      {
        "force": ObservationTermCfg(
          func=mdp.ext_force, params={"command_name": TEACHER}
        )
      },
      enable_corruption=False,
    )

  # ── Command: the analytic teacher (§1) ──────────────────────────────────────
  commands: dict[str, CommandTermCfg] = {
    TEACHER: mdp.TeacherCommandCfg(
      resampling_time_range=(1e9, 1e9),  # one path per episode; reset resamples
      debug_vis=True,
      robot_name="robot",
      ee_site_name=EE_SITE,
      attach_body_name=ATTACH_BODY,
      object_name=OBJECT_NAME if with_object else None,
      weld_enabled=with_object,
      supervise_stiffness=supervise_stiffness,
      emit_torque_target=torque_action,
      arm_joint_names=ARM_JOINTS if torque_action else (),
      # Freeze band left at baseline (0.015 / 0.04) so stiffness supervision is
      # the only variable changed from the un-supervised task.
      perturbation=mdp.PerturbationCfg(),
    )
  }

  # ── Events ──────────────────────────────────────────────────────────────────
  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
      },
    ),
  }
  if with_object:
    events["reset_object"] = EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.06, 0.06), "y": (-0.08, 0.08)},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(OBJECT_NAME),
      },
    )
  if stage == "C":
    events.update(_stage_c_randomization())

  # ── Rewards: tracking only (§3.2) ───────────────────────────────────────────
  rewards = {
    "position_tracking": RewardTermCfg(
      func=mdp.position_tracking,
      weight=2.0,
      params={
        "command_name": TEACHER,
        "sigma": 0.05,
        # Pull-time tolerance only when supervising stiffness; 0 keeps the
        # tight single-sigma behaviour for the pure-tracking task.
        "sigma_pull": pull_sigma if supervise_stiffness else 0.0,
      },
    ),
    "velocity_tracking": RewardTermCfg(
      func=mdp.velocity_tracking,
      weight=0.5,
      params={"command_name": TEACHER, "sigma": 0.25},
    ),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
  }
  if with_object:
    rewards["finger_force_tracking"] = RewardTermCfg(
      func=mdp.finger_force_tracking,
      weight=1.0,
      params={
        "command_name": TEACHER,
        "force_sensor_names": force_sensor_names,
        "normal_axis": normal_axis,
        "sigma": 5.0,
      },
    )
  if supervise_stiffness:
    rewards["stiffness_tracking"] = RewardTermCfg(
      func=mdp.stiffness_tracking,
      weight=stiffness_weight,
      params={"command_name": TEACHER, "action_name": IMPEDANCE, "sigma": 0.5},
    )
  if torque_action:
    rewards["torque_tracking"] = RewardTermCfg(
      func=mdp.torque_tracking,
      weight=torque_weight,
      params={"command_name": TEACHER, "action_name": TORQUE, "sigma": 0.25},
    )

  # ── Terminations ────────────────────────────────────────────────────────────
  terminations = {
    "diverged": TerminationTermCfg(
      func=mdp.ee_diverged, params={"command_name": TEACHER, "max_error": 0.6}
    ),
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
  }

  metrics = _diagnostics(
    with_object, force_sensor_names, normal_axis, impedance=not torque_action
  )

  entities = {}  # robot (+ object) filled in by the config layer
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=64,
      env_spacing=2.5,
      entities=entities,
      spec_fn=(
        (lambda spec: mdp.add_grasp_weld(spec, "robot", OBJECT_NAME))
        if with_object
        else None
      ),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    scale_rewards_by_dt=False,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ATTACH_BODY,
      distance=1.6,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=400,
      mujoco=MujocoCfg(timestep=1e-3, iterations=10, ls_iterations=20),
    ),
    decimation=10,  # 100 Hz policy over a 1 kHz impedance loop
    episode_length_s=8.0,
  )
  return cfg


def _diagnostics(
  with_object: bool,
  force_sensor_names: tuple[str, ...],
  normal_axis: int,
  impedance: bool = True,
) -> dict[str, MetricsTermCfg]:
  """Spec §3.4 diagnostics.

  ``impedance=False`` (direct-torque variant) drops the commanded-stiffness
  metrics, which read a ``CartesianImpedanceAction`` that is not present; the
  physical ``effective_k_pull`` stands in for them.
  """
  m: dict[str, MetricsTermCfg] = {
    "track_err": MetricsTermCfg(
      func=mdp.tracking_error, params={"command_name": TEACHER}
    ),
    "effective_k_pull": MetricsTermCfg(
      func=mdp.effective_k_pull, params={"command_name": TEACHER}
    ),
    "track_err_unperturbed": MetricsTermCfg(
      func=mdp.tracking_error_by_phase,
      params={"regime": "unperturbed", "command_name": TEACHER},
    ),
    "track_err_pull": MetricsTermCfg(
      func=mdp.tracking_error_by_phase,
      params={"regime": "pull", "command_name": TEACHER},
    ),
    "track_err_release": MetricsTermCfg(
      func=mdp.tracking_error_by_phase,
      params={"regime": "release", "command_name": TEACHER},
    ),
    "perturbed_fraction": MetricsTermCfg(
      func=mdp.perturbation_active, params={"command_name": TEACHER}
    ),
    "release_overshoot": MetricsTermCfg(
      func=mdp.release_overshoot, params={"command_name": TEACHER}, reduce="max"
    ),
    "path_parameter_final": MetricsTermCfg(
      func=mdp.path_parameter, params={"command_name": TEACHER}, reduce="last"
    ),
    "path_rate_fraction": MetricsTermCfg(
      func=mdp.path_rate_fraction, params={"command_name": TEACHER}
    ),
    "path_frozen_while_pulled": MetricsTermCfg(
      func=mdp.path_frozen_while_pulled, params={"command_name": TEACHER}
    ),
  }
  if impedance:
    m["k_parallel"] = MetricsTermCfg(
      func=mdp.commanded_k_parallel,
      params={"command_name": TEACHER, "action_name": IMPEDANCE},
    )
    m["k_perp"] = MetricsTermCfg(
      func=mdp.commanded_k_perp,
      params={"command_name": TEACHER, "action_name": IMPEDANCE},
    )
    m["k_anisotropy"] = MetricsTermCfg(
      func=mdp.k_anisotropy_ratio,
      params={"command_name": TEACHER, "action_name": IMPEDANCE},
    )
  if with_object:
    m["track_err_post_grasp"] = MetricsTermCfg(
      func=mdp.tracking_error_by_phase,
      params={"regime": "post_grasp", "command_name": TEACHER},
    )
    m["grasp_force_err"] = MetricsTermCfg(
      func=mdp.grasp_force_error_post_grasp,
      params={
        "command_name": TEACHER,
        "force_sensor_names": force_sensor_names,
        "normal_axis": normal_axis,
      },
    )
    m["grasp_force_err_pushed"] = MetricsTermCfg(
      func=mdp.grasp_force_error_while_pushed,
      params={
        "command_name": TEACHER,
        "force_sensor_names": force_sensor_names,
        "normal_axis": normal_axis,
      },
    )
  return m


def _stage_c_randomization() -> dict[str, EventTermCfg]:
  """Spec §3.3 Stage C: transfer randomization on the actuation path only.

  Actuator dynamics, transmission friction, and the sensed-state noise the actor
  has to infer external force through.  No contact randomization — by
  construction there is nothing for it to randomize (wrench-input perturbations,
  welded object).
  """
  arm = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
  return {
    # Transmission friction: an unmodelled joint-level force that shows up in
    # tau_meas exactly like a small external load, so this is the randomization
    # that stops the policy reading "torque offset" as "someone is pushing me".
    "dr_joint_friction": EventTermCfg(
      func=dr.joint_friction,
      mode="reset",
      params={"asset_cfg": arm, "operation": "abs", "ranges": (0.0, 0.4)},
    ),
    "dr_joint_damping": EventTermCfg(
      func=dr.joint_damping,
      mode="reset",
      params={"asset_cfg": arm, "operation": "scale", "ranges": (0.7, 1.4)},
    ),
    # Rotor inertia: the dominant term in the achievable torque bandwidth.
    "dr_joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode="reset",
      params={"asset_cfg": arm, "operation": "scale", "ranges": (0.8, 1.3)},
    ),
    "dr_encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="reset",
      params={"asset_cfg": arm, "bias_range": (-0.005, 0.005)},
    ),
    # Finger actuator gains == grasp-force-per-closure-command, the mapping the
    # student has to invert to hit F_grasp.
    "dr_finger_gains": EventTermCfg(
      func=mdp.randomize_actuator_gains_for_joint,
      mode="reset",
      params={
        "joint_name": FINGER_JOINT,
        "kp_range": (0.7, 1.4),
        "kd_range": (0.7, 1.4),
        "operation": "scale",
      },
    ),
    # Virtual-mass / payload mismatch, modelled as a point mass at the tool COM
    # (which is what dr.body_mass represents: mass without added inertia).
    "dr_payload": EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(EE_BODY,)),
        "operation": "add",
        "ranges": (0.0, 1.5),
      },
    ),
  }
