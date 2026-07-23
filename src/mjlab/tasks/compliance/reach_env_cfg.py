"""Stage-1 compliance env: free-space reach under stubborn human disturbance.

Wires the analytical Cartesian impedance controller (100 Hz policy over a 1 kHz
inner loop), the reach goal, the human-hand spring disturbance, the τ-history
observation, the §5.2 reward, the §6 displacement curriculum, and success/timeout
termination.

Robot-agnostic where it can be: the impedance controller and the τ observation
carry no reach/human assumptions, so a Stage-2 task can swap the command +
reward and reuse the rest (plan §0/§10).  The concrete robot (Flexiv+UMI, torque
actuators) is supplied by the ``config/`` layer.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import CartesianImpedanceActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.compliance import mdp
from mjlab.tasks.compliance.mdp.human import HumanDisturbance
from mjlab.tasks.velocity import mdp as base_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
EE_SITE = "grasp_site"
EE_BODY = "gripper_base"
ATTACH_BODY = "link7"


def make_reach_env_cfg() -> ManagerBasedRlEnvCfg:
  """Robot-independent Stage-1 reach-under-disturbance config.

  The ``config/`` layer must fill in ``scene.entities["robot"]`` and the viewer
  body, then it is ready to run.
  """
  # --- Actions: Cartesian impedance (policy emits Δx_ref + log K) ---
  actions: dict[str, ActionTermCfg] = {
    "impedance": CartesianImpedanceActionCfg(
      entity_name="robot",
      actuator_names=ARM_JOINTS,
      frame_name=EE_SITE,
      anchor_command_name="reach",
      delta_pos_scale=0.03,
      stiffness_range=(
        15.0,
        2000.0,
      ),  # K_min 50->15: let the arm truly slacken at large yields
      damping_ratio=0.8,
      effective_mass=2.0,
      rot_stiffness=50.0,
      nullspace_damping=2.0,
      effort_limit=(123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    )
  }

  # --- Observations (36): q, q̇, τ_meas, x_ee, ẋ_ee, x_g−x_ee, a_{t−1} ---
  arm = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
  ee = SceneEntityCfg("robot", body_names=(EE_BODY,))
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
    "ee_pos": ObservationTermCfg(func=mdp.ee_pos, params={"command_name": "reach"}),
    "ee_vel": ObservationTermCfg(func=mdp.ee_lin_vel, params={"asset_cfg": ee}),
    "goal_error": ObservationTermCfg(
      func=mdp.goal_error, params={"command_name": "reach"}
    ),
    # Use the impedance term's raw_action (sanitized) so the last-action obs is
    # finite even under the NaN-action fallback path.
    "actions": ObservationTermCfg(
      func=base_mdp.last_action, params={"action_name": "impedance"}
    ),
  }
  critic_terms = {
    **actor_terms,
    # Asymmetric critic: privileged human state (F_ext, u, hand-rel, K_h, phase).
    # Free variance reduction — the critic is never deployed.
    "privileged_human": ObservationTermCfg(
      func=mdp.privileged_human,
      params={"event_name": "human_disturbance", "command_name": "reach"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  # --- Command: reach goal ---
  commands: dict[str, CommandTermCfg] = {
    "reach": mdp.ReachCommandCfg(
      resampling_time_range=(1e9, 1e9),  # one goal per episode; success ends it
      debug_vis=True,
      robot_name="robot",
      ee_site_name=EE_SITE,
    )
  }

  # --- Events: reset + stubborn human disturbance ---
  events = {
    "reset_base": EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),  # plan §2: small home-pose perturbation
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
      },
    ),
    "human_disturbance": EventTermCfg(
      func=HumanDisturbance,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "attach_body": ATTACH_BODY,
        "f_max": 40.0,
        # Push early so it overlaps the reach (mean episode ~0.77 s): with the
        # old (1,3) s onset the policy reached the goal before the push ever
        # fired, so the force/stiffness rewards never engaged (yields the
        # observed "stiff everywhere", K∥/K⊥>1 failure).
        "push_time_range": (0.0, 0.3),
        # A1: a small unpredictable environment load (~N per axis) the arm must
        # reject with stiffness to hold the goal.  Makes orthogonal stiffness
        # valuable so arbitration (soft along the hand push, stiff elsewhere)
        # beats isotropic softness; without it K∥/K⊥→1 is reward-optimal.
        "env_load_std": 4.0,
        # e11 domain randomisation (OOD sweep found weak-hand and far-pull gaps):
        # e14: widen only toward HARDER.  e11 widened downward too (kh 30,
        # f_max 15): those pulls are too weak to move a stiff arm, so bracing
        # won and the compliant behaviour was unlearned (yield .93 -> .09).
        "kh_range": (100.0, 3000.0),
        "f_max_range": (35.0, 80.0),
        "release_frac_range": (0.4, 0.9),  # was fixed .7 -> overfit to one rule
        "release_hold_range": (0.1, 0.5),  # was fixed .2 s
        "grab_bodies": ("link5", "link6", "link7", "gripper_base"),  # was wrist only
        "d_range": (0.02, 0.05),  # curriculum level 0; updated by curriculum
      },
    ),
  }

  # --- Rewards (plan §5.2; per-step weights, scale_rewards_by_dt=False) ---
  rewards = {
    "progress": RewardTermCfg(
      func=mdp.progress, weight=10.0, params={"command_name": "reach"}
    ),
    "success": RewardTermCfg(
      func=mdp.success_bonus,
      weight=1.0,
      params={"command_name": "reach", "event_name": "human_disturbance"},
    ),
    "time": RewardTermCfg(func=mdp.time_penalty, weight=-0.005),
    "force": RewardTermCfg(
      func=mdp.force_penalty,
      weight=-0.004,
      params={"event_name": "human_disturbance", "deadband": 1.5},
    ),
    "stiffness": RewardTermCfg(
      func=mdp.stiffness_penalty, weight=-0.002, params={"action_name": "impedance"}
    ),
    # Directional: while the hand pulls, be soft *along* the push so the arm
    # actually yields (gets pulled) instead of stiff-repositioning to the hand.
    "push_softness": RewardTermCfg(  # primary: absolute K_par softness
      func=mdp.push_softness,
      weight=-0.08,
      params={"event_name": "human_disturbance", "action_name": "impedance"},
    ),
    "push_anisotropy": RewardTermCfg(  # secondary: keep K_perp firmer than K_par
      func=mdp.push_anisotropy,
      weight=-0.02,
      params={"event_name": "human_disturbance", "action_name": "impedance"},
    ),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
  }

  # --- Terminations: success (early) + episode timeout ---
  terminations = {
    "success": TerminationTermCfg(
      func=mdp.reached_goal,
      params={"command_name": "reach", "event_name": "human_disturbance"},
    ),
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
  }

  # --- Curriculum: displacement d only (global) ---
  curriculum = {
    "displacement": CurriculumTermCfg(
      func=mdp.displacement_curriculum,
      params={
        "event_name": "human_disturbance",
        "command_name": "reach",
        "window": 100,
        "promote_threshold": 0.7,
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=64,
      env_spacing=2.0,
      entities={},  # robot filled in by the config layer
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    scale_rewards_by_dt=False,  # plan §5.2 weights are per-step, not per-second
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ATTACH_BODY,
      distance=1.8,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      # Free-space reach: few contacts, but joint-limit + equality constraints
      # need headroom so nefc doesn't overflow (drops constraints silently).
      nconmax=40,
      njmax=256,
      mujoco=MujocoCfg(
        timestep=1e-3,  # 1 kHz inner loop
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=10,  # 100 Hz policy
    episode_length_s=7.0,  # e11: farther pulls need time to return  # plan §2
  )
  return cfg
