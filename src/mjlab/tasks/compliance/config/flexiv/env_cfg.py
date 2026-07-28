"""Flexiv Rizon4S + UMI gripper Stage-1 compliance env (+ E4 ablation variants)."""

from __future__ import annotations

from typing import Literal

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import (
  get_flexiv_bare_torque_robot_cfg,
  get_flexiv_torque_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import ArmTorqueActionCfg, CartesianImpedanceActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.compliance import mdp
from mjlab.tasks.compliance.mdp.curriculum import _LEVELS
from mjlab.tasks.compliance.reach_env_cfg import (
  ARM_JOINTS,
  make_reach_env_cfg,
)
from mjlab.tasks.velocity import mdp as base_mdp

Ablation = Literal["scalar_k", "no_curriculum", "shapes", "torque", "full"]
TorqueMode = Literal["pure", "residual"]

# Per-joint torque limits (Rizon 4S datasheet): J1..J7 in Nm.
_ARM_EFFORT_LIMIT = (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0)


def _apply_torque_mode(
  cfg: ManagerBasedRlEnvCfg,
  gravity_comp: bool,
  delta_frac: float | None = None,
  vel_penalty: float = 0.0,
  acc_penalty: float = 0.0,
  effort_weight: float = -0.005,
) -> None:
  """Swap the impedance action for direct joint-torque control and rebuild the
  compliance rewards accordingly (drop the stiffness-based terms, add a torque
  effort penalty, make the force penalty the primary compliance driver).

  ``gravity_comp=False`` is pure torque (the policy holds against gravity
  itself); ``True`` is residual torque on a gravity-compensated hold.
  ``delta_frac`` (if set) switches to incremental (rate) control: the policy
  outputs a bounded torque increment per step, capping the per-step change and
  killing spikes / reset-time flailing.
  """
  cfg.actions = {
    "torque": ArmTorqueActionCfg(
      entity_name="robot",
      actuator_names=ARM_JOINTS,
      effort_limit=_ARM_EFFORT_LIMIT,
      gravity_comp=gravity_comp,
      delta_frac=delta_frac,
    )
  }
  # The last-action observation must key on the new action term.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["actions"].params["action_name"] = "torque"

  # Rewards: stiffness / push_softness / push_anisotropy read impedance.stiffness
  # and have no meaning under torque control -- drop them.  Compliance is now
  # carried by the force penalty alone (yielding => small interaction force), so
  # double its weight; add a torque effort penalty as the bang-bang regulariser.
  # A rollout of the first curriculum policy saturated torque 14% of the time, so
  # the effort penalty is stronger here (-0.005) than the first pass (-0.002).
  for term in ("stiffness", "push_softness", "push_anisotropy"):
    cfg.rewards.pop(term, None)
  cfg.rewards["force"].weight = -0.008
  cfg.rewards["torque_effort"] = RewardTermCfg(
    func=mdp.torque_effort_penalty,
    weight=effort_weight,
    params={"action_name": "torque"},
  )
  if vel_penalty:
    # Penalise joint velocity to suppress the reset-time "whip to the goal": the
    # torque policies reach ~100x e15's early joint speed (measured), and that
    # fast motion -- not torque rate -- is the visible flailing.  A squared
    # penalty bites the whip hard while barely touching the calm settled / yield
    # motion, pushing the policy toward e15's critically-damped profile.
    cfg.rewards["joint_vel"] = RewardTermCfg(
      func=base_mdp.joint_vel_l2,
      weight=vel_penalty,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
  if acc_penalty:
    # Clamped joint-acceleration penalty: the reset "whip to the goal" is a burst
    # of high acceleration, whereas steady reach / yield / return is near-zero
    # acceleration, so this is more surgical than the velocity penalty (which
    # taxes the yield/return motion too).  Clamped to bound the reset numerical
    # spike (raw joint_acc_l2 is ~700000x concentrated at the reset step).
    cfg.rewards["joint_acc"] = RewardTermCfg(
      func=mdp.joint_acc_penalty,
      weight=acc_penalty,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )


def flexiv_reach_env_cfg(
  play: bool = False,
  ablation: Ablation | None = None,
  bare: bool = False,
  torque: TorqueMode | None = None,
  warmup: bool = False,
  torque_delta_frac: float | None = None,
  torque_vel_penalty: float = 0.0,
  torque_acc_penalty: float = 0.0,
  torque_effort_weight: float = -0.005,
) -> ManagerBasedRlEnvCfg:
  cfg = make_reach_env_cfg()
  if bare:
    # Bare Rizon 4S, no UMI gripper: the EE is the flange tool point and the
    # human can only grab arm links (gripper_base no longer exists).  The EE
    # velocity observation keys on the EE body, which is now link7 (the site
    # sits on it) instead of the missing gripper_base.
    cfg.scene.entities = {"robot": get_flexiv_bare_torque_robot_cfg()}
    cfg.events["human_disturbance"].params["grab_bodies"] = ("link5", "link6", "link7")
    ee = SceneEntityCfg("robot", body_names=("link7",))
    for group in ("actor", "critic"):
      cfg.observations[group].terms["ee_vel"].params["asset_cfg"] = ee
  else:
    cfg.scene.entities = {"robot": get_flexiv_torque_robot_cfg()}

  if ablation == "scalar_k":
    # A1: isotropic (scalar) stiffness — is per-axis anisotropy the benefit?
    action = cfg.actions["impedance"]
    assert isinstance(action, CartesianImpedanceActionCfg)
    action.isotropic_stiffness = True
  elif ablation == "no_curriculum":
    # A3: skip the curriculum, train directly on the final level-3 distribution.
    cfg.curriculum = {}
    cfg.events["human_disturbance"].params["d_range"] = _LEVELS[3]

  elif ablation == "shapes":
    # Item 6: the person does not only "pull to a point and hold" — also shake
    # and drag.  Release stays yield-conditioned in every mode (no timeout).
    cfg.events["human_disturbance"].params["shape_modes"] = ("hold", "shake", "drag")
  elif ablation == "torque":
    # Item 5: a real grip acts off the body CoM, so it transmits a moment r x F.
    # Fighting that with a *fixed* orientation lock is hopeless, so the policy
    # also gets to set K_rot (action dim 6 -> 7).
    cfg.events["human_disturbance"].params["grip_offset"] = 0.06
    action = cfg.actions["impedance"]
    assert isinstance(action, CartesianImpedanceActionCfg)
    action.learn_rot_stiffness = True

  elif ablation == "full":
    # e14: items 6 and 5 together on the corrected randomisation.
    cfg.events["human_disturbance"].params["shape_modes"] = ("hold", "shake", "drag")
    cfg.events["human_disturbance"].params["grip_offset"] = 0.06
    action = cfg.actions["impedance"]
    assert isinstance(action, CartesianImpedanceActionCfg)
    action.learn_rot_stiffness = True

  if torque is not None:
    # Direct joint-torque control on the e15 "full" disturbance (shapes + grip
    # moment); the policy outputs torque instead of modulating an impedance law.
    # learn_rot_stiffness above is an impedance concept and is discarded here.
    cfg.events["human_disturbance"].params["shape_modes"] = ("hold", "shake", "drag")
    cfg.events["human_disturbance"].params["grip_offset"] = 0.06
    _apply_torque_mode(
      cfg,
      gravity_comp=(torque == "residual"),
      delta_frac=torque_delta_frac,
      vel_penalty=torque_vel_penalty,
      acc_penalty=torque_acc_penalty,
      effort_weight=torque_effort_weight,
    )
    if warmup:
      # Curriculum phase A: disturbance OFF so the policy first learns the base
      # reach-and-hold under direct torque (the bottleneck for the torque
      # variants).  Phase B resumes from this checkpoint with the push on.  The
      # push simply never fires (onset time past any episode), so success is
      # never gated and progress is never zeroed -- clean reach learning.
      p = cfg.events["human_disturbance"].params
      p["push_time_range"] = (1e9, 1e9)
      p["second_push_prob"] = 0.0

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg
