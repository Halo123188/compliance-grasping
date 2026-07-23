"""Flexiv Rizon4S + UMI gripper Stage-1 compliance env (+ E4 ablation variants)."""

from __future__ import annotations

from typing import Literal

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import (
  get_flexiv_bare_torque_robot_cfg,
  get_flexiv_torque_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import CartesianImpedanceActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.compliance.mdp.curriculum import _LEVELS
from mjlab.tasks.compliance.reach_env_cfg import make_reach_env_cfg

Ablation = Literal["scalar_k", "no_curriculum", "shapes", "torque", "full"]


def flexiv_reach_env_cfg(
  play: bool = False, ablation: Ablation | None = None, bare: bool = False
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

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg
