"""Observation terms for the Stage-1 compliance task (plan §5.1).

The full obs vector (36) is:
  q(7), q̇(7), τ_meas(7), x_ee(3), ẋ_ee(3), (x_g − x_ee)(3), a_{t−1}(6)

``q`` / ``q̇`` / ``a_{t−1}`` reuse the framework's ``joint_pos_rel`` /
``joint_vel_rel`` / ``last_action``.  This module adds the rest.

Deliberately **not** observable (must be inferred from the τ history): ``K_h``,
the human state, ``x_hand``, ``F_ext``, the estimated ``τ̂_ext``.  Hence we feed
the *raw* total joint torque, not a model-based external-torque estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_LOG_KH_MAX = torch.log(torch.tensor(1000.0))


def joint_total_torque(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Raw measured joint torque ``τ_meas`` = actuator + external, joint space.

  Mirrors a real joint-torque sensor: it carries the external-load signal the
  policy must learn to extract.  It is *not* a clean ``τ̂_ext`` estimate (which
  would bake in model error), just the total transmitted torque.
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt = asset_cfg.joint_ids
  return asset.data.qfrc_actuator[:, jnt] + asset.data.qfrc_external[:, jnt]


def ee_pos(env: ManagerBasedRlEnv, command_name: str = "reach") -> torch.Tensor:
  """End-effector position relative to the env origin (origin-invariant)."""
  cmd = env.command_manager.get_term(command_name)
  return cmd.ee_pos_w() - env.scene.env_origins


def ee_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """End-effector body linear velocity (world frame)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  assert isinstance(body_ids, list), "ee_lin_vel requires explicit body_names"
  return asset.data.body_com_vel_w[:, body_ids[0], :3]


def goal_error(env: ManagerBasedRlEnv, command_name: str = "reach") -> torch.Tensor:
  """``x_g − x_ee`` in world frame (origin-invariant difference)."""
  cmd = env.command_manager.get_term(command_name)
  return cmd.command - cmd.ee_pos_w()


def privileged_human(
  env: ManagerBasedRlEnv,
  event_name: str = "human_disturbance",
  command_name: str = "reach",
) -> torch.Tensor:
  """Privileged human-disturbance state for the CRITIC only (asymmetric AC).

  Concatenates F_ext(3), push dir u(3), hand-rel-to-ee(3), log K_h norm(1),
  is_pushing(1), f_max(1), release_frac(1), grab(1) = 14 dims.  The critic is never deployed, so revealing the hidden
  disturbance costs nothing at test time but sharply cuts value-estimate variance
  (the actor still infers it all from the tau history).
  """
  human = env.event_manager.get_term_cfg(event_name).func
  cmd = env.command_manager.get_term(command_name)
  ee = cmd.ee_pos_w()
  f = human.external_force()  # (N, 3)
  u = human.push_dir()  # (N, 3)
  hand_rel = human._x_hand - ee  # (N, 3)
  log_kh = (torch.log(human._kh.clamp(min=1.0)) / _LOG_KH_MAX.to(f.device)).unsqueeze(
    -1
  )
  pushing = human.is_pushing().float().unsqueeze(-1)
  # e11: these are randomised per episode now, so the critic should see them too.
  f_max = (human._f_max_env / 100.0).unsqueeze(-1)
  rel_frac = human._release_frac_env.unsqueeze(-1)
  grab = (human._grab_idx.float() / max(len(human._grab_body_ids) - 1, 1)).unsqueeze(-1)
  return torch.cat([f, u, hand_rel, log_kh, pushing, f_max, rel_frac, grab], dim=-1)
