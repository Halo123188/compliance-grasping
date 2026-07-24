"""Direct joint-torque action for the arm (pure or gravity-compensated residual).

The policy emits one value per arm joint in ``[-1, 1]``; it is scaled to that
joint's torque limit and sent straight to the motor.  Two modes:

  * ``gravity_comp=False`` (pure torque): ``tau = a * effort_limit``.  The policy
    must learn to hold the arm against gravity itself -- any steady-state error
    shows up as sag/drift.
  * ``gravity_comp=True`` (residual torque): ``tau = a * effort_limit + qfrc_bias``
    where ``qfrc_bias`` is MuJoCo's gravity + Coriolis/centrifugal generalized
    force.  The policy only has to output the *residual* on top of a
    gravity-compensated hold, so zero action already holds station.

Both clip to the per-joint effort limit after the (optional) bias so the command
is always physical, and both sanitize non-finite policy outputs to zero (a
gravity-held hold in residual mode, a limp drop in pure mode) so a diverging
policy cannot inject NaNs into physics.  ``processed_torque`` exposes the final
commanded torque for the effort-penalty reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ArmTorqueActionCfg(BaseActionCfg):
  """Direct joint-torque control over the arm's actuators."""

  effort_limit: tuple[float, ...] = field(default_factory=tuple)
  """Per-joint torque limit (Nm), in ``actuator_names`` order.  Also the action
  scale: raw ``+-1`` maps to ``+-effort_limit``."""

  gravity_comp: bool = False
  """If True, add ``qfrc_bias`` (gravity + Coriolis) so the policy outputs a
  residual on top of a gravity-compensated hold."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> ArmTorqueAction:
    return ArmTorqueAction(self, env)


class ArmTorqueAction(BaseAction):
  """Apply policy output as joint torque (optionally gravity-compensated)."""

  cfg: ArmTorqueActionCfg
  _entity: Entity

  def __init__(self, cfg: ArmTorqueActionCfg, env: ManagerBasedRlEnv):
    # Scale raw actions by the per-joint effort limit; no offset/clip here (we
    # clip after the optional gravity term in apply_actions).
    super().__init__(cfg=cfg, env=env)
    assert len(cfg.effort_limit) == self._num_targets, (
      f"effort_limit has {len(cfg.effort_limit)} entries, expected {self._num_targets}"
    )
    self._limit = torch.tensor(cfg.effort_limit, device=self.device)  # (J,)
    # DOF velocity addresses for indexing qfrc_bias (mirrors the impedance term).
    self._dof_ids = self._entity.indexing.joint_v_adr[self._target_ids]
    self._torque = torch.zeros(self.num_envs, self._num_targets, device=self.device)

  @property
  def processed_torque(self) -> torch.Tensor:
    """Final commanded torque per joint (N, J), for the effort-penalty reward."""
    return self._torque

  @property
  def effort_limit(self) -> torch.Tensor:
    """Per-joint torque limit (J,), also the action scale."""
    return self._limit

  def process_actions(self, actions: torch.Tensor) -> None:
    # Sanitize non-finite outputs to a safe hold before scaling.
    safe = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    self._raw_actions[:] = safe
    self._processed_actions = torch.tanh(safe) * self._limit

  def apply_actions(self) -> None:
    tau = self._processed_actions
    if self.cfg.gravity_comp:
      tau = tau + self._env.sim.data.qfrc_bias[:, self._dof_ids]
    tau = torch.clamp(tau, -self._limit, self._limit)
    self._torque[:] = tau
    self._entity.set_joint_effort_target(tau, joint_ids=self._target_ids)
