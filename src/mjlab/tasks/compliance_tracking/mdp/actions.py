"""One-dimensional grasp action for the student policy (spec §2.3).

The arm is driven by the shared Cartesian impedance term; the hand gets a single
scalar.  The action commands a jaw *closure fraction*, which the position-
controlled finger converts to a squeeze force once the jaws meet the object.  We
command closure rather than force directly because that is what the real gripper
exposes, and because the student is scored on the force it *achieves*
(``F_finger`` vs the teacher's ``F_finger_target``), which keeps the mapping from
command to force something the policy has to learn rather than something the
action space hands it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GraspActionCfg(ActionTermCfg):
  """Configuration for the 1-D grasp action."""

  actuator_name: str
  """Actuator/joint name of the single driven finger DOF."""

  closed_position: float
  """Joint position (m or rad) at full closure; ``0`` is fully open."""

  open_position: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> GraspAction:
    return GraspAction(self, env)


class GraspAction(ActionTerm):
  """Maps ``tanh(a) -> [open, closed]`` finger joint position target."""

  cfg: GraspActionCfg
  _entity: Entity

  def __init__(self, cfg: GraspActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    joint_ids, _ = self._entity.find_joints_by_actuator_names([cfg.actuator_name])
    assert len(joint_ids) == 1, (
      f"GraspAction expects exactly one driven joint, got {len(joint_ids)}"
    )
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._target = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def closure(self) -> torch.Tensor:
    """Commanded closure fraction in [0, 1] (N,), for obs/metrics."""
    span = self.cfg.closed_position - self.cfg.open_position
    return ((self._target[:, 0] - self.cfg.open_position) / span).clamp(0.0, 1.0)

  def process_actions(self, actions: torch.Tensor) -> None:
    safe = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    self._raw_actions[:] = safe
    frac = 0.5 * (torch.tanh(safe) + 1.0)
    span = self.cfg.closed_position - self.cfg.open_position
    self._target[:] = self.cfg.open_position + frac * span

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._target[env_ids] = self.cfg.open_position
