"""Action terms specific to the two-finger grasp task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class WristAlignedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position control with the wrist roll solved in closed form.

  Squaring an antipodal jaw to a square prism's face is a one-line geometry
  problem, and six RL variants failed to learn it: four reward-side
  (AlignTight/Lift/Touch/Curr, all beaten by the ungated baseline) and two
  exploration-side (`free_wrist`, `wrist_rand` -- both raised joint7's action std
  from 3.5 deg to 11-20 deg and both left corr(cube yaw, joint7) at zero, +0.026
  and -0.060, while destroying the task). Meanwhile the payoff was never large:
  locking cube yaw square to a frozen wrist moves success 78.1% -> 89.1%.

  So compute it instead of learning it. `scripts/diag_wrist_sign.py` measures
  d(jaw yaw)/d(joint7) = -1.000 rad/rad exactly, so the target that squares the
  jaw RIGHT NOW is `q7 + fold(jaw_yaw - object_yaw)`, re-solved from measured
  state on every physics substep -- not an integrator, and it stays correct while
  the other arm joints move the hand around.

  The policy keeps a trim action on the same joint; give it a small `scale` (the
  0-22.5 deg band where alignment provably does not matter) so it can adjust
  without undoing the solution.

  This reads the object's pose, so it is only legitimate for the state-based
  task. The vision policy has to estimate yaw from the image and cannot use it.
  """

  wrist_joint: str = "joint7"
  object_name: str = "cube"
  pad_sites: tuple[str, str] = ("left_pad", "right_pad")
  gain: float = 1.0
  symmetry: float = math.pi / 2  # square cube + 180-deg-symmetric jaw

  def build(self, env: ManagerBasedRlEnv) -> WristAlignedJointPositionAction:
    return WristAlignedJointPositionAction(self, env)


class WristAlignedJointPositionAction(JointPositionAction):
  cfg: WristAlignedJointPositionActionCfg

  def __init__(self, cfg: WristAlignedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._scene = env.scene

    names = self.target_names
    assert cfg.wrist_joint in names, f"{cfg.wrist_joint} not in action dims {names}"
    self._wrist_col = names.index(cfg.wrist_joint)
    self._wrist_qid = list(self._entity.joint_names).index(cfg.wrist_joint)

    sites = list(self._entity.site_names)
    self._pad_ids = [sites.index(s) for s in cfg.pad_sites]

    offset = self._offset
    self._wrist_default = (
      offset[:, self._wrist_col] if isinstance(offset, torch.Tensor) else offset
    )

  def _yaw_error(self) -> torch.Tensor:
    """Signed jaw-vs-object yaw error, folded into +-symmetry/2."""
    pads = self._entity.data.site_pos_w[:, self._pad_ids]
    axis = pads[:, 1] - pads[:, 0]
    jaw = torch.atan2(axis[:, 1], axis[:, 0])

    obj: Entity = self._scene[self.cfg.object_name]
    q = obj.data.root_link_quat_w
    obj_yaw = torch.atan2(
      2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )

    sym = self.cfg.symmetry
    err = jaw - obj_yaw
    return err - sym * torch.round(err / sym)

  def apply_actions(self) -> None:
    # `JointPositionAction` sets `_offset` to the default joint pose, so the
    # wrist column of `_processed_actions` is `default_q7 + trim`. Strip the
    # default back off and replace it with the closed-form target, keeping the
    # policy's trim: d(jaw)/d(q7) = -1, so adding the error to q7 cancels it.
    target = self._processed_actions.clone()
    trim = target[:, self._wrist_col] - self._wrist_default
    q7 = self._entity.data.joint_pos[:, self._wrist_qid]
    target[:, self._wrist_col] = q7 + self.cfg.gain * self._yaw_error() + trim

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )
