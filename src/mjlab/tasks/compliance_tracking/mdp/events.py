"""Task-local event terms.

Only one, and it exists to bridge an index-space mismatch in the framework:
``dr.pd_gains`` reads ``SceneEntityCfg.actuator_ids`` as indices into the
entity's *actuator group* list, while ``SceneEntityCfg(actuator_names=...)``
resolves names to indices into the flat per-actuator list.  For a robot whose
arm and hand are separate groups those two index spaces differ, and naming the
finger joint resolves to an out-of-range group index.  Resolving the group by
its target name here keeps the config declarative and survives adding actuators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_actuator_gains_for_joint(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  joint_name: str,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_name: str = "robot",
  operation: str = "scale",
) -> None:
  """Randomize the PD gains of whichever actuator group drives ``joint_name``."""
  asset: Entity = env.scene[asset_name]
  group_ids = [
    i for i, act in enumerate(asset.actuators) if joint_name in act.target_names
  ]
  if not group_ids:
    raise ValueError(f"no actuator group on '{asset_name}' drives '{joint_name}'")
  dr.pd_gains(
    env,
    env_ids,
    kp_range=kp_range,
    kd_range=kd_range,
    asset_cfg=SceneEntityCfg(asset_name, actuator_ids=group_ids),
    operation=operation,
  )
