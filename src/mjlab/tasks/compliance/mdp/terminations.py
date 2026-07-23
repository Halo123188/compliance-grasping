"""Termination terms for the Stage-1 compliance task.

Success ends the episode early (plan §2).  Timeout is the standard framework
``time_out`` on the 5 s episode length — note this is the *episode* clock, not a
human-disturbance timeout (the human never times out; plan §4.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reached_goal(
  env: ManagerBasedRlEnv,
  command_name: str = "reach",
  event_name: str = "human_disturbance",
) -> torch.Tensor:
  """True when the EE has dwelled at the goal long enough (early success end).

  Suppressed while the human is actively holding the wrist (reaching/pushing):
  otherwise the policy can end the episode the instant it touches the goal,
  before the disturbance is resolved, so the force/stiffness rewards never
  engage.  Gating success on ``~is_pushing`` forces every episode to play out
  the push before it can count as success (the stubborn human still never times
  out; plan §4.3).
  """
  success = env.command_manager.get_term(command_name).success()
  human = env.event_manager.get_term_cfg(event_name).func
  return success & ~human.is_pushing()
