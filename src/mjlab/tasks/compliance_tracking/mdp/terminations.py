"""Terminations for the tracking task.

Episodes run to the timeout: the reward is dense tracking, so there is nothing
to succeed *at* early, and a success termination would bias the return toward
whatever ends the episode fastest.  The only non-timeout termination is a safety
bail-out for a diverged arm, which exists so one bad env cannot poison a batch
with garbage states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ee_diverged(
  env: ManagerBasedRlEnv, command_name: str = "teacher", max_error: float = 0.6
) -> torch.Tensor:
  """Terminate if the EE is absurdly far from the teacher target or non-finite."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, TeacherCommand)
  ee = term.ee_pos_w()
  err = torch.norm(ee - term.x_t, dim=-1)
  return ~torch.isfinite(err) | (err > max_error)
