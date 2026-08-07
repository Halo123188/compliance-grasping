"""Distillation with the DAgger mixing schedule RSL-RL leaves out.

RSL-RL's ``Distillation`` always executes the student's action. That is DAgger's
beta = 0 case, and it has a cold start: at iteration 0 the student is random, so
the states it visits are off-task, and the teacher's labels there are labels for
a situation the task never gets into. If the student cannot recover -- and on a
tabletop grasp it cannot, because a knocked-away object stays knocked away --
every subsequent sample is drawn from that same degenerate distribution and the
loss never descends.

Ross et al.'s DAgger executes a mixture instead: with probability ``beta`` take
the expert's action, otherwise the student's, annealing ``beta`` from 1 to 0.
Early data then comes from the teacher's own state distribution, which is where
the teacher's labels are worth something, and control transfers to the student
only as it becomes able to hold that distribution itself.

The labels are the teacher's either way. Mixing changes which states get
labelled, not what the label is.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict


class DaggerDistillation(Distillation):
  """Distillation whose rollout mixes teacher and student actions.

  Args:
    beta_start: Probability of executing the teacher's action at iteration 0.
      ``1.0`` means the first rollouts are pure teacher demonstrations.
    beta_end: Probability at and after ``beta_decay_iters``.
    beta_decay_iters: Iterations over which ``beta`` decays linearly. ``0``
      disables mixing entirely, reproducing the stock behaviour.

  The mixture is drawn per environment per step, so a single rollout batch
  contains both teacher-driven and student-driven trajectories rather than
  splitting the environments into two fixed populations.
  """

  def __init__(
    self,
    *args,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    beta_decay_iters: int = 0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.beta_decay_iters = beta_decay_iters

  @property
  def beta(self) -> float:
    """Current probability of executing the teacher's action."""
    if self.beta_decay_iters <= 0:
      return 0.0
    t = min(1.0, self.num_updates / self.beta_decay_iters)
    return self.beta_start + t * (self.beta_end - self.beta_start)

  def act(self, obs: TensorDict) -> torch.Tensor:
    actions = super().act(obs)
    beta = self.beta
    if beta <= 0.0:
      return actions
    # `super().act` has already stored the student's action as the transition's
    # action and the teacher's as the regression target. Only what gets EXECUTED
    # changes here; the stored target is untouched, and the stored action is not
    # read by the distillation loss at all.
    take_teacher = torch.rand(actions.shape[0], device=actions.device) < beta
    teacher_actions = self.transition.privileged_actions
    assert teacher_actions is not None
    return torch.where(take_teacher.unsqueeze(-1), teacher_actions, actions)
