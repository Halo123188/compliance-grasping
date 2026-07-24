"""Tracking-only reward (spec §3.2).

    r = w_pos · exp(−‖x_ee − x_t‖² / σ_pos²)
      + w_vel · exp(−‖ẋ_ee − ẋ_t‖² / σ_vel²)
      + w_f   · exp(−(F_finger − F_finger_target)² / σ_f²)
      − w_ar  · ‖a_t − a_{t−1}‖²

That is the complete list.  There is **no** progress term, no force penalty, no
stiffness penalty, no compliance term — and adding one would be a mistake, not an
improvement.  Every behaviour we want (yield under load, return on release,
resume the task where it stopped, hold the grasp force while the arm is being
pushed) is already encoded in ``x_t`` and ``F_finger_target`` by the teacher.  A
behavioural reward term would be a second, weaker, hand-tuned specification of
the same thing, competing with the first.

The action-rate term is the one exception, and it is not a behavioural term: it
regularizes the *actuation*, not the trajectory, and exists so the impedance
parameters do not chatter between steps.

Why this is not just behaviour cloning: the reward is evaluated on states the
student actually visits.  A BC loss on teacher rollouts is only defined on the
teacher's own state distribution, so small errors compound off-distribution with
no gradient pulling back.  On-policy RL against the same target signal samples
the student's distribution by construction.

One timing note: the env computes rewards before it computes commands, so these
terms score ``x_ee(t)`` against ``x_t(t-1)`` — a fixed one-step (10 ms) lag.  It
is uniform across every env and every step, so the MDP stays well defined; at the
teacher's nominal speed it is under 2 mm, far inside ``sigma_pos``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.arm_torque import ArmTorqueAction
from mjlab.envs.mdp.actions.cartesian_impedance import CartesianImpedanceAction
from mjlab.tasks.compliance_tracking.mdp.observations import (
  total_grasp_force,
)
from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _teacher(env: ManagerBasedRlEnv, command_name: str) -> TeacherCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, TeacherCommand)
  return term


def _impedance(env: ManagerBasedRlEnv, action_name: str) -> CartesianImpedanceAction:
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, CartesianImpedanceAction)
  return term


def position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "teacher",
  sigma: float = 0.05,
  sigma_pull: float = 0.0,
) -> torch.Tensor:
  """``exp(−‖x_ee − x_t‖² / σ²)``.

  ``sigma_pull`` (0 disables) widens the tolerance *only while a hand is
  pushing*.  The default σ is deliberately tight, which is correct for the
  free-motion task but makes any softening a pure tracking loss the policy will
  not pay for.  Relaxing σ during the pull buys the policy room to yield softly
  instead of stiffly repositioning, without loosening the precision the rest of
  the trajectory is scored on.
  """
  t = _teacher(env, command_name)
  err = torch.sum(torch.square(t.ee_pos_w() - t.x_t), dim=-1)
  if sigma_pull <= 0.0:
    return torch.exp(-err / (sigma * sigma))
  active = t.perturbation.active.float()
  s = sigma + (sigma_pull - sigma) * active
  return torch.exp(-err / (s * s))


def velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str = "teacher", sigma: float = 0.25
) -> torch.Tensor:
  """``exp(−‖ẋ_ee − ẋ_t‖² / σ²)``.

  Carries the *shape* of the yield and return transients: position tracking
  alone is satisfied by any path through the right points, and a policy that
  arrives correctly but overshoots on the way is not compliant.
  """
  t = _teacher(env, command_name)
  err = torch.sum(torch.square(t.ee_vel_w() - t.xd_t), dim=-1)
  return torch.exp(-err / (sigma * sigma))


def finger_force_tracking(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  command_name: str = "teacher",
  normal_axis: int = 0,
  sigma: float = 5.0,
) -> torch.Tensor:
  """``exp(−(F_finger − F_finger_target)² / σ²)``.

  Reads the true (un-noised) fingertip sensors: the reward may use privileged
  ground truth, only the actor's observations may not.
  """
  t = _teacher(env, command_name)
  force = total_grasp_force(env, force_sensor_names, normal_axis)
  err = torch.square(force - t.finger_force_target)
  return torch.exp(-err / (sigma * sigma))


def stiffness_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "teacher",
  action_name: str = "impedance",
  sigma: float = 0.5,
) -> torch.Tensor:
  """``exp(−mean_i (ln K_i − ln K_target_i)² / σ²)``.

  The one tracking term that is *not* about the trajectory.  It exists because
  the position/velocity terms constrain ``x_ee(t)`` alone, and "yielded because
  the arm went soft" and "yielded because the policy stiffly drove the reference
  out of the way" produce an identical ``x_ee(t)`` — so the trajectory reward
  cannot tell them apart and the policy learns the stiff one (measured
  ``K_∥/K_⊥ ≈ 1.8``, i.e. tracking by stiffening).  The teacher now emits a
  target stiffness ``K_target`` (precision-stiff when free, dropped isotropically
  to ``k_soft`` while a hand is pushing), and this term scores the commanded
  impedance against it in log space.

  This is a deliberate departure from pure tracking: ``K_target`` is a
  hand-authored impedance profile, i.e. a behavioural specification of the very
  kind the teacher–student framing set out to avoid.  Enable it, and its weight,
  knowing that — see the task README.
  """
  t = _teacher(env, command_name)
  k = _impedance(env, action_name).stiffness  # (N, 3), last commanded stiffness
  log_err = torch.square(torch.log(k) - torch.log(t.k_target)).mean(dim=-1)
  return torch.exp(-log_err / (sigma * sigma))


def torque_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "teacher",
  action_name: str = "arm_torque",
  sigma: float = 0.25,
) -> torch.Tensor:
  """``exp(−mean_j ((τ_j − τ_target_j)/limit_j)² / σ²)`` — exp-2 supervision.

  The per-joint error is normalized by that joint's torque limit before the
  Gaussian, so the big base joints do not dominate and ``σ`` is a dimensionless
  fraction-of-limit; raw-Nm errors of tens of Nm otherwise sit on the flat tail
  of the Gaussian and give the policy no gradient.

  The direct-torque policy has no commanded stiffness, so compliance cannot be
  supervised the way ``stiffness_tracking`` does.  Instead the teacher emits a
  full compliant joint-torque target ``τ_target`` (an anisotropic Cartesian
  impedance about ``x_ref``, soft along the pull and stiff perpendicular — the
  rotated stiffness a diagonal action K could not represent), and this term
  scores the policy's commanded joint torque against it.  Reproducing it makes
  the arm yield ``F_ext/k_soft`` along a *generic* push direction, which is the
  behaviour the diagonal-K action space structurally could not produce.
  """
  t = _teacher(env, command_name)
  assert t.torque_target is not None, "teacher.emit_torque_target must be True"
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, ArmTorqueAction)
  norm = (term.processed_torque - t.torque_target) / term.effort_limit.clamp(min=1e-3)
  err = torch.square(norm).mean(dim=-1)
  return torch.exp(-err / (sigma * sigma))
