"""Diagnostics (spec §3.4).

The tracking reward is a poor progress signal on its own: it goes up when the
policy gets stiffer *and* when it becomes genuinely compliant, and those are the
two hypotheses we need to tell apart.  These metrics separate them.

The one that actually decides whether the pipeline worked is the stiffness
anisotropy, ``K_∥`` vs ``K_⊥`` relative to the (privileged) pull direction.  A
policy that tracks well by raising stiffness on every axis has learned to
*resist* the teacher's yield rather than to reproduce it — it would track ``x_t``
by dragging the human along, which is exactly the failure this task exists to
avoid.  ``K_∥/K_⊥ < 1`` while being pulled is the signature of the intended
behaviour.

Tracking error is split by regime (untouched / during a pull / just after
release / post-grasp) because the aggregate hides the interesting part: the
unperturbed error is always small and always dominates the average.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.cartesian_impedance import CartesianImpedanceAction
from mjlab.tasks.compliance_tracking.mdp.observations import total_grasp_force
from mjlab.tasks.compliance_tracking.mdp.teacher import GRASP_HOLDING, TeacherCommand

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


def _tracking_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _teacher(env, command_name)
  return torch.norm(t.ee_pos_w() - t.x_t, dim=-1)


# --- tracking error, split by regime -----------------------------------------


class tracking_error_by_phase:
  """Mean ‖x_ee − x_t‖ restricted to one regime; 0 elsewhere.

  Reported as a masked mean, so read it together with the matching
  ``*_fraction`` metric (a regime that never occurred reports 0, not NaN).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._release_left = torch.zeros(env.num_envs, device=env.device)
    self._was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    regime: str,
    command_name: str = "teacher",
    release_window_s: float = 0.5,
  ) -> torch.Tensor:
    t = _teacher(env, command_name)
    active = t.perturbation.active
    released_now = self._was_active & ~active
    self._release_left = torch.where(
      released_now,
      torch.full_like(self._release_left, release_window_s),
      (self._release_left - env.step_dt).clamp(min=0.0),
    )
    self._was_active = active.clone()

    grasped = t.grasp_state == GRASP_HOLDING
    in_release = (self._release_left > 0.0) & ~active
    if regime == "unperturbed":
      mask = ~active & ~in_release & ~grasped
    elif regime == "pull":
      mask = active
    elif regime == "release":
      mask = in_release
    elif regime == "post_grasp":
      mask = grasped
    else:
      raise ValueError(f"unknown regime: {regime}")
    return _tracking_error(env, command_name) * mask.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._release_left[env_ids] = 0.0
    self._was_active[env_ids] = False


def tracking_error(
  env: ManagerBasedRlEnv, command_name: str = "teacher"
) -> torch.Tensor:
  return _tracking_error(env, command_name)


def perturbation_active(
  env: ManagerBasedRlEnv, command_name: str = "teacher"
) -> torch.Tensor:
  return _teacher(env, command_name).perturbation.active.float()


# --- stiffness anisotropy along the pull -------------------------------------


def _k_parallel_perp(
  env: ManagerBasedRlEnv, command_name: str, action_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  t = _teacher(env, command_name)
  k = _impedance(env, action_name).stiffness  # (N, 3)
  u = t.perturbation.direction  # (N, 3), zero when idle
  k_par = (u * u * k).sum(dim=-1)
  k_perp = (k.sum(dim=-1) - k_par) / 2.0
  return t.perturbation.active, k_par, k_perp


def commanded_k_parallel(
  env: ManagerBasedRlEnv, command_name: str = "teacher", action_name: str = "impedance"
) -> torch.Tensor:
  """``K_∥`` along the pull direction while being pulled (0 otherwise)."""
  active, k_par, _ = _k_parallel_perp(env, command_name, action_name)
  return k_par * active.float()


def commanded_k_perp(
  env: ManagerBasedRlEnv, command_name: str = "teacher", action_name: str = "impedance"
) -> torch.Tensor:
  """``K_⊥`` orthogonal to the pull while being pulled (0 otherwise)."""
  active, _, k_perp = _k_parallel_perp(env, command_name, action_name)
  return k_perp * active.float()


def k_anisotropy_ratio(
  env: ManagerBasedRlEnv, command_name: str = "teacher", action_name: str = "impedance"
) -> torch.Tensor:
  """Per-step ``K_∥ / K_⊥`` while pulled.  Below 1 == softening along the push.

  .. warning::

    Episode-averaging this term gives ``E[K_par/K_perp]``, which is convex in
    ``K_par`` and so **upward-biased** by Jensen's inequality whenever the
    per-step ratio is heavy-tailed — measured at 1.80 against a true 1.09 on one
    policy, a bias larger than the effect being measured.  For the honest figure
    take ``commanded_k_parallel / commanded_k_perp`` (a ratio of two separately
    averaged metrics) and use this one only to gauge the spread.
  """
  active, k_par, k_perp = _k_parallel_perp(env, command_name, action_name)
  return (k_par / k_perp.clamp(min=1e-3)) * active.float()


# --- release transient --------------------------------------------------------


class release_overshoot:
  """Distance past ``x_ref(s)`` on the way back after the hand lets go (m).

  The teacher's return is a well-damped second-order transient; a student that
  springs back hard will overshoot it.  Measured against ``x_ref``, not ``x_t``,
  so it reads as "did the arm settle where the task wanted it" rather than
  "did it match the filter".
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._left = torch.zeros(env.num_envs, device=env.device)
    self._was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._prev_err = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "teacher",
    window_s: float = 0.6,
  ) -> torch.Tensor:
    t = _teacher(env, command_name)
    active = t.perturbation.active
    released_now = self._was_active & ~active
    self._left = torch.where(
      released_now,
      torch.full_like(self._left, window_s),
      (self._left - env.step_dt).clamp(min=0.0),
    )
    self._was_active = active.clone()

    err = torch.norm(t.ee_pos_w() - t.x_ref(), dim=-1)
    # Overshoot = still moving away from the reference after having approached
    # it: the error growing again inside the post-release window.
    growing = (err - self._prev_err).clamp(min=0.0)
    self._prev_err = err
    return growing * (self._left > 0.0).float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._left[env_ids] = 0.0
    self._was_active[env_ids] = False
    self._prev_err[env_ids] = 0.0


# --- grasp force --------------------------------------------------------------


def grasp_force_error_post_grasp(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  command_name: str = "teacher",
  normal_axis: int = 0,
) -> torch.Tensor:
  """|F_finger − F_target| while holding a grasp (0 before closure)."""
  t = _teacher(env, command_name)
  force = total_grasp_force(env, force_sensor_names, normal_axis)
  holding = (t.grasp_state == GRASP_HOLDING).float()
  return (force - t.finger_force_target).abs() * holding


def grasp_force_error_while_pushed(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  command_name: str = "teacher",
  normal_axis: int = 0,
) -> torch.Tensor:
  """|F_finger − F_target| during a *post-grasp* push — the §4 headline claim."""
  t = _teacher(env, command_name)
  force = total_grasp_force(env, force_sensor_names, normal_axis)
  mask = ((t.grasp_state == GRASP_HOLDING) & t.perturbation.active).float()
  return (force - t.finger_force_target).abs() * mask


# --- s progression ------------------------------------------------------------


def path_parameter(env: ManagerBasedRlEnv, command_name: str = "teacher"):
  return _teacher(env, command_name).s


def path_rate_fraction(env: ManagerBasedRlEnv, command_name: str = "teacher"):
  """``ṡ`` as a fraction of nominal: 1 == free-running, 0 == fully frozen."""
  t = _teacher(env, command_name)
  return t.s_dot / t.path.s_rate


def path_frozen_while_pulled(
  env: ManagerBasedRlEnv, command_name: str = "teacher", threshold: float = 0.5
) -> torch.Tensor:
  """Fraction of pulled steps in which ``s`` was substantially frozen."""
  t = _teacher(env, command_name)
  frozen = (t.s_dot / t.path.s_rate) < threshold
  return (frozen & t.perturbation.active).float()
