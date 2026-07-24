"""Analytic oracle controller: the ceiling the student is trying to reach.

This is *not* a policy — it reads the teacher target directly, which the student
never can.  It exists for two jobs:

  * environment checks, where we need the arm actually driven so ``s`` advances
    and the perturbation schedule fires (a still arm freezes ``s`` immediately
    and is never pushed);
  * evaluation, as the upper bound on tracking.  Any gap between the student and
    this controller is the cost of proprioception-only inference; any residual
    error *this* controller still has is the cost of the impedance plant itself,
    and no policy can beat it.

It inverts the action encoding rather than bypassing it, so it is subject to the
same slew limit, leash, and stiffness range as the policy.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.cartesian_impedance import CartesianImpedanceAction
from mjlab.tasks.compliance_tracking.mdp.actions import GraspAction
from mjlab.tasks.compliance_tracking.mdp.observations import total_grasp_force
from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand
from mjlab.tasks.compliance_tracking.tracking_env_cfg import GRASP, IMPEDANCE, TEACHER


def _atanh_clamped(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
  return torch.atanh(x.clamp(-1.0 + eps, 1.0 - eps))


class OracleController:
  """Drives the commanded reference straight at the teacher target."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    stiffness: float = 600.0,
    force_sensor_names: tuple[str, ...] = (),
    normal_axis: int = 0,
    # Fast enough to close the jaws and reach F_grasp within the teacher's
    # ramp window; at 2e-3 the integrator was still converging ~1 s after
    # closure, which shows up as a grasp-force error that is the controller's,
    # not the plant's (the plant holds 15 N to ~2 N once converged).
    closure_gain: float = 8.0e-3,
  ) -> None:
    self._env = env
    self._force_sensors = force_sensor_names
    self._normal_axis = normal_axis
    self._closure_gain = closure_gain
    teacher = env.command_manager.get_term(TEACHER)
    assert isinstance(teacher, TeacherCommand)
    self._teacher = teacher

    imp = env.action_manager.get_term(IMPEDANCE)
    assert isinstance(imp, CartesianImpedanceAction)
    self._imp = imp

    self._grasp: GraspAction | None = None
    if GRASP in env.action_manager.active_terms:
      grasp = env.action_manager.get_term(GRASP)
      assert isinstance(grasp, GraspAction)
      self._grasp = grasp
      assert force_sensor_names, "grasp control needs the fingertip force sensors"
    self._closure = torch.zeros(env.num_envs, device=env.device)

    k_lo, k_hi = imp.cfg.stiffness_range
    log_frac = (torch.log(torch.tensor(stiffness)) - torch.log(torch.tensor(k_lo))) / (
      torch.log(torch.tensor(k_hi)) - torch.log(torch.tensor(k_lo))
    )
    self._k_raw = float(_atanh_clamped(2.0 * log_frac - 1.0))
    self._n_stiff = imp.action_dim - 3

  def act(self) -> torch.Tensor:
    """Return the action that steers ``x_cmd`` toward ``x_t`` this step."""
    delta = self._teacher.x_t - self._imp.commanded_reference
    frac = delta / self._imp.cfg.delta_pos_scale
    parts = [
      _atanh_clamped(frac),
      torch.full(
        (self._env.num_envs, self._n_stiff), self._k_raw, device=self._env.device
      ),
    ]
    if self._grasp is not None:
      parts.append(_atanh_clamped(2.0 * self._closure_command() - 1.0).unsqueeze(-1))
    return torch.cat(parts, dim=-1)

  def _closure_command(self) -> torch.Tensor:
    """Integral force regulation on the jaws (spec §1.5: close under position
    control until the force threshold, then hold ``F_grasp``).

    A fixed closure fraction cannot work: how much force a given jaw position
    produces depends on the object and on the finger gains, which is exactly the
    mapping the student is being asked to learn.
    """
    assert self._grasp is not None
    target = self._teacher.finger_force_target
    measured = total_grasp_force(self._env, self._force_sensors, self._normal_axis)
    self._closure = torch.where(
      target > 0.0,
      (self._closure + self._closure_gain * (target - measured)).clamp(0.0, 1.0),
      torch.zeros_like(self._closure),
    )
    return self._closure

  def reset(self) -> None:
    self._closure.zero_()
