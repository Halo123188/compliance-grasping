"""Task-specific MDP terms for the Flexiv fragile-lift experiment.

The object is fragile: the total grasp normal force ``Fn`` (summed over both
fingertip FT sensors) must stay inside a feasible band ``[Fn_min, F_break]``.

- ``Fn_min = m * g / mu`` is the minimum normal force to hold the object against
  gravity with a two-finger friction grasp (total-force convention).
- ``F_break = k * Fn_min`` with ``k`` sampled per episode sets the crush limit;
  ``k`` parameterizes the feasible-band width ``(k - 1) * Fn_min``.

Only three quantities are randomized per episode: object mass ``m``, friction
``mu``, and the band-width factor ``k`` (which sets ``F_break``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GRAVITY = 9.81


# ─────────────────────────── per-episode grasp state ────────────────────────


class _GraspState:
  """Per-env buffers for the sampled grasp parameters and hold timer."""

  def __init__(self, num_envs: int, device: torch.device | str) -> None:
    z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
    self.mass = z()
    self.mu = z()
    self.k = z()
    self.fn_min = z()
    self.f_break = z()
    self.hold_steps = z()
    # Cached model indices (resolved lazily on first reset).
    self.cube_bid: int | None = None
    self.finger_gids: list[int] | None = None


def _state(env: ManagerBasedRlEnv) -> _GraspState:
  st = getattr(env, "_flexiv_grasp_state", None)
  if st is None:
    st = _GraspState(env.num_envs, env.device)
    env._flexiv_grasp_state = st  # type: ignore[attr-defined]
  return st


def _uniform(lo: float, hi: float, n: int, device: torch.device | str) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(n, device=device)


# ─────────────────────────────── reset event ────────────────────────────────


@requires_model_fields(
  "body_mass", "body_inertia", "geom_friction", recompute=RecomputeLevel.set_const
)
def randomize_grasp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  mu_range: tuple[float, float],
  k_range: tuple[float, float],
  object_name: str = "cube",
  cube_half_extent: float = 0.02,
  robot_name: str = "robot",
  finger_geom_names: tuple[str, ...] = ("left_finger_col", "right_finger_col"),
) -> None:
  """Sample (m, mu, k), apply mass+friction to the sim, and store the budget.

  Writes the object's mass (with a consistent solid-cube inertia) and the finger
  friction (which has contact priority, so it sets the effective coefficient),
  then computes ``Fn_min = m g / mu`` and ``F_break = k Fn_min``.
  """
  st = _state(env)
  if env_ids is None:
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    ids = env_ids.to(env.device, dtype=torch.long)
  n = int(ids.numel())

  cube: Entity = env.scene[object_name]
  if st.cube_bid is None:
    st.cube_bid = cube.indexing.root_body_id
    st.finger_gids = [
      env.sim.mj_model.geom(f"{robot_name}/{g}").id for g in finger_geom_names
    ]

  m = _uniform(*mass_range, n, env.device)
  mu = _uniform(*mu_range, n, env.device)
  k = _uniform(*k_range, n, env.device)

  # Object mass + physically consistent inertia for a solid cube
  # (principal moment = m * (2*half)^2 / 6 on each axis).
  env.sim.model.body_mass[ids, st.cube_bid] = m
  side = 2.0 * cube_half_extent
  inertia = (m * side * side / 6.0).unsqueeze(-1).expand(n, 3)
  env.sim.model.body_inertia[ids, st.cube_bid] = inertia

  # Effective friction coefficient = finger slide friction (fingers have
  # contact priority over the cube).
  assert st.finger_gids is not None
  for gid in st.finger_gids:
    env.sim.model.geom_friction[ids, gid, 0] = mu

  fn_min = m * GRAVITY / mu
  f_break = k * fn_min

  st.mass[ids] = m
  st.mu[ids] = mu
  st.k[ids] = k
  st.fn_min[ids] = fn_min
  st.f_break[ids] = f_break
  st.hold_steps[ids] = 0.0


# ─────────────────────────────── observations ───────────────────────────────


def ft_wrench(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  torque_sensor_names: tuple[str, ...],
) -> torch.Tensor:
  """Concatenated FT readings: [f_left, f_right, tau_left, tau_right] → (B, 12)."""
  parts = [env.scene[name].data for name in force_sensor_names]
  parts += [env.scene[name].data for name in torque_sensor_names]
  return torch.cat(parts, dim=-1)


def privileged_grasp_params(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged critic obs: per-episode [mass, mu, F_break] → (B, 3)."""
  st = _state(env)
  return torch.stack([st.mass, st.mu, st.f_break], dim=-1)


# ───────────────────────────────── helpers ──────────────────────────────────


def total_normal_force(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  normal_axis: int = 0,
) -> torch.Tensor:
  """Sum of |normal force| over both fingertip FT sensors → (B,)."""
  total = torch.zeros(env.num_envs, device=env.device)
  for name in force_sensor_names:
    total = total + env.scene[name].data[:, normal_axis].abs()
  return total


# ──────────────────────────────── rewards ───────────────────────────────────


def force_margin_penalty(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  ramp: float = 0.8,
  normal_axis: int = 0,
) -> torch.Tensor:
  """Quadratic penalty as the grasp force enters the top of the feasible band.

  Zero while ``Fn <= ramp * F_break``; grows to 1 at ``Fn = F_break``. Encourages
  the policy to keep margin below the crush limit. Return is positive; use a
  negative weight.
  """
  st = _state(env)
  fn = total_normal_force(env, force_sensor_names, normal_axis)
  band = (st.f_break * (1.0 - ramp)).clamp_min(1e-6)
  over = (fn - ramp * st.f_break).clamp_min(0.0) / band
  return over * over


# ────────────────────────────── terminations ────────────────────────────────


def object_crushed(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  normal_axis: int = 0,
) -> torch.Tensor:
  """True when total grasp normal force exceeds F_break (object breaks)."""
  st = _state(env)
  fn = total_normal_force(env, force_sensor_names, normal_axis)
  return fn > st.f_break


def lift_hold_success(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  height_tol: float = 0.02,
  hold_time_s: float = 2.0,
) -> torch.Tensor:
  """True once the object stays within ``height_tol`` of the goal for 2 s.

  Maintains a per-env consecutive-hold counter (reset when the object leaves the
  tolerance region, and at episode reset by :func:`randomize_grasp`).
  """
  st = _state(env)
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  error = torch.norm(command.target_pos - obj.data.root_link_pos_w, dim=-1)
  at_goal = error < height_tol
  st.hold_steps = torch.where(
    at_goal, st.hold_steps + 1.0, torch.zeros_like(st.hold_steps)
  )
  needed = round(hold_time_s / env.step_dt)
  return st.hold_steps >= needed
