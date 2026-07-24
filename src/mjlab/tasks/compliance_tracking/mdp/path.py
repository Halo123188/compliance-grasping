"""Nominal reach-and-grasp reference path ``x_ref(s)`` (spec §1.1).

The path is indexed by a **path parameter** ``s ∈ [0, 1]``, never by wall-clock
time.  That distinction is the whole point: freezing ``s`` (§1.2) is what encodes
"stop making task progress while you are being pushed, then carry on from where
you left off".  If the reference were a function of time, a push would cost the
policy a chunk of trajectory it could never recover, and "resume" would have to
be invented by the policy instead of being handed to it by the target signal.

Phases are laid out along ``s`` with fixed fractional widths:

    ┌──────────── approach ────────────┬─ descend ─┬─ close ─┬──── lift ────┐
    0                                s_app      s_pre     s_close          1
    x_start                        x_above    x_grasp   x_grasp        x_lift

* ``approach``  travel from the start pose to a standoff above the object,
* ``descend``   straight-line descent onto the grasp pose,
* ``close``     a dwell at the grasp pose while the scripted fingers close,
* ``lift``      the optional post-grasp lift/hold.

Each phase gets a nominal *duration*; ``s`` is normalised time along that
schedule, so ``ds/dt`` is piecewise constant and the Cartesian speed of each
segment is (segment length)/(segment duration).  Waypoints are per-env tensors,
so every env can have its own object position without changing this code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Phase codes, also used by the observation/metric terms.
PHASE_APPROACH = 0
PHASE_DESCEND = 1
PHASE_CLOSE = 2
PHASE_LIFT = 3
NUM_PHASES = 4


@dataclass(frozen=True)
class PathSchedule:
  """Nominal per-phase durations (seconds) for the reach-and-grasp path."""

  approach: float = 1.2
  descend: float = 0.8
  close: float = 0.6
  lift: float = 0.8

  @property
  def total(self) -> float:
    return self.approach + self.descend + self.close + self.lift

  @property
  def breakpoints(self) -> tuple[float, float, float]:
    """``(s_app, s_pre, s_close)`` — the phase boundaries in ``s``."""
    t = self.total
    s_app = self.approach / t
    s_pre = (self.approach + self.descend) / t
    s_close = (self.approach + self.descend + self.close) / t
    return s_app, s_pre, s_close


class ReferencePath:
  """Per-env piecewise-linear ``x_ref(s)`` over the four reach-and-grasp phases.

  Waypoints are written by :meth:`set_waypoints` at episode start; ``s`` is
  advanced by the caller (the teacher), which owns the freezing law.
  """

  def __init__(
    self, num_envs: int, device: torch.device | str, schedule: PathSchedule
  ) -> None:
    self.schedule = schedule
    self._device = device
    z = lambda: torch.zeros(num_envs, 3, device=device)  # noqa: E731
    self.x_start = z()
    self.x_above = z()
    self.x_grasp = z()
    self.x_lift = z()
    s_app, s_pre, s_close = schedule.breakpoints
    self._s_app, self._s_pre, self._s_close = s_app, s_pre, s_close
    # Nominal ds/dt is constant (s == normalised time along the schedule).
    self.s_rate = 1.0 / schedule.total

  # -- setup ------------------------------------------------------------------

  def set_waypoints(
    self,
    env_ids: torch.Tensor,
    x_start: torch.Tensor,
    x_grasp: torch.Tensor,
    standoff: float,
    lift_height: float,
  ) -> None:
    """Define the path for ``env_ids`` from a start pose and a grasp pose."""
    up = torch.zeros_like(x_grasp)
    up[:, 2] = 1.0
    self.x_start[env_ids] = x_start
    self.x_above[env_ids] = x_grasp + up * standoff
    self.x_grasp[env_ids] = x_grasp
    self.x_lift[env_ids] = x_grasp + up * lift_height

  # -- evaluation -------------------------------------------------------------

  def phase(self, s: torch.Tensor) -> torch.Tensor:
    """Phase code (N,) at path parameter ``s``."""
    ph = torch.full_like(s, float(PHASE_APPROACH))
    ph = torch.where(s >= self._s_app, torch.full_like(s, float(PHASE_DESCEND)), ph)
    ph = torch.where(s >= self._s_pre, torch.full_like(s, float(PHASE_CLOSE)), ph)
    ph = torch.where(s >= self._s_close, torch.full_like(s, float(PHASE_LIFT)), ph)
    return ph.long()

  def position(self, s: torch.Tensor) -> torch.Tensor:
    """``x_ref(s)`` (N, 3), piecewise-linear across the four phases."""
    s = s.clamp(0.0, 1.0).unsqueeze(-1)
    s_app, s_pre, s_close = self._s_app, self._s_pre, self._s_close

    def lerp(a: torch.Tensor, b: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
      return a + (b - a) * u.clamp(0.0, 1.0)

    approach = lerp(self.x_start, self.x_above, s / s_app)
    descend = lerp(self.x_above, self.x_grasp, (s - s_app) / (s_pre - s_app))
    close = self.x_grasp.expand_as(descend)
    lift = lerp(self.x_grasp, self.x_lift, (s - s_close) / (1.0 - s_close))

    out = torch.where(s >= s_app, descend, approach)
    out = torch.where(s >= s_pre, close, out)
    out = torch.where(s >= s_close, lift, out)
    return out

  def velocity(self, s: torch.Tensor, s_dot: torch.Tensor) -> torch.Tensor:
    """``dx_ref/dt = (dx_ref/ds)·ṡ`` (N, 3)."""
    s = s.clamp(0.0, 1.0).unsqueeze(-1)
    s_app, s_pre, s_close = self._s_app, self._s_pre, self._s_close
    d_approach = (self.x_above - self.x_start) / s_app
    d_descend = (self.x_grasp - self.x_above) / (s_pre - s_app)
    d_close = torch.zeros_like(d_descend)
    d_lift = (self.x_lift - self.x_grasp) / (1.0 - s_close)

    out = torch.where(s >= s_app, d_descend, d_approach)
    out = torch.where(s >= s_pre, d_close, out)
    out = torch.where(s >= s_close, d_lift, out)
    return out * s_dot.unsqueeze(-1)
