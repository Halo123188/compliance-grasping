"""Spring-to-anchor "human hand" perturbation model (spec §1.4).

A person pushing a robot arm is *not* white noise.  Real pushes are low
frequency, sustained, and gradually varying: a hand grabs the arm, moves toward
somewhere it wants the arm to be, holds, then lets go.  Modelling that as a
spring to a moving anchor reproduces all of those properties for free, and — this
is the part that matters for the teacher — it gives a force that depends on the
arm's *response*, so a compliant arm experiences a decaying force while a stiff
arm experiences a sustained one.  A force sampled open-loop could not do that.

    F_ext = K_h · (x_anchor(t) − x_att) + D_h · (ẋ_anchor(t) − ẋ_att)
    F_ext ← F_ext · min(1, F_max / ‖F_ext‖)              (saturate)

The anchor ramps from the attachment point to a randomized offset along a
min-jerk profile, holds, then the event ends and the force releases.  Both the
ramp and the saturation are needed: a teleported anchor with ``K_h = 800`` and a
20 cm offset is a 160 N impact spike, which is neither humanly possible nor
something the arm should be asked to survive.

Per episode we randomize the number of events (including **none**, for a
meaningful fraction of episodes — see ``p_no_perturbation`` — so the policy still
learns to track cleanly when untouched), and per event the onset phase, the
direction, the magnitude (``K_h`` and the anchor displacement), and the duration.

The force is applied to a single body (the wrist) and is exposed verbatim to the
teacher's admittance law and to the critic; the actor never sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import sample_uniform

# Event phase codes.
_IDLE, _RAMP, _HOLD = 0, 1, 2

MAX_EVENTS = 3
"""Hard cap on scheduled perturbation events per episode (buffer width)."""


@dataclass(kw_only=True)
class PerturbationCfg:
  """Randomization ranges for the human-hand perturbation (spec §1.4)."""

  p_no_perturbation: float = 0.25
  """Fraction of episodes with no perturbation at all (spec: ~20–30%)."""

  num_events_range: tuple[int, int] = (1, 2)
  """Inclusive range on the number of events in a perturbed episode."""

  onset_s_range: tuple[float, float] = (0.05, 0.95)
  """Onset triggers, expressed as thresholds on the teacher's path parameter
  ``s``.  Keying onset to ``s`` rather than to time is what makes "push during
  the approach" / "push after the grasp" well-defined even though ``s`` freezes:
  an event scheduled at ``s = 0.9`` always lands post-grasp."""

  ramp_time_range: tuple[float, float] = (0.25, 0.7)
  """Duration of the min-jerk anchor ramp (s) — how fast the hand moves in."""

  hold_time_range: tuple[float, float] = (0.4, 1.6)
  """Duration of the sustained hold after the ramp completes (s)."""

  displacement_range: tuple[float, float] = (0.03, 0.20)
  """Anchor displacement ‖x_anchor − x_att(t_onset)‖ (m): light touch to strong
  pull."""

  stiffness_range: tuple[float, float] = (80.0, 900.0)
  """Hand stiffness ``K_h`` (N/m), sampled log-uniformly."""

  damping_ratio: float = 0.7
  """ζ_h for ``D_h = 2 ζ_h √(K_h m_h)``."""

  virtual_mass: float = 2.0
  """``m_h`` (kg) used only to derive ``D_h`` from ``K_h``."""

  force_limit: float = 40.0
  """Saturation on ‖F_ext‖ (N).  A human two-handed push tops out around here."""

  min_event_gap_s: float = 0.4
  """Minimum quiet time between the end of one event and the start of the next,
  so the release transient of one push is observable before the next begins."""


class HumanPerturbation:
  """Per-env scheduler + spring model for the human-hand perturbation.

  Owned and ticked by the teacher command term (which needs ``F_ext`` in the
  same call that integrates the admittance law, so the two cannot be separate
  manager terms without introducing a step of skew between them).
  """

  def __init__(self, cfg: PerturbationCfg, num_envs: int, device: str) -> None:
    self.cfg = cfg
    self._n = num_envs
    self._dev = device

    z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
    # Schedule (per event).
    self._onset_s = z(num_envs, MAX_EVENTS)
    self._ramp_T = z(num_envs, MAX_EVENTS)
    self._hold_T = z(num_envs, MAX_EVENTS)
    self._disp = z(num_envs, MAX_EVENTS)
    self._dir = z(num_envs, MAX_EVENTS, 3)
    self._kh_sched = z(num_envs, MAX_EVENTS)
    self._num_events = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Active-event state.
    self._event_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._phase = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._t_in_event = z(num_envs)
    self._quiet_left = z(num_envs)
    self._x_ramp_start = z(num_envs, 3)
    self._anchor_target = z(num_envs, 3)
    self._x_anchor = z(num_envs, 3)
    self._v_anchor = z(num_envs, 3)
    self._kh = z(num_envs)
    self._dh = z(num_envs)
    self._force = z(num_envs, 3)

  # -- properties (privileged; critic + diagnostics only) ---------------------

  @property
  def force(self) -> torch.Tensor:
    """Applied external force ``F_ext`` (N, 3), zero when idle."""
    return self._force

  @property
  def anchor_pos(self) -> torch.Tensor:
    """Hand anchor position (N, 3), world frame."""
    return self._x_anchor

  @property
  def stiffness(self) -> torch.Tensor:
    """Active ``K_h`` (N,), zero when idle."""
    return self._kh

  @property
  def active(self) -> torch.Tensor:
    """Whether a perturbation event is currently applying force (N,) bool."""
    return self._phase != _IDLE

  @property
  def direction(self) -> torch.Tensor:
    """Unit pull direction ``u`` of the active event (N, 3), zero when idle."""
    # _event_idx runs one past the last event once the schedule is exhausted, so
    # clamp before gathering; those envs are idle and get zeroed anyway.
    idx = self._event_idx.clamp(max=MAX_EVENTS - 1)
    gathered = self._dir.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    return gathered * self.active.unsqueeze(-1)

  @property
  def events_remaining(self) -> torch.Tensor:
    """Number of scheduled events not yet started (N,), for the critic."""
    return (self._num_events - self._event_idx).clamp(min=0)

  # -- episode setup ----------------------------------------------------------

  def reset(self, env_ids: torch.Tensor) -> None:
    """Sample a fresh perturbation schedule for ``env_ids``."""
    n = int(env_ids.numel())
    if n == 0:
      return
    cfg = self.cfg
    dev = self._dev

    lo, hi = cfg.num_events_range
    k = torch.randint(lo, hi + 1, (n,), device=dev)
    # ~p_no_perturbation of episodes are left completely untouched so clean
    # tracking stays in-distribution (spec §1.4).
    none = torch.rand(n, device=dev) < cfg.p_no_perturbation
    self._num_events[env_ids] = torch.where(none, torch.zeros_like(k), k)

    shape = (n, MAX_EVENTS)

    def uniform(rng: tuple[float, float]) -> torch.Tensor:
      return sample_uniform(rng[0], rng[1], shape, dev)

    # Sort the onsets so events fire in schedule order.
    self._onset_s[env_ids] = torch.sort(uniform(cfg.onset_s_range), dim=-1).values
    self._ramp_T[env_ids] = uniform(cfg.ramp_time_range)
    self._hold_T[env_ids] = uniform(cfg.hold_time_range)
    self._disp[env_ids] = uniform(cfg.displacement_range)

    u = torch.randn(n, MAX_EVENTS, 3, device=dev)
    self._dir[env_ids] = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    log_lo, log_hi = (math.log(v) for v in cfg.stiffness_range)
    self._kh_sched[env_ids] = torch.exp(sample_uniform(log_lo, log_hi, shape, dev))

    self._event_idx[env_ids] = 0
    self._phase[env_ids] = _IDLE
    self._t_in_event[env_ids] = 0.0
    self._quiet_left[env_ids] = 0.0
    self._force[env_ids] = 0.0
    self._kh[env_ids] = 0.0
    self._dh[env_ids] = 0.0
    self._x_ramp_start[env_ids] = 0.0
    self._anchor_target[env_ids] = 0.0
    self._x_anchor[env_ids] = 0.0
    self._v_anchor[env_ids] = 0.0

  # -- per-step tick ----------------------------------------------------------

  def update(
    self, dt: float, s: torch.Tensor, x_att: torch.Tensor, v_att: torch.Tensor
  ) -> torch.Tensor:
    """Advance the state machine one env step and return ``F_ext`` (N, 3).

    Args:
      dt: env step (s).
      s: teacher path parameter (N,), used for onset triggering.
      x_att / v_att: world position / linear velocity of the attachment body.
    """
    cfg = self.cfg
    self._quiet_left = (self._quiet_left - dt).clamp(min=0.0)

    # --- idle -> ramp: next scheduled event whose onset s has been crossed ----
    pending = self._event_idx < self._num_events
    onset_s = self._onset_s.gather(
      1, self._event_idx.clamp(max=MAX_EVENTS - 1, min=0).unsqueeze(-1)
    ).squeeze(-1)
    start = (
      (self._phase == _IDLE) & pending & (s >= onset_s) & (self._quiet_left <= 0.0)
    )
    if bool(start.any()):
      idx = self._event_idx.clamp(max=MAX_EVENTS - 1)
      gathered = lambda buf: buf.gather(1, idx.unsqueeze(-1)).squeeze(-1)  # noqa: E731
      dirs = self._dir.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
      kh = gathered(self._kh_sched)
      self._x_ramp_start = torch.where(start.unsqueeze(-1), x_att, self._x_ramp_start)
      target = x_att + dirs * gathered(self._disp).unsqueeze(-1)
      self._x_anchor = torch.where(start.unsqueeze(-1), x_att, self._x_anchor)
      self._anchor_target = torch.where(
        start.unsqueeze(-1), target, self._anchor_target
      )
      self._kh = torch.where(start, kh, self._kh)
      self._dh = torch.where(
        start, 2.0 * cfg.damping_ratio * torch.sqrt(kh * cfg.virtual_mass), self._dh
      )
      self._t_in_event = torch.where(
        start, torch.zeros_like(self._t_in_event), self._t_in_event
      )
      self._phase = torch.where(start, torch.full_like(self._phase, _RAMP), self._phase)

    active = self._phase != _IDLE
    self._t_in_event = torch.where(
      active, self._t_in_event + dt, torch.zeros_like(self._t_in_event)
    )

    idx = self._event_idx.clamp(max=MAX_EVENTS - 1)
    ramp_T = self._ramp_T.gather(1, idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
    hold_T = self._hold_T.gather(1, idx.unsqueeze(-1)).squeeze(-1)

    # --- min-jerk anchor ramp, then hold -------------------------------------
    u = (self._t_in_event / ramp_T).clamp(0.0, 1.0)
    p = 6 * u**5 - 15 * u**4 + 10 * u**3
    dpdu = 30 * u**4 - 60 * u**3 + 30 * u**2
    delta = self._anchor_target - self._x_ramp_start
    self._x_anchor = torch.where(
      active.unsqueeze(-1), self._x_ramp_start + delta * p.unsqueeze(-1), self._x_anchor
    )
    self._v_anchor = torch.where(
      active.unsqueeze(-1),
      delta * (dpdu / ramp_T).unsqueeze(-1),
      torch.zeros_like(self._v_anchor),
    )
    self._phase = torch.where(
      (self._phase == _RAMP) & (self._t_in_event >= ramp_T),
      torch.full_like(self._phase, _HOLD),
      self._phase,
    )

    # --- saturated spring force ----------------------------------------------
    f = self._kh.unsqueeze(-1) * (self._x_anchor - x_att) + self._dh.unsqueeze(-1) * (
      self._v_anchor - v_att
    )
    norm = f.norm(dim=-1, keepdim=True)
    f = f * torch.clamp(cfg.force_limit / norm.clamp(min=1e-9), max=1.0)
    self._force = torch.where(active.unsqueeze(-1), f, torch.zeros_like(f))

    # --- release: the hold expires, the hand lets go --------------------------
    done = (self._phase == _HOLD) & (self._t_in_event >= ramp_T + hold_T)
    if bool(done.any()):
      self._phase = torch.where(done, torch.full_like(self._phase, _IDLE), self._phase)
      self._event_idx = torch.where(done, self._event_idx + 1, self._event_idx)
      self._t_in_event = torch.where(
        done, torch.zeros_like(self._t_in_event), self._t_in_event
      )
      self._quiet_left = torch.where(
        done,
        torch.full_like(self._quiet_left, cfg.min_event_gap_s),
        self._quiet_left,
      )
      self._kh = torch.where(done, torch.zeros_like(self._kh), self._kh)
      self._dh = torch.where(done, torch.zeros_like(self._dh), self._dh)
      self._force = torch.where(
        done.unsqueeze(-1), torch.zeros_like(self._force), self._force
      )

    return self._force
