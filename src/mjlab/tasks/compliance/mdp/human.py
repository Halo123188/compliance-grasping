"""Stubborn human-hand disturbance (plan §4).

A per-env state machine ``idle -> reaching -> pushing -> released`` models a
person who reaches into the workspace, grabs the wrist (link7), and tries to
push it to a target displacement ``d`` along a random direction ``u``.  The hand
is a saturated spring:

    x_hand : min-jerk ramp from x_ee(t_push) to x_ee(t_push) + u·d, then held
    F_raw  = K_h (x_hand − x_ee) + D_h (ẋ_hand − ẋ_ee)
    F_ext  = F_raw · min(1, F_max / ‖F_raw‖)          # saturate at F_max
    K_h    ~ loguniform(kh_range) N/m,   D_h = 2·0.7·√(K_h·2)

Structural rules (do not "fix" these — see plan §4.3):

* **No timeout.** The person is stubborn: if the arm never yields, the push
  lasts to the end of the episode and success is unreachable.  A timeout would
  turn "hold your ground" into the optimal policy and destroy the experiment.
* **Release is by position, not force**: ``‖x_ee − x_hand‖ < 2 cm`` held 0.2 s.
  A force threshold would make a stiff person demand tighter tolerance — an
  unobservable difficulty knob keyed on the hidden ``K_h``.
* Min-jerk ramp **and** force saturation are both required; either alone leaks a
  200 N impact spike.

The grab point is sampled per episode from ``grab_bodies`` (forearm..gripper);
``f_max`` and the release rule are randomised per episode too (e11).  ``d`` is curriculum-controlled
(plan §6); everything else is randomized from step one.  The force is written to
``xfrc_applied`` and drawn as an arrow; the hand target is drawn as a sphere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Phase codes.
_IDLE, _REACHING, _PUSHING, _RELEASED = 0, 1, 2, 3


class HumanDisturbance:
  """Stateful ``mode="step"`` event term (see module docstring)."""

  @dataclass
  class VizCfg:
    force_color: tuple[float, float, float, float] = (0.95, 0.35, 0.1, 0.95)
    force_scale: float = 0.012  # arrow metres per Newton
    force_width: float = 0.03
    min_force: float = 1.0  # N, below which no arrow
    hand_color: tuple[float, float, float, float] = (0.1, 0.7, 0.95, 0.9)
    hand_radius: float = 0.03

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._num_envs = env.num_envs
    self._device = env.device
    self._dt = env.step_dt

    p = cfg.params
    self._asset: Entity = env.scene[p["asset_cfg"].name]
    # Grab point: a person does not always take the same link.  Sample one of
    # ``grab_bodies`` per episode (item 4); falls back to the single
    # ``attach_body`` when not given.
    grab_names = p.get("grab_bodies", None) or [p["attach_body"]]
    ids = []
    for nm in grab_names:
      bid, _ = self._asset.find_bodies(nm)
      ids.append(bid[0])
    self._grab_body_ids = ids
    self._body_id = ids[0]  # default / viz fallback

    self._f_max: float = p.get("f_max", 40.0)
    # Randomised per episode (items 3 and 7).  Defaults reproduce the old fixed
    # behaviour so existing configs are unchanged.
    self._f_max_range: tuple[float, float] = p.get(
      "f_max_range", (self._f_max, self._f_max)
    )
    self._release_frac_range: tuple[float, float] = (
      p.get("release_frac_range", None) or None
    )
    self._release_hold_range: tuple[float, float] = (
      p.get("release_hold_range", None) or None
    )
    self._push_time_range: tuple[float, float] = p.get("push_time_range", (0.0, 0.3))
    self._reach_time_range: tuple[float, float] = p.get("reach_time_range", (0.3, 0.8))
    self._kh_range: tuple[float, float] = p.get("kh_range", (100.0, 1000.0))
    self._release_dist: float = p.get("release_dist", 0.02)
    self._release_frac: float = p.get("release_frac", 0.7)
    self._release_hold: float = p.get("release_hold", 0.2)
    self._zeta_h: float = p.get("zeta_h", 0.7)
    self._m_eff: float = p.get("m_eff", 2.0)
    self._second_push_prob: float = p.get("second_push_prob", 0.3)
    self._second_delay_range: tuple[float, float] = p.get(
      "second_delay_range", (0.5, 1.5)
    )
    # A1 environment load: unpredictable force (N, per-axis std) the arm must
    # reject with stiffness to hold the goal.  Resampled every
    # ``env_load_resample_s`` so it cannot be cancelled by an x_ref bias.  0
    # disables it (the original free-space task).  Not counted by the force
    # penalty (see __call__).
    # Item 6: disturbance shape sampled per push ("hold" | "shake" | "drag").
    self._shape_modes: tuple[str, ...] = tuple(p.get("shape_modes", ("hold",)))
    self._shake_amp_range: tuple[float, float] = p.get("shake_amp_range", (0.03, 0.08))
    self._shake_hz_range: tuple[float, float] = p.get("shake_hz_range", (0.5, 2.0))
    self._drag_speed_range: tuple[float, float] = p.get(
      "drag_speed_range", (0.03, 0.12)
    )
    # Item 5a: a real grip acts at an offset from the body CoM, so it also
    # transmits a moment r x F.  0 keeps the old pure-force behaviour.
    self._grip_offset: float = p.get("grip_offset", 0.0)
    self._env_load_std: float = p.get("env_load_std", 0.0)
    self._env_load_resample_s: float = p.get("env_load_resample_s", 0.2)
    self._viz: HumanDisturbance.VizCfg = p.get("viz", HumanDisturbance.VizCfg())

    # Curriculum-controlled displacement range (metres); level 0 default.
    self.d_range: tuple[float, float] = p.get("d_range", (0.02, 0.05))

    self._alloc()

  # ---- state ----------------------------------------------------------------

  def _alloc(self) -> None:
    n, dev = self._num_envs, self._device

    def z(*s: int) -> torch.Tensor:
      return torch.zeros(*s, device=dev)

    self._phase = torch.zeros(n, dtype=torch.long, device=dev)
    self._t = z(n)
    self._t_push = z(n)
    self._x_start = z(n, 3)
    self._x_target = z(n, 3)
    self._reach_T = z(n)
    self._reach_t0 = z(n)
    self._kh = z(n)
    self._dh = z(n)
    self._release_dwell = z(n)
    self._second_time = z(n)
    self._second_pending = torch.zeros(n, dtype=torch.bool, device=dev)
    # Cached for viz.
    self._x_hand = z(n, 3)
    self._force = z(n, 3)
    # A1 environment load (current value + next-resample time).
    self._env_load = z(n, 3)
    self._env_load_t = z(n)
    # Per-episode randomised human properties (items 3, 4, 7).
    self._f_max_env = z(n)
    self._release_frac_env = z(n)
    self._release_hold_env = z(n)
    self._grab_idx = torch.zeros(n, dtype=torch.long, device=dev)
    self._shape = torch.zeros(n, dtype=torch.long, device=dev)
    self._shake_axis = z(n, 3)
    self._shake_amp = z(n)
    self._shake_w = z(n)
    self._drag_v = z(n)
    self._grip_r = z(n, 3)
    self._grab_body_t = torch.tensor(self._grab_body_ids, dtype=torch.long, device=dev)
    self._env_arange = torch.arange(n, device=dev)

  # ---- helpers --------------------------------------------------------------

  def _grab_body(self) -> torch.Tensor:
    """Per-env body id currently being grabbed."""
    return self._grab_body_t[self._grab_idx]

  def _attach_pos(self) -> torch.Tensor:
    return self._asset.data.body_com_pos_w[self._env_arange, self._grab_body()]

  def _attach_vel(self) -> torch.Tensor:
    return self._asset.data.body_com_vel_w[self._env_arange, self._grab_body(), :3]

  def _sample_push(self, ids: torch.Tensor, x_now: torch.Tensor) -> None:
    """(Re)initialise a push for the given envs starting at ``x_now``."""
    n = len(ids)
    dev = self._device
    u = torch.randn(n, 3, device=dev)
    u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-9)
    d = sample_uniform(self.d_range[0], self.d_range[1], (n, 1), device=dev)
    self._x_start[ids] = x_now
    self._x_target[ids] = x_now + u * d
    self._reach_T[ids] = sample_uniform(*self._reach_time_range, (n,), device=dev)
    self._reach_t0[ids] = self._t[ids]
    kh = torch.exp(
      sample_uniform(
        torch.log(torch.tensor(self._kh_range[0])),
        torch.log(torch.tensor(self._kh_range[1])),
        (n,),
        device=dev,
      )
    )
    self._kh[ids] = kh
    self._dh[ids] = 2.0 * self._zeta_h * torch.sqrt(kh * self._m_eff)
    self._release_dwell[ids] = 0.0
    # Item 6: pick the interaction shape for this push.
    if len(self._shape_modes) > 1:
      self._shape[ids] = torch.randint(0, len(self._shape_modes), (n,), device=dev)
    ax = torch.randn(n, 3, device=dev)
    self._shake_axis[ids] = ax / (torch.norm(ax, dim=-1, keepdim=True) + 1e-9)
    self._shake_amp[ids] = sample_uniform(*self._shake_amp_range, (n,), device=dev)
    self._shake_w[ids] = (
      2.0 * math.pi * sample_uniform(*self._shake_hz_range, (n,), device=dev)
    )
    self._drag_v[ids] = sample_uniform(*self._drag_speed_range, (n,), device=dev)
    # Item 5a: random lever arm for the grip moment.
    if self._grip_offset > 0.0:
      r = torch.randn(n, 3, device=dev)
      r = r / (torch.norm(r, dim=-1, keepdim=True) + 1e-9)
      self._grip_r[ids] = r * sample_uniform(0.0, self._grip_offset, (n, 1), device=dev)
    self._phase[ids] = _REACHING

  def _hand_state(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-jerk hand position and velocity for all envs (held after ramp)."""
    s = torch.clamp(
      (self._t - self._reach_t0) / self._reach_T.clamp(min=1e-6), 0.0, 1.0
    )
    p = 6 * s**5 - 15 * s**4 + 10 * s**3
    dpds = 30 * s**4 - 60 * s**3 + 30 * s**2
    delta = self._x_target - self._x_start
    x_hand = self._x_start + delta * p.unsqueeze(-1)
    v_hand = delta * (dpds / self._reach_T.clamp(min=1e-6)).unsqueeze(-1)

    # Item 6: after the ramp completes, the interaction continues differently
    # depending on the sampled shape.  Release is still yield-conditioned in every
    # mode (no timeout — plan §4.3).
    if len(self._shape_modes) > 1:
      t_after = torch.clamp(self._t - self._reach_t0 - self._reach_T, min=0.0)
      names = self._shape_modes
      if "shake" in names:
        j = names.index("shake")
        osc = torch.sin(self._shake_w * t_after) * self._shake_amp
        dosc = torch.cos(self._shake_w * t_after) * self._shake_amp * self._shake_w
        m = (self._shape == j).unsqueeze(-1)
        x_hand = torch.where(m, x_hand + self._shake_axis * osc.unsqueeze(-1), x_hand)
        v_hand = torch.where(m, v_hand + self._shake_axis * dosc.unsqueeze(-1), v_hand)
      if "drag" in names:
        j = names.index("drag")
        u = delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-9)
        m = (self._shape == j).unsqueeze(-1)
        x_hand = torch.where(
          m, x_hand + u * (self._drag_v * t_after).unsqueeze(-1), x_hand
        )
        v_hand = torch.where(m, v_hand + u * self._drag_v.unsqueeze(-1), v_hand)
    return x_hand, v_hand

  # ---- main tick ------------------------------------------------------------

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **kwargs) -> None:
    del env, env_ids, kwargs
    self._t += self._dt
    x_ee = self._attach_pos()
    v_ee = self._attach_vel()

    # idle -> reaching (onset).
    onset = (self._phase == _IDLE) & (self._t >= self._t_push)
    if onset.any():
      self._sample_push(onset.nonzero(as_tuple=False).squeeze(-1), x_ee[onset])

    # released -> reaching (scheduled second push).
    second = (
      self._second_pending & (self._phase == _RELEASED) & (self._t >= self._second_time)
    )
    if second.any():
      ids = second.nonzero(as_tuple=False).squeeze(-1)
      self._second_pending[ids] = False
      self._sample_push(ids, x_ee[ids])

    # reaching -> pushing when the ramp completes.
    ramp_done = (self._phase == _REACHING) & (self._t - self._reach_t0 >= self._reach_T)
    self._phase[ramp_done] = _PUSHING

    active = (self._phase == _REACHING) | (self._phase == _PUSHING)

    # Spring force (saturated), only for active envs.
    x_hand, v_hand = self._hand_state()
    f_raw = self._kh.unsqueeze(-1) * (x_hand - x_ee) + self._dh.unsqueeze(-1) * (
      v_hand - v_ee
    )
    f_norm = torch.norm(f_raw, dim=-1, keepdim=True)
    scale = torch.clamp(self._f_max_env.unsqueeze(-1) / (f_norm + 1e-9), max=1.0)
    force = f_raw * scale
    force = torch.where(active.unsqueeze(-1), force, torch.zeros_like(force))
    self._x_hand[:] = x_hand
    self._force[:] = force

    # A1 environment load: a small unpredictable force the arm must reject with
    # stiffness to hold the goal.  Resampled every env_load_resample_s so the
    # policy cannot cancel it with an x_ref bias (only stiffness rejects an
    # unpredictable load).  It is applied to the sim but NOT stored in
    # self._force, so the force penalty keys only on the human hand — the arm is
    # meant to resist this one, yielding only to the hand.
    if self._env_load_std > 0.0:
      due = self._t >= self._env_load_t
      if due.any():
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        self._env_load[ids] = (
          torch.randn(len(ids), 3, device=self._device) * self._env_load_std
        )
        self._env_load_t[ids] = self._t[ids] + self._env_load_resample_s
      applied = force + self._env_load
    else:
      applied = force

    # Write wrench to sim (zero torque for now; item 5 will add a moment).
    # Each candidate grab body is written every step -- zero for the envs not
    # currently grabbing there, so no stale wrench is left behind.
    # Item 5a: moment from the grip lever arm (hand force only; the A1 load stays
    # a pure force).  Zero when grip_offset == 0.
    if self._grip_offset > 0.0:
      torque = torch.cross(self._grip_r, force, dim=-1)
    else:
      torque = torch.zeros_like(applied)
    for j, b in enumerate(self._grab_body_ids):
      m = (self._grab_idx == j).unsqueeze(-1)
      fj = torch.where(m, applied, torch.zeros_like(applied))
      tj = torch.where(m, torque, torch.zeros_like(torque))
      self._asset.write_external_wrench_to_sim(
        fj.unsqueeze(1), tj.unsqueeze(1), body_ids=[b]
      )

    # Release check (position-based, held): only while actively pushing.
    # Two robust position-based triggers (either one, held ``release_hold``):
    #   * the arm reached the hand target within ``release_dist``, or
    #   * the arm yielded >= ``release_frac`` of the pull distance along u.
    # The yield-fraction path is robust to the A1 jitter and to the arm not
    # landing exactly on H, so "go soft, get pulled most of the way" reliably
    # earns a release — which is what closes the yield -> release -> return loop.
    # It stays purely kinematic (no force threshold; plan §4.3).
    delta_th = self._x_target - self._x_start
    d_th = torch.norm(delta_th, dim=-1)
    proj = ((x_ee - self._x_start) * delta_th).sum(-1) / d_th.clamp(min=1e-6)
    yielded = proj >= self._release_frac_env * d_th
    close = torch.norm(x_ee - x_hand, dim=-1) < self._release_dist
    within = yielded | close
    self._release_dwell = torch.where(
      active & within,
      self._release_dwell + self._dt,
      torch.zeros_like(self._release_dwell),
    )
    released_now = active & (self._release_dwell >= self._release_hold_env)
    if released_now.any():
      ids = released_now.nonzero(as_tuple=False).squeeze(-1)
      self._phase[ids] = _RELEASED
      # Schedule a possible second push.
      roll = torch.rand(len(ids), device=self._device) < self._second_push_prob
      self._second_pending[ids] = roll
      delay = sample_uniform(
        *self._second_delay_range, (len(ids),), device=self._device
      )
      self._second_time[ids] = self._t[ids] + delay

  # ---- viz ------------------------------------------------------------------

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    x_ee = self._attach_pos()
    for i in visualizer.get_env_indices(self._num_envs):
      if self._phase[i].item() not in (_REACHING, _PUSHING):
        continue
      hand = self._x_hand[i].cpu().numpy()
      visualizer.add_sphere(
        center=hand,
        radius=self._viz.hand_radius,
        color=self._viz.hand_color,
        label=f"human_hand_{i}",
      )
      f = self._force[i]
      if torch.norm(f).item() < self._viz.min_force:
        continue
      start = x_ee[i].cpu().numpy()
      end = start + f.cpu().numpy() * self._viz.force_scale
      visualizer.add_arrow(
        start=start,
        end=end,
        color=self._viz.force_color,
        width=self._viz.force_width,
        label=f"human_force_{i}",
      )

  # ---- reset ----------------------------------------------------------------

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    # Zero any lingering wrench.
    if isinstance(env_ids, slice):
      ids = torch.arange(self._num_envs, device=self._device)
    else:
      ids = env_ids
    zeros = torch.zeros(len(ids), 1, 3, device=self._device)
    for b in self._grab_body_ids:
      self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=[b])

    self._phase[env_ids] = _IDLE
    self._t[env_ids] = 0.0
    self._t_push[env_ids] = sample_uniform(
      *self._push_time_range, (len(ids),), device=self._device
    )
    self._release_dwell[env_ids] = 0.0
    self._second_pending[env_ids] = False
    self._force[env_ids] = 0.0
    self._env_load[env_ids] = 0.0
    self._env_load_t[env_ids] = 0.0
    n_ids = len(ids)
    self._f_max_env[env_ids] = sample_uniform(
      *self._f_max_range, (n_ids,), device=self._device
    )
    if self._release_frac_range is not None:
      self._release_frac_env[env_ids] = sample_uniform(
        *self._release_frac_range, (n_ids,), device=self._device
      )
    else:
      self._release_frac_env[env_ids] = self._release_frac
    if self._release_hold_range is not None:
      self._release_hold_env[env_ids] = sample_uniform(
        *self._release_hold_range, (n_ids,), device=self._device
      )
    else:
      self._release_hold_env[env_ids] = self._release_hold
    if len(self._grab_body_ids) > 1:
      self._grab_idx[env_ids] = torch.randint(
        0, len(self._grab_body_ids), (n_ids,), device=self._device
      )

  # ---- introspection for reward/metrics (privileged) -----------------------

  def external_force(self) -> torch.Tensor:
    """Applied human force ``F_ext`` per env (N, 3). Privileged (reward only)."""
    return self._force

  def push_dir(self) -> torch.Tensor:
    """Unit push direction ``u`` per env (N, 3); zero when not pushing."""
    delta = self._x_target - self._x_start
    return delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-9)

  def is_pushing(self) -> torch.Tensor:
    return (self._phase == _REACHING) | (self._phase == _PUSHING)


@dataclass(kw_only=True)
class HumanDisturbanceCfg:
  """Convenience factory for the human-disturbance ``EventTermCfg``.

  Use :meth:`make` to get an ``EventTermCfg(mode="step", func=HumanDisturbance, ...)``.
  """

  asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))
  attach_body: str = "link7"
  f_max: float = 40.0
  push_time_range: tuple[float, float] = (0.0, 0.3)
  reach_time_range: tuple[float, float] = (0.3, 0.8)
  kh_range: tuple[float, float] = (100.0, 1000.0)
  release_dist: float = 0.02
  release_frac: float = 0.7
  release_hold: float = 0.2
  f_max_range: tuple[float, float] | None = None
  release_frac_range: tuple[float, float] | None = None
  release_hold_range: tuple[float, float] | None = None
  grab_bodies: tuple[str, ...] | None = None
  shape_modes: tuple[str, ...] = ("hold",)
  grip_offset: float = 0.0
  second_push_prob: float = 0.3
  second_delay_range: tuple[float, float] = (0.5, 1.5)
  env_load_std: float = 0.0
  env_load_resample_s: float = 0.2
  d_range: tuple[float, float] = (0.02, 0.05)
