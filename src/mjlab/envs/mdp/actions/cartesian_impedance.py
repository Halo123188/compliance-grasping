"""Analytical Cartesian impedance action space.

A reusable task-space controller: the policy runs at the (slow) env rate and
emits, per step, a small Cartesian equilibrium offset ``Delta x_ref`` and a
per-axis translational stiffness ``log K``.  An analytical impedance law runs on
every physics substep (the fast inner loop) and turns those into joint torques:

    x_eq   = x_anchor + Delta x_ref                     ("anchor" mode)
    x_eq   = x_cmd,  x_cmd += Delta x_ref               ("integrate" mode)
    F_pos  = K ⊙ (x_eq − x_ee) − D ⊙ ẋ_ee               (D bound to K, see below)
    F_rot  = K_rot · rot_err(q_ee, q_lock) − D_rot · ω_ee
    τ_cmd  = Jpᵀ F_pos + Jrᵀ F_rot + qfrc_bias + τ_null

The policy **never** emits joint torques directly and **never** emits the
damping ``D``: ``D = 2 ζ √(K m_eff)`` is derived from ``K`` so the closed loop
stays passive by construction.  Orientation is locked to the pose captured at
reset with a fixed high stiffness.  A small null-space joint damping keeps the
elbow from drifting on the redundant 7-DoF arm.

The two reference modes trade authority for safety.  In ``"anchor"`` mode the
equilibrium is pinned within ``delta_pos_scale`` of a task-supplied anchor, so
the reachable set is the task's to define.  In ``"integrate"`` mode the policy
steers the reference itself and can therefore express a whole trajectory —
approach, dwell, retreat under load, return — which is what a tracking policy
needs; a leash around the EE keeps the commanded spring extension (and hence the
wrench) bounded.

This module carries **no task assumptions** (no reach goal, no human model): the
equilibrium anchor is read from a named command term, so a different task can
supply a different anchor without touching this file.  The controller writes
joint *effort* targets, so the controlled joints must be motor/torque actuators.

Design notes for the framework:

* ``apply_actions`` runs once per decimation substep, giving the 1 kHz inner
  loop when ``sim.timestep = 1e-3`` and ``decimation = 10``.
* Inside the substep loop ``site_xpos`` lags ``qpos`` by one substep (the env
  only calls ``sim.forward`` once after the loop); at 1 ms this is negligible.
* Fallback: non-finite actions or state collapse the command to a safe pure
  gravity-compensated hold, so a diverging policy can't inject NaNs into physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import compute_pose_error, quat_from_matrix

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CartesianImpedanceActionCfg(ActionTermCfg):
  """Configuration for the analytical Cartesian impedance action term.

  Action layout (6): ``[Δx_ref (3), log K (3)]`` in raw policy units.  Raw
  actions are squashed with ``tanh`` and mapped:

  - ``Δx_ref`` -> ``±delta_pos_scale`` metres (per axis, hard clip).
  - ``log K``  -> ``[log k_min, log k_max]`` -> ``K = exp(...)`` in N/m
    (per-axis, base-frame diagonal stiffness).
  """

  actuator_names: tuple[str, ...] | list[str]
  """Actuator-name expressions selecting the controlled (torque) joints."""

  frame_name: str
  """End-effector site whose pose the impedance law regulates."""

  anchor_command_name: str | None = None
  """Command term name providing the equilibrium anchor ``x_anchor`` (N, >=3).

  Required for ``reference_mode="anchor"``; ignored (and normally ``None``) for
  ``reference_mode="integrate"``."""

  # --- Reference semantics ---
  reference_mode: Literal["anchor", "integrate"] = "anchor"
  """How ``Δx_ref`` maps onto the equilibrium point.

  - ``"anchor"``: ``x_eq = x_anchor + Δx_ref`` — an absolute offset about a
    task-supplied anchor.  The policy cannot move the equilibrium further than
    ``delta_pos_scale`` from the anchor.
  - ``"integrate"``: ``x_cmd ← x_cmd + Δx_ref``, ``x_eq = x_cmd`` — the policy
    *owns* the commanded reference and steers it, so it can represent an entire
    trajectory (approach, dwell, retreat, return) rather than a fixed
    attractor.  ``x_cmd`` is seeded with the EE position on the first step after
    a reset and leashed to ``leash_radius`` around the EE so the achievable
    spring force stays bounded.
  """

  # --- Δx_ref mapping ---
  delta_pos_scale: float = 0.03
  """Half-range of the commanded Cartesian offset, metres.

  In ``"anchor"`` mode this is an absolute offset from the anchor; in
  ``"integrate"`` mode it is the *per-step increment*, i.e. the reference slew
  limit is ``delta_pos_scale / step_dt`` m/s."""

  leash_radius: float | None = None
  """``"integrate"`` mode only: max ‖x_cmd − x_ee‖ (metres).

  Bounds the commanded spring extension, hence the impedance wrench
  (``‖F‖ ≤ K_max · leash_radius``), so a diverging reference cannot ask for an
  unbounded torque.  ``None`` disables the leash."""

  # --- Stiffness mapping ---
  stiffness_range: tuple[float, float] = (50.0, 2000.0)
  """Per-axis translational stiffness range, N/m (plan §3.2)."""

  damping_ratio: float = 0.8
  """ζ for the translational axes; ``D = 2 ζ √(K m_eff)`` (plan §3.2)."""

  effective_mass: float = 2.0
  """Nominal Cartesian mass ``m_eff`` (kg) used for ``D`` (plan §3.2)."""

  # --- Orientation lock ---
  rot_stiffness: float = 50.0
  """Rotational stiffness ``K_rot`` (Nm/rad), fixed (plan §3.1)."""

  rot_damping_ratio: float = 1.0
  """ζ_rot for the orientation lock."""

  rot_effective_inertia: float = 0.1
  """Nominal rotational inertia (kg·m²) used for ``D_rot``."""

  # --- Null-space ---
  nullspace_damping: float = 2.0
  """Joint-space damping (Nm·s/rad) projected into the task null-space."""

  learn_rot_stiffness: bool = False
  """If True the policy also emits ``log K_rot`` (action dim +1), mapped into
  ``rot_stiffness_range``.  Needed once the disturbance carries a *moment*: with
  a fixed orientation lock the arm can only fight the twist, never yield to it."""

  rot_stiffness_range: tuple[float, float] = (5.0, 200.0)
  """``log K_rot`` mapping range (Nm/rad) when ``learn_rot_stiffness``."""

  # --- Ablation ---
  isotropic_stiffness: bool = False
  """If True, the policy emits a single scalar ``log K`` broadcast to all 3 axes
  (action dim 4 = Δx(3)+logK(1)). Used by ablation A1 to test whether the
  anisotropic (per-axis) stiffness is the source of the arbitration benefit."""

  # --- Safety ---
  effort_limit: tuple[float, ...] | None = None
  """Per-joint torque clip (Nm). ``None`` -> no explicit clip here."""

  def build(self, env: ManagerBasedRlEnv) -> CartesianImpedanceAction:
    return CartesianImpedanceAction(self, env)


class CartesianImpedanceAction(ActionTerm):
  """Analytical Cartesian impedance controller (see module docstring)."""

  cfg: CartesianImpedanceActionCfg
  _entity: Entity

  def __init__(self, cfg: CartesianImpedanceActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._num_joints = len(joint_ids)
    self._dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]

    # Resolve EE site -> global site id and its parent body.
    site_local, _ = self._entity.find_sites(cfg.frame_name)
    self._site_id = int(self._entity.indexing.site_ids[site_local[0]].item())
    self._body_id = int(self._env.sim.mj_model.site_bodyid[self._site_id])

    self._n_stiff = 1 if cfg.isotropic_stiffness else 3
    self._n_rot = 1 if cfg.learn_rot_stiffness else 0
    self._action_dim = 3 + self._n_stiff + self._n_rot
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    # Decoded per-step targets (held across the substep loop).
    self._delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._stiffness = torch.zeros(self.num_envs, 3, device=self.device)
    self._damping = torch.zeros(self.num_envs, 3, device=self.device)
    # Final commanded joint torque (post gravity comp / nullspace / clip), exposed
    # for analysis and as a distillation target for a torque policy.
    self._tau = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    # Orientation lock target, captured at reset.
    self._lock_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self._lock_quat[:, 0] = 1.0

    # "integrate" mode: the policy-owned commanded reference. Seeded from the EE
    # position on the first step after a reset (site_xpos is stale inside
    # ``reset()``, which runs before the env's single ``sim.forward()``).
    if cfg.reference_mode == "anchor":
      assert cfg.anchor_command_name is not None, (
        "reference_mode='anchor' requires anchor_command_name"
      )
    self._x_cmd = torch.zeros(self.num_envs, 3, device=self.device)
    self._needs_seed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    # Stiffness map constants.
    k_lo, k_hi = cfg.stiffness_range
    self._log_k_lo = math.log(k_lo)
    self._log_k_span = math.log(k_hi) - math.log(k_lo)

    self._rot_damping = (
      2.0
      * cfg.rot_damping_ratio
      * math.sqrt(cfg.rot_stiffness * cfg.rot_effective_inertia)
    )
    # Per-env rotational impedance (constant unless learn_rot_stiffness).
    self._rot_k = torch.full((self.num_envs, 1), cfg.rot_stiffness, device=self.device)
    self._rot_d = torch.full((self.num_envs, 1), self._rot_damping, device=self.device)
    rk_lo, rk_hi = cfg.rot_stiffness_range
    self._log_rk_lo = math.log(rk_lo)
    self._log_rk_span = math.log(rk_hi) - math.log(rk_lo)

    if cfg.effort_limit is not None:
      self._effort_limit = torch.tensor(
        cfg.effort_limit, device=self.device, dtype=torch.float32
      )
      assert self._effort_limit.shape == (self._num_joints,)
    else:
      self._effort_limit = None

    # Warp Jacobian buffers (mirrors DifferentialIKAction).
    nworld = self.num_envs
    nv = self._env.sim.mj_model.nv
    with wp.ScopedDevice(self._env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)
      self._body_wp.fill_(self._body_id)
    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def stiffness(self) -> torch.Tensor:
    """Decoded per-axis Cartesian stiffness (N, 3), for obs/reward/metrics."""
    return self._stiffness

  @property
  def commanded_reference(self) -> torch.Tensor:
    """The equilibrium point ``x_eq`` the impedance law regulates to (N, 3).

    Only meaningful in ``"integrate"`` mode, where it is the policy-owned
    reference; in ``"anchor"`` mode the equilibrium is anchor-relative and this
    buffer is unused.
    """
    return self._x_cmd

  def process_actions(self, actions: torch.Tensor) -> None:
    # Fallback: sanitize non-finite policy outputs to a safe hold (Δx=0, mid K).
    # Store the sanitized action so the `last_action` observation is finite too.
    safe = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    self._raw_actions[:] = safe
    squashed = torch.tanh(safe)
    self._delta_pos[:] = squashed[:, :3] * self.cfg.delta_pos_scale
    # log K -> [log k_min, log k_max]. Isotropic mode broadcasts one scalar.
    n_k = 3 + self._n_stiff
    log_k = self._log_k_lo + 0.5 * (squashed[:, 3:n_k] + 1.0) * self._log_k_span
    self._stiffness[:] = torch.exp(log_k).expand(self.num_envs, 3)
    if self._n_rot:
      log_rk = self._log_rk_lo + 0.5 * (squashed[:, n_k:] + 1.0) * self._log_rk_span
      self._rot_k[:] = torch.exp(log_rk)
      self._rot_d[:] = (
        2.0
        * self.cfg.rot_damping_ratio
        * torch.sqrt(self._rot_k * self.cfg.rot_effective_inertia)
      )
    self._damping[:] = (
      2.0
      * self.cfg.damping_ratio
      * torch.sqrt(self._stiffness * self.cfg.effective_mass)
    )

    self._seed_after_reset()
    if self.cfg.reference_mode == "integrate":
      self._integrate_reference()

  def _seed_after_reset(self) -> None:
    """Capture the post-reset EE pose for freshly-reset envs.

    ``reset()`` cannot do this: it runs before the env's single ``sim.forward()``,
    so ``site_xpos``/``site_xmat`` there still describe the *previous* episode's
    final pose.  ``process_actions`` runs at the top of ``step()``, after that
    forward, so the pose read here is the true initial pose.
    """
    if not bool(self._needs_seed.any()):
      return
    seed = self._needs_seed
    x_ee = self._env.sim.data.site_xpos[:, self._site_id]
    q_ee = quat_from_matrix(self._env.sim.data.site_xmat[:, self._site_id])
    self._x_cmd[:] = torch.where(seed.unsqueeze(-1), x_ee, self._x_cmd)
    self._lock_quat[:] = torch.where(seed.unsqueeze(-1), q_ee, self._lock_quat)
    self._needs_seed[:] = False

  def _integrate_reference(self) -> None:
    """Advance the policy-owned commanded reference by ``Δx_ref`` (leashed)."""
    x_ee = self._env.sim.data.site_xpos[:, self._site_id]  # (N, 3)
    x_cmd = self._x_cmd + self._delta_pos
    leash = self.cfg.leash_radius
    if leash is not None:
      offset = x_cmd - x_ee
      dist = torch.norm(offset, dim=-1, keepdim=True)
      scale = torch.clamp(leash / dist.clamp(min=1e-9), max=1.0)
      x_cmd = x_ee + offset * scale
    self._x_cmd[:] = torch.nan_to_num(x_cmd, nan=0.0, posinf=0.0, neginf=0.0)

  def apply_actions(self) -> None:
    data = self._env.sim.data

    # Current EE pose.
    x_ee = data.site_xpos[:, self._site_id]  # (N, 3)
    q_ee = quat_from_matrix(data.site_xmat[:, self._site_id])  # (N, 4)

    # Jacobian at the EE point.
    self._point_torch[:] = x_ee
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )
    jacp = self._jacp_torch  # (N, 3, nv)
    jacr = self._jacr_torch
    qvel = data.qvel  # (N, nv)

    # EE spatial velocity from full Jacobian (all dofs).
    v_ee = torch.einsum("bij,bj->bi", jacp, qvel)  # (N, 3)
    w_ee = torch.einsum("bij,bj->bi", jacr, qvel)  # (N, 3)

    # Equilibrium point: anchor+offset, or the policy's integrated reference.
    if self.cfg.reference_mode == "integrate":
      x_eq = self._x_cmd
    else:
      assert self.cfg.anchor_command_name is not None
      anchor = self._env.command_manager.get_command(self.cfg.anchor_command_name)
      assert anchor is not None, "impedance anchor command not found"
      x_eq = anchor[:, :3] + self._delta_pos

    # Translational impedance wrench.
    f_pos = self._stiffness * (x_eq - x_ee) - self._damping * v_ee  # (N, 3)

    # Orientation lock wrench.
    _, rot_err = compute_pose_error(x_ee, q_ee, x_ee, self._lock_quat)  # (N, 3)
    f_rot = self._rot_k * rot_err - self._rot_d * w_ee  # (N, 3)

    # Map wrench to arm joint torques (arm columns only).
    jacp_arm = jacp[:, :, self._dof_ids]  # (N, 3, nj)
    jacr_arm = jacr[:, :, self._dof_ids]
    tau = torch.einsum("bij,bi->bj", jacp_arm, f_pos) + torch.einsum(
      "bij,bi->bj", jacr_arm, f_rot
    )

    # Gravity + Coriolis compensation on the controlled dofs.
    tau = tau + data.qfrc_bias[:, self._dof_ids]

    # Null-space joint damping: project pure joint damping through (I - J⁺J).
    qd_arm = qvel[:, self._dof_ids]
    tau = tau + self._nullspace_torque(
      jacp_arm, jacr_arm, -self.cfg.nullspace_damping * qd_arm
    )

    # Safety: sanitize and clip.
    tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    if self._effort_limit is not None:
      tau = torch.clamp(tau, -self._effort_limit, self._effort_limit)

    self._tau[:] = tau
    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  def _nullspace_torque(
    self, jacp_arm: torch.Tensor, jacr_arm: torch.Tensor, tau_secondary: torch.Tensor
  ) -> torch.Tensor:
    """Project a secondary joint torque into the task null-space.

    N = I − Jᵀ (J Jᵀ)⁻¹ J, with J the stacked position+orientation Jacobian.
    Damped inverse keeps it well-conditioned near singularities.
    """
    j = torch.cat([jacp_arm, jacr_arm], dim=1)  # (N, 6, nj)
    jjt = torch.einsum("bik,bjk->bij", j, j)  # (N, 6, 6)
    eye6 = torch.eye(6, device=self.device).expand_as(jjt)
    jjt_inv = torch.linalg.inv(jjt + 1e-4 * eye6)
    # Apply the task-projection Jᵀ (JJᵀ)⁻¹ J to the secondary torque, then remove
    # it so only the null-space component survives.
    j_tau = torch.einsum("bik,bk->bi", j, tau_secondary)  # (N, 6)
    proj = torch.einsum("bik,bi->bk", j, torch.einsum("bij,bj->bi", jjt_inv, j_tau))
    return tau_secondary - proj

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._delta_pos[env_ids] = 0.0
    # The orientation-lock target and the commanded reference are both captured
    # from the post-reset EE pose on the next ``process_actions``; see
    # ``_seed_after_reset``.
    self._needs_seed[env_ids] = True
