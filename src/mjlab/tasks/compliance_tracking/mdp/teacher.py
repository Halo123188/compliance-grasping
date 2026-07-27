"""Privileged analytic teacher for compliant reach-and-grasp (spec §1).

The teacher is the *whole* specification of desired behaviour.  It runs only in
sim, sees everything, and emits a target signal ``(x_t, ẋ_t, F_finger_target)``
that the student is trained to track.  Nothing about yielding, returning,
resuming, or holding a grasp appears in the reward — all of it is already in
this signal.  That is the point of the teacher–student framing: the alternative,
a hand-weighted sum of compliance / progress / force terms, has to be re-tuned
every time the task changes, and its optimum is only incidentally the behaviour
you wanted.

Four mechanisms, in the order they compose:

**Path-parameter freezing (§1.2).**  ``s`` advances at the nominal rate only
while the arm is inside a tube around the reference; deviation scales ``ṡ``
smoothly to zero.  So a push does not consume trajectory, and on release the
reference is exactly where the arm left it — "resume where you left off" is a
property of the target, not something the policy must invent.

**Admittance integration (§1.3).**  The target is *not* ``x_ref + F/k``.  A
quasi-static displacement law has no transient: it snaps on contact and snaps
back on release, and a policy trained to track it learns to snap too.  Instead we
integrate a reference impedance

    m ẍ_t + d ẋ_t + k (x_t − x_ref(s)) = F_ext

with ``ζ ≈ 0.9``, giving dynamically consistent yielding under load and a smooth,
correctly-damped return when the hand lets go.

**Scripted grasp (§1.5).**  Once ``s`` reaches the closure phase with small
deviation, the finger force target ramps to ``F_grasp`` and holds.  The student
learns to *time and integrate* this, not to discover a grasp.

**Weld on grasp (§1.6).**  Grasp stability under an external push is exactly the
regime where the contact solver is least trustworthy, and RL will happily farm
solver artifacts.  So once the grasp is established we weld the object to the
hand and stop asking the physics that question.  The accepted cost is stated
plainly: this pipeline cannot teach slip recovery, and does not try to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.compliance_tracking.mdp.path import (
  PHASE_CLOSE,
  PathSchedule,
  ReferencePath,
)
from mjlab.tasks.compliance_tracking.mdp.perturbation import (
  HumanPerturbation,
  PerturbationCfg,
)
from mjlab.utils.lab_api.math import (
  compute_pose_error,
  quat_from_matrix,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

WELD_EQUALITY_NAME = "teacher_grasp_weld"
"""Name of the (initially inactive) object-to-hand weld the teacher activates."""

# Grasp-script codes.
GRASP_OPEN = 0
GRASP_CLOSING = 1
GRASP_HOLDING = 2


@dataclass(kw_only=True)
class TeacherCommandCfg(CommandTermCfg):
  """Configuration for the analytic teacher."""

  robot_name: str = "robot"
  ee_site_name: str = "grasp_site"
  attach_body_name: str = "link7"
  """Body the human perturbation force is applied to (the wrist)."""

  push_body_names: tuple[str, ...] = ()
  """If non-empty, the push location is randomized per episode over these bodies
  (base->wrist, e.g. link4..link7) instead of always ``attach_body_name``.  This
  trains a policy that yields to a push *anywhere* on the arm, not just the
  wrist."""

  push_admittance_grade: tuple[float, ...] = ()
  """Per-body admittance stiffness matching ``push_body_names`` order, so the
  compliance is *graded* by where the push lands: soft (low k, big yield) near
  the wrist, stiff (high k) near the base.  Empty falls back to the scalar
  ``admittance_stiffness`` for every location."""

  # --- Task geometry ---------------------------------------------------------
  object_name: str | None = None
  """Scene entity grasped in Stage B+.  ``None`` (Stage A) means free-space
  reach: the grasp waypoint is sampled in ``grasp_box`` and no weld is made."""

  grasp_box_min: tuple[float, float, float] = (0.42, -0.10, 0.02)
  grasp_box_max: tuple[float, float, float] = (0.58, 0.10, 0.02)
  """Box the (Stage A) virtual grasp waypoint is sampled from, base frame."""

  standoff: float = 0.12
  """Height of the pre-grasp standoff above the grasp pose (m)."""

  lift_height: float = 0.10
  """Post-grasp lift above the grasp pose (m)."""

  schedule: PathSchedule = field(default_factory=PathSchedule)

  # --- §1.2 path-parameter freezing ------------------------------------------
  freeze_deviation: float = 0.04
  """``d_freeze`` (m): deviation at which ``ṡ`` has fallen to zero (spec: 3–5 cm)."""

  freeze_onset: float = 0.015
  """Deviation (m) below which ``ṡ`` is still at the full nominal rate.  Between
  this and ``freeze_deviation`` the rate scales down smoothly — a hard switch
  would put a discontinuity in the target velocity the student has to track."""

  # --- §1.3 reference impedance ----------------------------------------------
  admittance_mass: float = 2.0
  """``m`` (kg)."""

  admittance_stiffness: float = 300.0
  """``k`` (N/m).  Sets how far the target yields per newton in steady state."""

  admittance_damping_ratio: float = 0.9
  """``ζ``; ``d = 2 ζ √(k m)`` (spec §1.3: well damped, 0.8–1.0)."""

  # --- Stiffness supervision (optional, breaks pure-tracking; see rewards) ----
  supervise_stiffness: bool = False
  """Emit a target Cartesian stiffness ``K_target`` alongside ``x_t`` so the
  reward can score the impedance the policy commands, not only the trajectory it
  produces.  Off by default: with it on, ``K_target`` is a hand-authored
  anisotropy profile, which is exactly the behavioural specification the
  teacher–student framing was meant to avoid.  It exists because the trajectory
  reward alone cannot distinguish "yielded because soft" from "stiffly drove the
  reference away" — see ``rewards.stiffness_tracking`` and the task README."""

  k_precision: float = 900.0
  """``K`` (N/m) targeted on every axis when not being pushed.  High enough to
  track, with headroom below the 2000 action ceiling."""

  k_soft: float = 200.0
  """``K`` (N/m) targeted *along the pull direction* while a hand is pushing
  (perpendicular axes stay at ``k_precision``).  Below the admittance ``k`` (300)
  so yielding along the push is genuinely low-effort."""

  # --- Compliant torque target (exp-2: direct-torque action) ------------------
  emit_torque_target: bool = False
  """Compute a compliant joint-torque target ``tau_target`` for the torque-action
  variant.  It is an anisotropic Cartesian impedance about the *un-yielded*
  reference ``x_ref``: a full rank-1 stiffness ``K(u) = k_precision·I −
  (k_precision − k_soft)·uuᵀ`` (soft along the pull, stiff perpendicular — the
  rotated matrix a diagonal action stiffness cannot represent), plus orientation
  lock, null-space damping and gravity comp, mapped to joint torque via ``Jᵀ``.
  The direct-torque policy is rewarded for reproducing it (``mdp.torque_tracking``)
  from proprioception.  Because ``K`` is applied about ``x_ref`` with ``k_soft``
  along ``u``, the arm yields ``F_ext/k_soft`` along the push by construction."""

  arm_joint_names: tuple[str, ...] = ()
  """Arm joints ``tau_target`` drives (required when ``emit_torque_target``)."""

  torque_rot_stiffness: float = 50.0
  """Orientation-lock rotational stiffness in ``tau_target`` (Nm/rad)."""

  torque_nullspace_damping: float = 2.0
  """Null-space joint damping in ``tau_target`` (Nm·s/rad)."""

  # --- §1.5 scripted grasp ----------------------------------------------------
  grasp_force: float = 15.0
  """``F_grasp`` (N), the held total normal force after closure."""

  grasp_ramp_time: float = 0.25
  """Time (s) over which the finger force target ramps from 0 to ``F_grasp``."""

  grasp_deviation_gate: float = 0.03
  """Closure only starts when the tracking deviation is below this (m)."""

  # --- §1.6 weld --------------------------------------------------------------
  weld_enabled: bool = True
  weld_distance: float = 0.05
  """Max object-to-grasp-site distance (m) at which the weld may be made."""

  # --- Perturbation ------------------------------------------------------------
  perturbation: PerturbationCfg = field(default_factory=PerturbationCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.15, 0.85, 0.35, 0.6)
    reference_color: tuple[float, float, float, float] = (0.95, 0.85, 0.15, 0.5)
    hand_color: tuple[float, float, float, float] = (0.15, 0.65, 0.95, 0.85)
    force_color: tuple[float, float, float, float] = (0.95, 0.35, 0.1, 0.95)
    marker_radius: float = 0.022
    force_scale: float = 0.008
    force_width: float = 0.025

  viz: VizCfg = field(default_factory=lambda: TeacherCommandCfg.VizCfg())

  def build(self, env: ManagerBasedRlEnv) -> TeacherCommand:
    return TeacherCommand(self, env)


class TeacherCommand(CommandTerm):
  """The analytic teacher (see module docstring)."""

  cfg: TeacherCommandCfg

  def __init__(self, cfg: TeacherCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.robot_name]
    site_local, _ = self.robot.find_sites(cfg.ee_site_name)
    self._site_local = int(site_local[0])
    self._site_id = int(self.robot.indexing.site_ids[site_local[0]].item())
    body_ids, _ = self.robot.find_bodies(cfg.attach_body_name)
    self._attach_body = body_ids[0]

    n, dev = self.num_envs, self.device
    # Randomized / graded push location (optional). Resolve the candidate bodies
    # and their per-location admittance stiffness once; sampling is per episode.
    self._multi_push = len(cfg.push_body_names) > 0
    if self._multi_push:
      assert len(cfg.push_admittance_grade) == len(cfg.push_body_names), (
        "push_admittance_grade must match push_body_names"
      )
      self._push_body_ids = [
        self.robot.find_bodies(nm)[0][0] for nm in cfg.push_body_names
      ]
      self._push_grade = torch.tensor(cfg.push_admittance_grade, device=dev)
      self._push_body_id_t = torch.tensor(self._push_body_ids, device=dev)
    z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731

    self.path = ReferencePath(n, dev, cfg.schedule)
    self.perturbation = HumanPerturbation(cfg.perturbation, n, dev)

    # Path parameter and admittance state.
    self.s = z(n)
    self.s_dot = z(n)
    self.x_t = z(n, 3)
    self.xd_t = z(n, 3)
    self.deviation = z(n)

    # Scripted grasp.
    self.grasp_state = torch.zeros(n, dtype=torch.long, device=dev)
    self._grasp_timer = z(n)
    self.finger_force_target = z(n)
    self.welded = torch.zeros(n, dtype=torch.bool, device=dev)

    # Target Cartesian stiffness the reward tracks when supervise_stiffness is on
    # (isotropic k_precision when idle; softened along the pull while pushed).
    self.k_target = torch.full((n, 3), cfg.k_precision, device=dev)

    # Episode start pose is only known after the reset events have been written
    # *and* forward() has run, which is after ``reset()``; defer to the first
    # ``_update_command`` (see _seed).
    self._needs_seed = torch.ones(n, dtype=torch.bool, device=dev)
    self._grasp_target = z(n, 3)
    self._dt = env.step_dt

    self._damping = (
      2.0
      * cfg.admittance_damping_ratio
      * math.sqrt(cfg.admittance_stiffness * cfg.admittance_mass)
    )
    # Per-env admittance k / damping (graded by push location when _multi_push;
    # otherwise the scalar above for every env). _push_slot indexes push_body_ids.
    self._push_slot = torch.zeros(n, dtype=torch.long, device=dev)
    self._k_adm = torch.full((n,), cfg.admittance_stiffness, device=dev)
    self._damp_adm = torch.full((n,), self._damping, device=dev)

    # Weld bookkeeping.
    self._weld_eq_id: int | None = None
    self._object_body: int | None = None
    self._gripper_body: int = 0
    self._object: Entity | None = None
    if cfg.object_name is not None:
      self._object = env.scene[cfg.object_name]
      if cfg.weld_enabled:
        self._setup_weld(env)

    # Compliant torque target (exp-2). Orientation lock is seeded like x_t.
    self.torque_target: torch.Tensor | None = None
    self._lock_quat = torch.zeros(n, 4, device=dev)
    self._lock_quat[:, 0] = 1.0
    if cfg.emit_torque_target:
      self._setup_torque_target(env)

    self.metrics["path_parameter"] = z(n)
    self.metrics["tracking_error"] = z(n)
    self.metrics["perturbation_force"] = z(n)
    self.metrics["grasped"] = z(n)

  # -- weld setup ---------------------------------------------------------------

  def _setup_weld(self, env: ManagerBasedRlEnv) -> None:
    """Resolve the weld equality and make ``eq_data`` per-env writable."""
    model = env.sim.mj_model
    try:
      eq = model.equality(WELD_EQUALITY_NAME)
    except KeyError as exc:  # pragma: no cover - config error
      raise RuntimeError(
        f"Weld equality '{WELD_EQUALITY_NAME}' not found in the compiled model. "
        "The scene must add it (see add_grasp_weld) when weld_enabled=True."
      ) from exc
    self._weld_eq_id = int(eq.id)
    # Each env welds at its own relative pose, so eq_data must be per-world.
    env.sim.expand_model_fields(("eq_data",))
    self._gripper_body = int(model.body(f"{self.cfg.robot_name}/gripper_base").id)
    assert self._object is not None
    self._object_body = int(self._object.indexing.root_body_id)

  # -- exp-2 compliant torque target -------------------------------------------

  def _setup_torque_target(self, env: ManagerBasedRlEnv) -> None:
    """Warp Jacobian buffers + arm DOF ids (mirrors CartesianImpedanceAction)."""
    cfg = self.cfg
    assert cfg.arm_joint_names, "emit_torque_target requires arm_joint_names"
    jids, _ = self.robot.find_joints(list(cfg.arm_joint_names), preserve_order=True)
    self._arm_joint_ids = torch.tensor(jids, device=self.device, dtype=torch.long)
    self._arm_dof_ids = self.robot.indexing.joint_v_adr[self._arm_joint_ids]
    self._body_id = int(env.sim.mj_model.site_bodyid[self._site_id])
    nworld, nv = self.num_envs, env.sim.mj_model.nv
    with wp.ScopedDevice(env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)
      self._body_wp.fill_(self._body_id)
    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)
    self.torque_target = torch.zeros(self.num_envs, len(jids), device=self.device)
    m = cfg.admittance_mass
    dr = cfg.admittance_damping_ratio
    self._td_soft = 2.0 * dr * math.sqrt(cfg.k_soft * m)
    self._td_stiff = 2.0 * dr * math.sqrt(cfg.k_precision * m)
    self._trot_d = 2.0 * dr * math.sqrt(cfg.torque_rot_stiffness * m)

  def _compute_torque_target(self) -> None:
    """``tau_target`` = anisotropic Cartesian impedance about ``x_ref``, via Jᵀ.

    Full rank-1 stiffness ``K(u) = k_precision·I − (k_precision − k_soft)·uuᵀ``:
    soft along the pull ``u``, stiff perpendicular — the rotated matrix a diagonal
    action stiffness cannot represent.  Applied about the *un-yielded* reference
    ``x_ref`` so the arm yields ``F_ext/k_soft`` along the push by construction.
    """
    data = self._env.sim.data
    x_ee = self.ee_pos_w()
    q_ee = quat_from_matrix(data.site_xmat[:, self._site_id])
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
    jacp, jacr = self._jacp_torch, self._jacr_torch
    qvel = data.qvel
    v_ee = torch.einsum("bij,bj->bi", jacp, qvel)
    w_ee = torch.einsum("bij,bj->bi", jacr, qvel)

    u = self.perturbation.direction  # (N, 3), unit while pushed, 0 idle
    uut = u.unsqueeze(-1) * u.unsqueeze(-2)  # (N, 3, 3)
    eye = torch.eye(3, device=self.device).expand(self.num_envs, 3, 3)
    kp, ks = self.cfg.k_precision, self.cfg.k_soft
    k_full = kp * eye - (kp - ks) * uut
    d_full = self._td_stiff * eye - (self._td_stiff - self._td_soft) * uut

    x_ref = self.path.position(self.s)
    f_pos = torch.einsum("bij,bj->bi", k_full, x_ref - x_ee) - torch.einsum(
      "bij,bj->bi", d_full, v_ee
    )
    _, rot_err = compute_pose_error(x_ee, q_ee, x_ee, self._lock_quat)
    f_rot = self.cfg.torque_rot_stiffness * rot_err - self._trot_d * w_ee

    jacp_arm = jacp[:, :, self._arm_dof_ids]
    jacr_arm = jacr[:, :, self._arm_dof_ids]
    tau = torch.einsum("bij,bi->bj", jacp_arm, f_pos) + torch.einsum(
      "bij,bi->bj", jacr_arm, f_rot
    )
    tau = tau + data.qfrc_bias[:, self._arm_dof_ids]
    qd_arm = qvel[:, self._arm_dof_ids]
    tau = tau + self._nullspace_torque(
      jacp_arm, jacr_arm, -self.cfg.torque_nullspace_damping * qd_arm
    )
    self.torque_target = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

  def _nullspace_torque(
    self, jacp_arm: torch.Tensor, jacr_arm: torch.Tensor, tau_secondary: torch.Tensor
  ) -> torch.Tensor:
    """Project a secondary joint torque into the task null-space (I − J⁺J)."""
    j = torch.cat([jacp_arm, jacr_arm], dim=1)  # (N, 6, nj)
    jjt = torch.einsum("bik,bjk->bij", j, j)
    eye6 = torch.eye(6, device=self.device).expand_as(jjt)
    jjt_inv = torch.linalg.inv(jjt + 1e-4 * eye6)
    j_tau = torch.einsum("bik,bk->bi", j, tau_secondary)
    proj = torch.einsum("bik,bi->bk", j, torch.einsum("bij,bj->bi", jjt_inv, j_tau))
    return tau_secondary - proj

  # -- public accessors (privileged: critic, reward, metrics only) --------------

  @property
  def command(self) -> torch.Tensor:
    """Teacher target ``[x_t (3), ẋ_t (3), F_finger_target (1)]`` (N, 7)."""
    return torch.cat([self.x_t, self.xd_t, self.finger_force_target.unsqueeze(-1)], -1)

  def ee_pos_w(self) -> torch.Tensor:
    return self._env.sim.data.site_xpos[:, self._site_id]

  def ee_vel_w(self) -> torch.Tensor:
    """Linear velocity *of the grasp site* (N, 3).

    Must be the site's velocity, not the wrist body's: the position target lives
    in grasp-site space, and the two differ by ω × r with r ~ 0.18 m, so pairing
    site position with wrist velocity would ask the student to track a velocity
    its position error cannot explain.
    """
    return self.robot.data.site_lin_vel_w[:, self._site_local]

  def x_ref(self) -> torch.Tensor:
    """The *un*-yielded nominal reference ``x_ref(s)`` (N, 3)."""
    return self.path.position(self.s)

  @property
  def goal_pos(self) -> torch.Tensor:
    """World-frame grasp target (N, 3) — the *task* goal, not the teacher target.

    Non-privileged: a deployed robot is told where the object is.
    """
    return self._grasp_target

  @property
  def grasp_flag(self) -> torch.Tensor:
    """Binary "grasp phase active" flag — allowed in the actor obs (§2.1): the
    real robot knows whether it has been told to close."""
    return (self.grasp_state != GRASP_OPEN).float()

  # -- CommandTerm API ----------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Start a new episode: sample the perturbation schedule and grasp target."""
    n = int(env_ids.numel())
    if n == 0:
      return
    self.s[env_ids] = 0.0
    self.s_dot[env_ids] = 0.0
    self.xd_t[env_ids] = 0.0
    self.deviation[env_ids] = 0.0
    self.grasp_state[env_ids] = GRASP_OPEN
    self._grasp_timer[env_ids] = 0.0
    self.finger_force_target[env_ids] = 0.0
    self._needs_seed[env_ids] = True
    self.perturbation.reset(env_ids)
    self._release_weld(env_ids)

    if self._multi_push:
      slot = torch.randint(0, len(self.cfg.push_body_names), (n,), device=self.device)
      self._push_slot[env_ids] = slot
      k = self._push_grade[slot]
      self._k_adm[env_ids] = k
      self._damp_adm[env_ids] = (
        2.0
        * self.cfg.admittance_damping_ratio
        * torch.sqrt(k * self.cfg.admittance_mass)
      )

    if self.cfg.object_name is None:
      lo = torch.tensor(self.cfg.grasp_box_min, device=self.device)
      hi = torch.tensor(self.cfg.grasp_box_max, device=self.device)
      origins = self._env.scene.env_origins[env_ids]
      self._grasp_target[env_ids] = (
        sample_uniform(lo, hi, (n, 3), self.device) + origins
      )

  def _seed(self) -> None:
    """One-shot per-episode init that needs the post-reset EE / object pose."""
    seed = self._needs_seed
    if not bool(seed.any()):
      return
    ids = seed.nonzero(as_tuple=False).squeeze(-1)
    x_ee = self.ee_pos_w()[ids]
    if self._object is not None:
      # Grasp the object where it actually settled after the reset event.
      grasp = self._object.data.root_link_pos_w[ids]
      self._grasp_target[ids] = grasp
    else:
      grasp = self._grasp_target[ids]
    self.path.set_waypoints(ids, x_ee, grasp, self.cfg.standoff, self.cfg.lift_height)
    self.x_t[ids] = x_ee
    self.xd_t[ids] = 0.0
    if self.cfg.emit_torque_target:
      # Lock the orientation target to the post-reset EE pose (like x_t).
      q_ee = quat_from_matrix(self._env.sim.data.site_xmat[:, self._site_id])
      self._lock_quat[ids] = q_ee[ids]
    self._needs_seed[ids] = False

  def compute(self, dt: float) -> None:
    # The base class passes dt to compute() but not down to _update_command, and
    # it is called with dt=0 from env.reset(). Using env.step_dt there would
    # advance the teacher a full step before the episode's first action, so
    # stash the real dt instead.
    self._dt = dt
    super().compute(dt)

  def _update_command(self) -> None:
    self._seed()
    dt = self._dt
    cfg = self.cfg

    x_ee = self.ee_pos_w()

    # --- perturbation: force first, everything downstream consumes it ---------
    # The spring anchors to the *push point*, which is per-env when the location
    # is randomized (gather that body's CoM), else the fixed wrist.
    if self._multi_push:
      arange = torch.arange(self.num_envs, device=self.device)
      body_per_env = self._push_body_id_t[self._push_slot]
      x_att = self.robot.data.body_com_pos_w[arange, body_per_env]
      v_att = self.robot.data.body_com_vel_w[arange, body_per_env, :3]
    else:
      x_att = self.robot.data.body_com_pos_w[:, self._attach_body]
      v_att = self.robot.data.body_com_vel_w[:, self._attach_body, :3]
    f_ext = self.perturbation.update(dt, self.s, x_att, v_att)
    torque = torch.zeros_like(f_ext)
    if self._multi_push:
      # One wrench slot per candidate body; each env's force lands only in its
      # sampled slot (write_external_wrench applies the same body list to all
      # envs, so a per-env mask selects the location).
      nb = len(self._push_body_ids)
      forces = torch.zeros(self.num_envs, nb, 3, device=self.device)
      forces[arange, self._push_slot] = f_ext
      self.robot.write_external_wrench_to_sim(
        forces, torch.zeros_like(forces), body_ids=self._push_body_ids
      )
    else:
      self.robot.write_external_wrench_to_sim(
        f_ext.unsqueeze(1), torque.unsqueeze(1), body_ids=[self._attach_body]
      )

    # --- §1.2 freeze s by tracking deviation ---------------------------------
    x_ref = self.path.position(self.s)
    self.deviation = torch.norm(x_ee - x_ref, dim=-1)
    gate = freeze_gate(self.deviation, cfg.freeze_onset, cfg.freeze_deviation)
    self.s_dot = self.path.s_rate * gate
    self.s = (self.s + self.s_dot * dt).clamp(max=1.0)

    # --- §1.3 integrate the reference impedance ------------------------------
    x_ref = self.path.position(self.s)
    xd_ref = self.path.velocity(self.s, self.s_dot)
    m = cfg.admittance_mass
    # Per-env stiffness/damping so compliance can be graded by push location.
    k = self._k_adm.unsqueeze(-1)
    d = self._damp_adm.unsqueeze(-1)
    acc = (f_ext - d * (self.xd_t - xd_ref) - k * (self.x_t - x_ref)) / m
    # Semi-implicit Euler: velocity first, then position from the new velocity.
    # Explicit Euler at this stiffness (omega_n ~ 12 rad/s) is stable at 100 Hz
    # but leaks energy in the wrong direction on the return transient.
    self.xd_t = self.xd_t + acc * dt
    self.x_t = self.x_t + self.xd_t * dt

    # --- Stiffness supervision target ----------------------------------------
    if cfg.supervise_stiffness:
      # Anisotropic (rank-1) softening: drop toward k_soft ONLY along the pull
      # direction u, hold k_precision on the two perpendicular axes. This is the
      # one softening that does not fight tracking -- the teacher's x_t yields
      # along u but stays on x_ref perpendicular, so staying stiff there is
      # exactly what the perpendicular tracking needs. (The earlier isotropic
      # target softened every axis and the perpendicular precision loss made the
      # dominant position reward refuse to soften at all; see the README.) u is
      # unit while pushed and exactly zero when idle -> k_precision everywhere.
      u = self.perturbation.direction
      self.k_target = cfg.k_precision + (cfg.k_soft - cfg.k_precision) * (u * u)

    # --- exp-2 compliant torque target ---------------------------------------
    if cfg.emit_torque_target:
      self._compute_torque_target()

    # --- §1.5 scripted grasp + §1.6 weld -------------------------------------
    self._update_grasp(dt)

  def _update_grasp(self, dt: float) -> None:
    cfg = self.cfg
    phase = self.path.phase(self.s)
    in_closure = phase >= PHASE_CLOSE

    # OPEN -> CLOSING: the path has reached closure and the arm is on-reference.
    start = (
      (self.grasp_state == GRASP_OPEN)
      & in_closure
      & (self.deviation < cfg.grasp_deviation_gate)
    )
    self.grasp_state = torch.where(
      start, torch.full_like(self.grasp_state, GRASP_CLOSING), self.grasp_state
    )
    self._grasp_timer = torch.where(
      start, torch.zeros_like(self._grasp_timer), self._grasp_timer
    )

    closing = self.grasp_state == GRASP_CLOSING
    self._grasp_timer = torch.where(
      closing | (self.grasp_state == GRASP_HOLDING),
      self._grasp_timer + dt,
      self._grasp_timer,
    )
    ramp = (self._grasp_timer / max(cfg.grasp_ramp_time, 1e-6)).clamp(0.0, 1.0)
    self.finger_force_target = torch.where(
      self.grasp_state == GRASP_OPEN,
      torch.zeros_like(ramp),
      ramp * cfg.grasp_force,
    )
    self.grasp_state = torch.where(
      closing & (self._grasp_timer >= cfg.grasp_ramp_time),
      torch.full_like(self.grasp_state, GRASP_HOLDING),
      self.grasp_state,
    )

    if self._weld_eq_id is not None:
      self._maybe_weld()

  # -- §1.6 weld ---------------------------------------------------------------

  def _maybe_weld(self) -> None:
    """Activate the object-to-hand weld for envs that just completed closure."""
    assert self._object is not None and self._object_body is not None
    x_obj = self._object.data.root_link_pos_w
    near = torch.norm(x_obj - self.ee_pos_w(), dim=-1) < self.cfg.weld_distance
    fresh = (self.grasp_state == GRASP_HOLDING) & (~self.welded) & near
    if not bool(fresh.any()):
      return
    ids = fresh.nonzero(as_tuple=False).squeeze(-1)

    data = self._env.sim.data
    x_g = data.xpos[ids, self._gripper_body]  # (n, 3)
    q_g = data.xquat[ids, self._gripper_body]  # (n, 4)
    x_o = data.xpos[ids, self._object_body]
    q_o = data.xquat[ids, self._object_body]

    # Freeze the *current* relative pose so the weld never teleports the object:
    #   anchor2 = R_g^T (x_o - x_g)   -> constrained point on the gripper
    #   anchor1 = 0                   -> constrained point on the object
    #   relpose = q_g^-1 * q_o
    r_g = _quat_to_mat(q_g)
    anchor2 = torch.einsum("bji,bj->bi", r_g, x_o - x_g)
    relpose = _quat_mul(_quat_conj(q_g), q_o)

    eq_data = self._env.sim.model.eq_data  # (nworld, neq, 11)
    eq_data[ids, self._weld_eq_id, 0:3] = 0.0
    eq_data[ids, self._weld_eq_id, 3:6] = anchor2
    eq_data[ids, self._weld_eq_id, 6:10] = relpose
    eq_data[ids, self._weld_eq_id, 10] = 1.0
    self._env.sim.data.eq_active[ids, self._weld_eq_id] = True
    self.welded[ids] = True

  def _release_weld(self, env_ids: torch.Tensor) -> None:
    if self._weld_eq_id is None:
      return
    self._env.sim.data.eq_active[env_ids, self._weld_eq_id] = False
    self.welded[env_ids] = False

  # -- metrics / viz ------------------------------------------------------------

  def _update_metrics(self) -> None:
    # Runs before _update_command inside compute(), so seed here too or the
    # first post-reset sample compares the new pose against the old target.
    self._seed()
    self.metrics["path_parameter"] = self.s
    self.metrics["tracking_error"] = torch.norm(self.ee_pos_w() - self.x_t, dim=-1)
    self.metrics["perturbation_force"] = torch.norm(self.perturbation.force, dim=-1)
    self.metrics["grasped"] = (self.grasp_state == GRASP_HOLDING).float()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    viz = self.cfg.viz
    x_ref = self.path.position(self.s)
    for i in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.x_t[i].cpu().numpy(),
        radius=viz.marker_radius,
        color=viz.target_color,
        label=f"teacher_target_{i}",
      )
      visualizer.add_sphere(
        center=x_ref[i].cpu().numpy(),
        radius=viz.marker_radius * 0.8,
        color=viz.reference_color,
        label=f"teacher_reference_{i}",
      )
      if not bool(self.perturbation.active[i]):
        continue
      visualizer.add_sphere(
        center=self.perturbation.anchor_pos[i].cpu().numpy(),
        radius=viz.marker_radius,
        color=viz.hand_color,
        label=f"human_anchor_{i}",
      )
      f = self.perturbation.force[i]
      start = self.robot.data.body_com_pos_w[i, self._attach_body].cpu().numpy()
      visualizer.add_arrow(
        start=start,
        end=start + f.cpu().numpy() * viz.force_scale,
        color=viz.force_color,
        width=viz.force_width,
        label=f"human_force_{i}",
      )


def freeze_gate(
  deviation: torch.Tensor, onset: float, full_freeze: float
) -> torch.Tensor:
  """Scale factor on ``ṡ``: 1 inside the tube, 0 at ``full_freeze`` (spec §1.2).

  Smoothstep rather than a hard switch, so ``ṡ`` is C1 in the deviation and the
  target velocity the student tracks has no kink at the tube boundary.
  """
  u = ((deviation - onset) / max(full_freeze - onset, 1e-6)).clamp(0.0, 1.0)
  return 1.0 - u * u * (3.0 - 2.0 * u)


# --- quaternion helpers (wxyz), batched -------------------------------------


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dim=-1,
  )


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    [
      1 - 2 * (y * y + z * z),
      2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z - w * x),
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
    ],
    dim=-1,
  ).view(*q.shape[:-1], 3, 3)


def add_grasp_weld(spec: mujoco.MjSpec, robot_name: str, object_name: str) -> None:
  """Add the (initially inactive) object-to-hand weld to the assembled scene.

  Registered as ``SceneCfg.spec_fn``; the equality has to be created at model
  build time because MuJoCo cannot add constraints to a compiled model.  The
  teacher activates it per-env and writes the relative pose at grasp time.
  """
  eq = spec.add_equality()
  eq.name = WELD_EQUALITY_NAME
  eq.type = mujoco.mjtEq.mjEQ_WELD
  eq.objtype = mujoco.mjtObj.mjOBJ_BODY
  eq.name1 = f"{robot_name}/gripper_base"
  eq.name2 = f"{object_name}/{object_name}"
  eq.active = False
  # [anchor1(3), anchor2(3), relpose quat(4), torquescale]; the teacher
  # overwrites anchor2 and the quaternion per-env at grasp time.
  eq.data[:] = [0.0] * 6 + [1.0, 0.0, 0.0, 0.0] + [1.0]
  # A slightly soft weld: a hard constraint at 1 kHz fights the finger contacts,
  # which are still present (we weld *in addition to* the grasp, not instead).
  eq.solref[:] = [0.005, 1.0]
