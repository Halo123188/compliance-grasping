"""Action terms specific to the two-finger grasp task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class WristAlignedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position control with the wrist roll solved in closed form.

  Squaring an antipodal jaw to a square prism's face is a one-line geometry
  problem, and six RL variants failed to learn it: four reward-side
  (AlignTight/Lift/Touch/Curr, all beaten by the ungated baseline) and two
  exploration-side (`free_wrist`, `wrist_rand` -- both raised joint7's action std
  from 3.5 deg to 11-20 deg and both left corr(cube yaw, joint7) at zero, +0.026
  and -0.060, while destroying the task). Meanwhile the payoff was never large:
  locking cube yaw square to a frozen wrist moves success 78.1% -> 89.1%.

  So compute it instead of learning it. `scripts/diag_wrist_sign.py` measures
  d(jaw yaw)/d(joint7) = -1.000 rad/rad exactly, so the target that squares the
  jaw RIGHT NOW is `q7 + fold(jaw_yaw - object_yaw)`, re-solved from measured
  state on every physics substep -- not an integrator, and it stays correct while
  the other arm joints move the hand around.

  The policy keeps a trim action on the same joint; give it a small `scale` (the
  0-22.5 deg band where alignment provably does not matter) so it can adjust
  without undoing the solution.

  This reads the object's pose, so it is only legitimate for the state-based
  task. The vision policy has to estimate yaw from the image and cannot use it.
  """

  wrist_joint: str = "joint7"
  object_name: str = "cube"
  pad_sites: tuple[str, str] = ("left_pad", "right_pad")
  gain: float = 1.0
  symmetry: float = math.pi / 2  # square cube + 180-deg-symmetric jaw

  def build(self, env: ManagerBasedRlEnv) -> WristAlignedJointPositionAction:
    return WristAlignedJointPositionAction(self, env)


class WristAlignedJointPositionAction(JointPositionAction):
  # No `cfg` re-declaration, for the same reason as
  # `SmoothedJointPositionAction` below: what this class needs from the cfg is
  # unpacked in __init__, and narrowing the attribute is what makes pyright
  # complain about overriding BaseAction's.

  def __init__(self, cfg: WristAlignedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._scene = env.scene
    self._object_name = cfg.object_name
    self._symmetry = cfg.symmetry
    self._gain = cfg.gain

    names = self.target_names
    assert cfg.wrist_joint in names, f"{cfg.wrist_joint} not in action dims {names}"
    self._wrist_col = names.index(cfg.wrist_joint)
    self._wrist_qid = list(self._entity.joint_names).index(cfg.wrist_joint)

    sites = list(self._entity.site_names)
    self._pad_ids = [sites.index(s) for s in cfg.pad_sites]

    offset = self._offset
    self._wrist_default = (
      offset[:, self._wrist_col] if isinstance(offset, torch.Tensor) else offset
    )

  def _yaw_error(self) -> torch.Tensor:
    """Signed jaw-vs-object yaw error, folded into +-symmetry/2."""
    pads = self._entity.data.site_pos_w[:, self._pad_ids]
    axis = pads[:, 1] - pads[:, 0]
    jaw = torch.atan2(axis[:, 1], axis[:, 0])

    obj: Entity = self._scene[self._object_name]
    q = obj.data.root_link_quat_w
    obj_yaw = torch.atan2(
      2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )

    sym = self._symmetry
    err = jaw - obj_yaw
    return err - sym * torch.round(err / sym)

  def apply_actions(self) -> None:
    # `JointPositionAction` sets `_offset` to the default joint pose, so the
    # wrist column of `_processed_actions` is `default_q7 + trim`. Strip the
    # default back off and replace it with the closed-form target, keeping the
    # policy's trim: d(jaw)/d(q7) = -1, so adding the error to q7 cancels it.
    target = self._processed_actions.clone()
    trim = target[:, self._wrist_col] - self._wrist_default
    q7 = self._entity.data.joint_pos[:, self._wrist_qid]
    target[:, self._wrist_col] = q7 + self._gain * self._yaw_error() + trim

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )


@dataclass(kw_only=True)
class SmoothedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position control with a bounded rate of change on the COMMAND.

  The stock term is memoryless: every control step it publishes
  ``default + scale * action`` and nothing looks at what it published last
  time, so a one-step swing of the network's output is handed to the servo as a
  step. Measured on the DR teacher (``scripts/wide_speed_audit.py``, run 60866)
  that is not hypothetical -- the commanded target moves at up to 24.6 rad/s on
  the arm and 63.6 rad/s on a finger, i.e. joint2 jumps 0.49 rad inside one
  20 ms step, and the arm's tool point reaches 2.27 m/s.

  This term keeps the published command as state and bounds how fast it may
  move. Two independent shapes, either or both:

    rate_limit  a hard slew cap in rad/s. The command may not move faster than
                this, ever, which is a guarantee rather than a tendency and is
                the only reason to prefer this over a velocity penalty. The
                same audit says why a penalty struggles: the whole motion is
                0.6 s of a 20 s episode, so a per-step charge is diluted 20:1
                by the stationary tail, and the weight that would bite the
                burst is the weight that already regressed this task into
                "pinch and press down" (see the joint_vel_hinge note in
                env_cfgs).
    ema_tau     a first-order low-pass time constant in SECONDS. Softer: it
                shaves the peak of a step (a 0.8 rad jump moves 0.16 rad on the
                first control step at tau = 0.09 s) but bounds nothing in
                steady state.

  BOTH ARE IN PHYSICAL UNITS, NOT PER-STEP UNITS, and that is load-bearing.
  ``apply_actions`` runs once per PHYSICS SUBSTEP -- four times per control step
  at decimation=4 -- so a per-step delta accumulated here would be four times
  the intended limit, silently and without an error. Expressing the limit in
  rad/s and multiplying by ``physics_dt`` makes it correct at any decimation,
  gives the servo a 200 Hz ramp instead of a 50 Hz staircase, and means the
  deployment mirror stays valid at whatever rate the robot host publishes at.

  NO OBSERVATION TERM IS ADDED FOR THE INTERNAL COMMAND, deliberately. It is
  hidden state in principle, but it is nearly recoverable from what the policy
  already sees: the servo is stiff (kp 186..673) and gravity-compensated, so in
  free space the tracking error needed to move at 1 rad/s is on the order of
  0.01 rad, and the policy observes both ``joint_pos`` and its own last action
  -- from which the backlog ``target - q_cmd`` follows to within that error.
  Keeping the observation at 43 dims is what lets a pre-limit checkpoint be
  warm-started into a limited run at all. If an arm fails to LEARN rather than
  merely scoring worse, adding ``q_cmd - default`` to the actor group is the
  first thing to try.

  Not combinable with ``WristAlignedJointPositionActionCfg``: that one re-solves
  joint7 from measured state on every substep, which is an unbounded command by
  construction. ``solve_wrist`` is off on every task that uses this.
  """

  rate_limit: float | dict[str, float] | None = None
  """Slew cap in rad/s, as a float or a dict of actuator-name patterns."""

  ema_tau: float | dict[str, float] | None = None
  """Low-pass time constant in seconds. 0 (the default) is no filtering."""

  incremental: bool = False
  """Treat the action as a STEP from the current command, not an absolute pose.

  ``target = command + incremental_scale * action``, and ``scale`` / ``offset``
  are then NOT used to drive the arm. They are deliberately left alone anyway,
  because they are how an absolute-form teacher's output is still interpreted
  when its labels are converted -- see
  `mjlab.tasks.manipulation.rl.distillation`. A teacher stays valid across this
  switch; it is re-read, not re-fitted.

  With ``incremental_scale`` at its default, saturation stops being expressible
  at all: a full-range action is exactly one step of the limiter.
  """

  incremental_scale: float | dict[str, float] | None = None
  """Per-control-step travel of a full-range incremental action, rad.

  ``None`` means one limiter allowance (``rate_limit * step_dt``), which is the
  setting that makes overdrive unrepresentable. Larger re-admits saturation;
  smaller makes the cap unreachable and slows the arm below it.

  This is the whole point of the incremental form. Under ``relabel_achievable``
  every label lies within one allowance of the command -- 0.02 rad against the
  arm's 0.40 rad ``scale`` -- so the absolute parameterization spends 95% of its
  output range on targets the limiter would clip, and regresses onto a band 5%
  as wide as the space it predicts in. Here that band IS the range.
  """

  def build(self, env: ManagerBasedRlEnv) -> SmoothedJointPositionAction:
    return SmoothedJointPositionAction(self, env)


class SmoothedJointPositionAction(JointPositionAction):
  # No `cfg` re-declaration: everything from it is unpacked into tensors in
  # __init__, and narrowing the attribute is what makes pyright complain about
  # overriding BaseAction's.

  def __init__(self, cfg: SmoothedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # Unpacked rather than read off `self.cfg`, for the same reason the rest of
    # this class does: narrowing the `cfg` attribute is what makes pyright
    # complain about overriding BaseAction's.
    self.cfg_incremental = cfg.incremental

    self._cmd = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    # A scalar the curriculum multiplies the whole limit vector by, so a
    # schedule is one number rather than a per-joint table (see
    # `action_rate_limit_curriculum`).
    self.rate_scale = 1.0

    # How hard the policy is leaning on the limiter, accumulated across the
    # substeps of one control step so a reward term can charge for it. See
    # `saturation` for what the number means and why it is worth charging for.
    self._sat = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._sat_substeps = 0

    # Defaults are the identity: no cap, no filtering.
    self._rate = self._per_dim(cfg.rate_limit, float("inf"))
    self._tau = self._per_dim(cfg.ema_tau, 0.0)
    # One limiter allowance by default, which is what makes overdrive
    # unrepresentable. Uncapped joints would get `inf` from that, so they fall
    # back to the absolute `scale` and behave as they always did.
    if cfg.incremental_scale is None:
      self._incr = self._rate * self._env.step_dt
      self._incr = torch.where(torch.isfinite(self._incr), self._incr, 1.0)
    else:
      self._incr = self._per_dim(cfg.incremental_scale, 1.0)
    if cfg.incremental and (self._incr <= 0).any():
      raise ValueError(f"incremental_scale must be positive, got {self._incr}")
    if (self._rate <= 0).any():
      raise ValueError(f"rate_limit must be positive, got {cfg.rate_limit}")
    if (self._tau < 0).any():
      raise ValueError(f"ema_tau must be non-negative, got {cfg.ema_tau}")

  def _per_dim(
    self, value: float | dict[str, float] | None, default: float
  ) -> torch.Tensor:
    """Expand a float / name-pattern dict / None over the action dimensions."""
    out = torch.full((self.action_dim,), default, device=self.device)
    if value is None:
      return out
    if isinstance(value, (float, int)):
      return torch.full_like(out, float(value))
    index_list, _, value_list = resolve_matching_names_values(value, self._target_names)
    out[index_list] = torch.tensor(value_list, device=self.device, dtype=out.dtype)
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    # Start from where the joints ACTUALLY are, not from the default pose. The
    # reset events place the arm (`reset_robot_joints` jitters it), so seeding
    # with the default would open a gap the limiter then has to ramp across --
    # a scripted move at the start of every episode that no action asked for.
    self._cmd[env_ids] = self._entity.data.joint_pos[env_ids][:, self._target_ids]
    self._sat[env_ids] = 0.0

  @property
  def command(self) -> torch.Tensor:
    """The command actually published to the servo, (num_envs, action_dim).

    This is the limiter's internal state, and the policy does not otherwise
    observe it. Expose it so `smoothed_action_command` can put it in the
    observation group.
    """
    return self._cmd

  @property
  def incremental_scale(self) -> torch.Tensor:
    """Per-control-step travel of a full-range incremental action, rad."""
    return self._incr

  @property
  def step_allowance(self) -> torch.Tensor:
    """Most the command can move in ONE CONTROL STEP, ``(action_dim,)``, rad.

    ``apply_actions`` clamps every PHYSICS substep to ``rate * rate_scale *
    physics_dt`` and the target is held fixed across the substeps of one control
    step, so the reachable total is that times ``decimation`` -- i.e. the same
    product taken against ``step_dt``. Uncapped joints report ``inf``.

    Exposed because a distillation loss has to know which requests the limiter
    can actually execute in order to stop teaching the ones it cannot; see
    `mjlab.tasks.manipulation.rl.distillation`.
    """
    if (self._tau > 0).any():
      raise NotImplementedError(
        "step_allowance is exact only for a pure rate limit. This term also "
        "low-passes, which makes the reachable set depend on the whole command "
        "history rather than on one step's budget."
      )
    return self._rate * (self.rate_scale * self._env.step_dt)

  @property
  def saturation(self) -> torch.Tensor:
    """Per-joint overdrive, averaged over the last control step's substeps.

    ``clamp_min(|target - cmd| / step - 1, 0)``: 0 while the limiter can follow
    the request, and N when the policy is asking for N+1 times what the cap can
    deliver. Dimensionless on purpose -- it is the same number whatever the cap
    is, so a curriculum that tightens the cap does not silently rescale the
    reward that charges for it.

    Worth charging for because saturation is a DEGENERACY, not just waste: once
    the limiter is pinned, the published motion depends only on the SIGN of
    ``target - cmd``, so magnitude stops affecting the rollout and stops
    receiving gradient. The output is then free to drift outward, which deepens
    the saturation, and the joint ends up in bang-bang -- moving at the cap and
    reversing whenever the network output crosses. Measured on the SlewCurr
    teacher (job 61823) that is what happened: actions ran to 27.0 against a
    trained range of about |5|, and the resulting chatter is visible in
    `scripts/render_student.py` clips.

    Joints with no cap have ``step = inf`` and so report exactly 0.
    """
    if self._sat_substeps == 0:
      return torch.zeros_like(self._sat)
    return self._sat / self._sat_substeps

  def process_actions(self, actions: torch.Tensor) -> None:
    # One control step's worth of saturation starts here: the reward manager
    # reads `saturation` once per control step, but `apply_actions` runs once
    # per PHYSICS substep, so the accumulator has to be cleared on the control
    # boundary rather than inside the substep loop.
    super().process_actions(actions)
    self._sat.zero_()
    self._sat_substeps = 0

  def apply_actions(self) -> None:
    dt = self._env.physics_dt
    target = self._processed_actions

    if self.cfg_incremental:
      # INCREMENTAL: the action is a step to take from where the command
      # already is, not a place to be. Built from `raw_action` rather than from
      # `_processed_actions`, because `scale`/`offset` still carry the ABSOLUTE
      # meaning an existing teacher's output is read with.
      #
      # The command is state, so this integrates: the reachable workspace is
      # not bounded by `incremental_scale` the way the absolute form's is by
      # `scale`.
      target = self._cmd + self._incr * self.raw_action

    if (self._tau > 0).any():
      # alpha = exp(-dt/tau) per substep, so the filter is the same filter at
      # any decimation. tau = 0 gives alpha = 0, i.e. pass-through.
      alpha = torch.where(
        self._tau > 0, torch.exp(-dt / self._tau.clamp_min(1e-9)), 0.0
      )
      target = alpha * self._cmd + (1.0 - alpha) * target

    step = self._rate * (self.rate_scale * dt)
    backlog = target - self._cmd
    # Measure BEFORE clamping: after it, the discarded part is gone.
    self._sat += (backlog.abs() / step - 1.0).clamp_min(0.0)
    self._sat_substeps += 1
    self._cmd = self._cmd + backlog.clamp(-step, step)

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._cmd - encoder_bias, joint_ids=self._target_ids
    )


class action_rate_limit_curriculum:
  """Loosen or tighten a `SmoothedJointPositionAction`'s slew cap over training.

  Scales the whole per-joint limit vector by one number, so a schedule reads as
  "start 3x looser, end at the configured value" rather than as a per-joint
  table that has to be kept in sync with the term's own config.

  The reason to schedule it at all is the same reason `align_std_schedule`
  exists: a constraint that is correct at the end can be the thing that stops
  the behaviour being discovered at the start. Grasping on this task emerges
  from exploration around iteration 750-2000, and a policy that cannot move
  fast enough to reach the cube before it has learned WHY it is reaching gets
  no gradient at all -- the same shape of failure as the 15 deg alignment
  kernel, which lost 13.5 points by being right too early.

  Example::

    CurriculumTermCfg(
      func=manipulation_mdp.action_rate_limit_curriculum,
      params={"stages": [{"step": 0, "scale": 3.0}, {"step": 60000, "scale": 1.0}]},
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    term = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
    if not isinstance(term, SmoothedJointPositionAction):
      raise TypeError(
        f"action_rate_limit_curriculum needs a SmoothedJointPositionAction, "
        f"got {type(term).__name__}"
      )
    self._term = term
    self._stages = sorted(cfg.params["stages"], key=lambda s: s["step"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[dict],
    action_name: str = "joint_pos",
  ) -> dict[str, torch.Tensor]:
    del env_ids, stages, action_name
    scale = self._term.rate_scale
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        scale = float(stage["scale"])
    self._term.rate_scale = scale
    return {"rate_scale": torch.tensor(scale)}
