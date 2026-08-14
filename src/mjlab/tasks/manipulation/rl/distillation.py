"""DAgger that teaches an ACHIEVABLE action and charges the student for chatter.

WHY THIS EXISTS. The anti-chatter ablation fixed the TEACHER and the fix did not
survive distillation. Measured on g1_satpen (jobs 64722 / 64737, carry phase):

                     dwell   |acc|   tcp jerk   requested/published
  teacher g1_satpen   0.73     0.9          7                  1.5x
  student s_g1_satpen 0.07    32.0        117                  3.2x
  student s_b0        0.11    37.4        107                  1.0x (no limiter)

The saturation penalty that cured the teacher is a REWARD term, and
`Distillation.update` optimises a behaviour-cloning loss only -- it never reads
`storage.rewards`. So the penalty has exactly zero gradient on the student, and
the student's published command sits pinned at the cap from the MEDIAN step
upward. Two independent causes, and this class attacks both:

  RELABELLING (the systematic half). RSL-RL regresses the student onto
  `teacher(obs)`, which is the teacher's PRE-limiter request. The teacher asks
  for 1.5x what its own limiter publishes, so the student is being fitted to a
  target the servo provably never receives. `_achievable` maps the request
  through the limiter first, so the regression target is inside the reachable
  set by construction.

  This is EXACTLY BEHAVIOUR-PRESERVING, which is why it needs no retune of
  anything else: executing the request and executing the relabelled action put
  the same number on the servo. Within one control step the target is fixed, so
  the limiter walks the command toward it at the cap and stops at
  `cmd + clamp(target - cmd, +-allowance)` -- which is precisely what the
  relabelled action asks for directly. Only the LABEL changes; the rollout, and
  therefore the state distribution DAgger collects, is untouched.

  PENALTIES (the noise-driven half, and the larger one). A BC loss is
  per-timestep: it prices `|a_t - a*_t|` at each t independently and says
  nothing about how the student's output moves between t and t+1. But the
  requested RATE is `d(a)/dt`, so a residual that is merely INDEPENDENT across
  frames -- which perception error is -- produces rate out of nothing. At arm
  scale 0.40 rad and a 20 ms step, an error of 0.05 action units alternating
  sign is 0.05 * 0.40 / 0.02 = 1.0 rad/s, i.e. exactly the arm's cap. A student
  can therefore be accurate by every BC measure and still saturate on every
  step, which is what s_g1_satpen does.

  `saturation_weight` and `action_rate_weight` add the two quantities the
  teacher was charged for, in their differentiable form. Only ACTION-space terms
  can go here: `action_saturation` and `action_rate_l2` are functions of the
  action itself, whereas `joint_vel_hinge`, `lift` and `grasp` read simulator
  state and there is no gradient path back through MuJoCo.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

from mjlab.rl.distillation import DaggerDistillation
from mjlab.tasks.manipulation.mdp.actions import SmoothedJointPositionAction


class SmoothedDaggerDistillation(DaggerDistillation):
  """DAgger with an achievable regression target and action-space penalties.

  Args:
    relabel_achievable: Regress onto the teacher's request AFTER the limiter has
      clipped it, rather than the raw request. Behaviour-preserving; see module
      docstring.
    saturation_weight: Weight on the student's own overdrive,
      ``clamp(|target - cmd| / allowance - 1, 0)`` averaged over joints. 0
      disables. Dimensionless, so a rate-limit curriculum does not rescale it.
    action_rate_weight: Weight on ``||a_t - a_{t-1}||^2``, masked at episode
      boundaries. 0 disables.
    action_name: The action term carrying the limiter.

  All three are no-ops unless the task's action term is a
  `SmoothedJointPositionAction`; on an unlimited task this class is the stock
  DAgger, so the same run script works for both.
  """

  def __init__(
    self,
    *args,
    relabel_achievable: bool = True,
    saturation_weight: float = 0.0,
    action_rate_weight: float = 0.0,
    penalty_ramp_start: int = 0,
    penalty_ramp_end: int = 0,
    std_end: float = 0.0,
    std_anneal_start: int = 0,
    std_anneal_end: int = 0,
    action_name: str = "joint_pos",
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.relabel_achievable = relabel_achievable
    self.saturation_weight = saturation_weight
    self.action_rate_weight = action_rate_weight
    self.penalty_ramp_start = penalty_ramp_start
    self.penalty_ramp_end = penalty_ramp_end
    self.std_end = std_end
    self.std_anneal_start = std_anneal_start
    self.std_anneal_end = std_anneal_end
    self._std_start: float | None = None
    self._action_name = action_name
    self._term: SmoothedJointPositionAction | None = None
    # The published command per rollout timestep. The saturation penalty is a
    # statement about where the command WAS when the action was chosen, and the
    # rollout storage has nowhere to put that, so it is kept alongside.
    self._cmd: torch.Tensor | None = None
    self._step = 0

  @property
  def penalty_scale(self) -> float:
    """Fraction of the configured penalty weights charged right now.

    MEASURED, not guessed. The un-ramped run (job 65248, weights 0.05 / 2.0 from
    iteration 0) was healthy for 1400 iterations -- saturation driven to 0.0019,
    20x below the relabel-only arm, success a flat 1.0 -- and came apart the
    moment ``beta`` hit 0 at 1500, oscillating 0.15 / 0.86 / 0.17 / 0.00 / 0.89
    with the behaviour loss 3.4x worse and saturation reflating to 0.112, i.e.
    worse than charging nothing.

    So the failure is TIMING, not magnitude: those weights are right while the
    teacher is still driving part of the rollout and too much the first time the
    student meets its own error distribution alone. Holding the penalties at
    zero until the DAgger handoff has finished and ramping in afterwards is the
    same late-ramp shape every other penalty on this task needs, and for the
    same reason -- see `action_rate_limit_curriculum` and the
    `action_saturation_weight` schedule in env_cfgs.
    """
    if self.penalty_ramp_end <= self.penalty_ramp_start:
      return 1.0
    span = self.penalty_ramp_end - self.penalty_ramp_start
    t = (self.num_updates - self.penalty_ramp_start) / span
    return float(min(1.0, max(0.0, t)))

  def _anneal_std(self) -> float:
    """Shrink the student's EXECUTED exploration noise once DAgger has handed off.

    `student_std` is fixed at 0.02 and gets no gradient (the loss regresses the
    MEAN), but DAgger executes it, so it is injected command rate: at arm scale
    0.40 rad and a 20 ms step, 0.02 action units is 0.4 rad/s, i.e. 40% of the
    arm's 1.0 rad/s cap, on every step of every rollout. Its job is to widen the
    state distribution the teacher labels, which is worth paying for while beta
    is still mixing and worth much less once the student is already holding its
    own distribution.

    The cost is measured. `flexiv_two_finger_distill_runner_cfg` records a sweep
    of the EXECUTED std against lift rate: 0.0 -> 98.6%, 0.01 -> 98.6%,
    0.02 -> 97.8%, 0.05 -> 42.5%. So the value in use is already the first one
    that costs anything.

    Returns the std actually in force, for logging.
    """
    dist = self._raw_student.distribution
    param = getattr(dist, "std_param", None)
    if param is None or self.std_anneal_end <= self.std_anneal_start:
      return float(param.mean().item()) if param is not None else 0.0
    if self._std_start is None:
      self._std_start = float(param.mean().item())
    span = self.std_anneal_end - self.std_anneal_start
    t = min(1.0, max(0.0, (self.num_updates - self.std_anneal_start) / span))
    std = self._std_start + t * (self.std_end - self._std_start)
    param.data.fill_(std)
    return std

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    # `Distillation.construct_algorithm` re-resolves `class_name` from the cfg,
    # which points here, so this returns an instance of THIS class. Overridden
    # only because the base signature drops `env` before the algorithm sees it
    # and the limiter lives on the env.
    alg = DaggerDistillation.construct_algorithm(obs, env, cfg, device)
    assert isinstance(alg, SmoothedDaggerDistillation)
    alg.bind_env(env)
    return alg

  def bind_env(self, env) -> None:
    """Attach the limiter this loss reasons about. Silent no-op without one."""
    term = env.unwrapped.action_manager.get_term(self._action_name)
    if not isinstance(term, SmoothedJointPositionAction):
      if self.saturation_weight or self.relabel_achievable:
        raise TypeError(
          f"relabelling and the saturation penalty need a "
          f"SmoothedJointPositionAction on '{self._action_name}', got "
          f"{type(term).__name__}. Set relabel_achievable=False and "
          f"saturation_weight=0 to distil an unlimited task with this class."
        )
      return
    self._term = term
    self._cmd = torch.zeros(
      self.storage.num_transitions_per_env,
      term.num_envs,
      term.action_dim,
      device=self.device,
    )

  def _achievable(self, request: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """The action whose target the limiter would pass through unchanged.

    ``request`` is always read in the ABSOLUTE parameterization, because that is
    what the teacher was trained in and its weights are frozen. Only the answer
    changes form.
    """
    assert self._term is not None
    scale, offset = self._term.scale, self._term.offset
    allowance = self._term.step_allowance
    target = offset + scale * request
    reach = cmd + torch.clamp(target - cmd, min=-allowance, max=allowance)
    if self._term.cfg_incremental:
      return (reach - cmd) / self._term.incremental_scale
    return (reach - offset) / scale

  def act(self, obs: TensorDict) -> torch.Tensor:
    if self._term is None:
      return super().act(obs)
    cmd = self._term.command.clone()

    # Deliberately NOT `DaggerDistillation.act`: the label has to be rewritten
    # BEFORE the beta mixture picks what to execute. In the absolute form that
    # is cosmetic -- the raw request and the relabelled action put the same
    # number on the servo -- but in the incremental form the teacher's raw
    # output means something else entirely, and executing it unconverted would
    # drive the arm somewhere unrelated to what the teacher asked for.
    Distillation.act(self, obs)
    request = self.transition.privileged_actions
    actions = self.transition.actions
    assert request is not None and actions is not None
    if self.relabel_achievable or self._term.cfg_incremental:
      request = self._achievable(request, cmd)
      self.transition.privileged_actions = request

    if self._cmd is not None and self._step < self._cmd.shape[0]:
      self._cmd[self._step].copy_(cmd)
    self._step += 1

    beta = self.beta
    if beta <= 0.0:
      return actions
    take_teacher = torch.rand(actions.shape[0], device=actions.device) < beta
    return torch.where(take_teacher.unsqueeze(-1), request, actions)

  def update(self) -> dict[str, float]:
    """`Distillation.update` plus the two action-space penalties.

    Copied rather than wrapped because the penalties need the per-timestep
    batch, and the base class exposes no hook inside its generator loop. The
    behaviour-cloning half is unchanged; if RSL-RL's version moves, this is the
    method to re-sync.
    """
    self.num_updates += 1
    sums = {"behavior": 0.0, "saturation": 0.0, "action_rate": 0.0}
    scale = self.penalty_scale
    sat_weight = self.saturation_weight * scale
    rate_weight = self.action_rate_weight * scale
    std_now = self._anneal_std()
    loss = 0
    cnt = 0

    for _ in range(self.num_learning_epochs):
      self.student.reset(hidden_state=self.last_hidden_states[0])
      self.teacher.reset(hidden_state=self.last_hidden_states[1])
      self.student.detach_hidden_state()
      # Per epoch, because the generator restarts at timestep 0 and there is no
      # predecessor for the first batch of a pass.
      prev_actions: torch.Tensor | None = None
      prev_alive: torch.Tensor | None = None

      for step, batch in enumerate(self.storage.generator()):
        dones = batch.dones
        assert dones is not None
        actions = self.student(batch.observations)

        behavior_loss = self.loss_fn(actions, batch.privileged_actions)
        sums["behavior"] += behavior_loss.item()
        step_loss = behavior_loss

        # Both penalties are COMPUTED whenever they can be and only ADDED when
        # weighted, so a zero-weight run still reports what a weighted one would
        # be charging. Their scale against `behavior` is what sizes the weights,
        # and there is no way to guess it from outside a run.
        if self._term is not None:
          cmd = self._cmd[step]  # type: ignore[index]
          # In the incremental form this is identically 0 whenever
          # `incremental_scale` is one allowance -- overdrive is not
          # expressible -- so it is a check on that claim rather than a
          # penalty. Computed the same way regardless so the column stays
          # comparable across the two parameterizations.
          target = (
            cmd + self._term.incremental_scale * actions
            if self._term.cfg_incremental
            else self._term.offset + self._term.scale * actions
          )
          overdrive = (
            (target - cmd).abs() / self._term.step_allowance - 1.0
          ).clamp_min(0.0)
          sat = overdrive.mean()
          sums["saturation"] += sat.item()
          if sat_weight:
            step_loss = step_loss + sat_weight * sat

        if prev_actions is not None:
          # `prev_actions` is detached: this prices how far the student MOVED
          # from where it was, and letting the gradient reach backwards as well
          # would also try to walk the graph through an optimizer step that has
          # already run.
          rate = (((actions - prev_actions) * prev_alive) ** 2).mean()
          sums["action_rate"] += rate.item()
          if rate_weight:
            step_loss = step_loss + rate_weight * rate

        # A reset breaks the sequence: the next action follows a teleport, not
        # the previous command, and charging for that would price the reset.
        prev_actions = actions.detach()
        prev_alive = 1.0 - dones.float().view(-1, 1)

        loss = loss + step_loss
        cnt += 1

        if cnt % self.gradient_length == 0:
          self.optimizer.zero_grad()
          loss.backward()
          if self.is_multi_gpu:
            self.reduce_parameters()
          if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
              self.student.parameters(), self.max_grad_norm
            )
          self.optimizer.step()
          self.student.detach_hidden_state()
          loss = 0

        self.student.reset(dones.view(-1))
        self.teacher.reset(dones.view(-1))
        self.student.detach_hidden_state(dones.view(-1))

    self.storage.clear()
    self._step = 0
    self.last_hidden_states = (
      self.student.get_hidden_state(),
      self.teacher.get_hidden_state(),
    )
    self.student.detach_hidden_state()

    out = {k: v / max(cnt, 1) for k, v in sums.items()}
    out["penalty_scale"] = scale
    out["exec_std"] = std_now
    return out
