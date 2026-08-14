"""PPO fine-tuning of an already-distilled camera student.

WHY THIS EXISTS. Distillation leaves the student accurate per frame and rough in
time, and the objective is structurally unable to see the difference. Measured on
p3_FaceLevel and its student (jobs 67452 / 67455, carry phase):

                      dwell   |acc|   tcp jerk   rev/s
  teacher FaceLevel    0.65     0.7          7    0.29
  student s_facelevel  0.18     7.6         30   10.36

The student's final behaviour-cloning loss is 0.0048, i.e. an RMS residual of
0.069 action units = 0.028 rad = 1.6 deg per joint. That residual is invisible in
a position trace and enormous in a rate one: fresh every frame, it is
0.028 * sqrt(2) / 0.02 = 2.0 rad/s against a 1.0 rad/s cap. The chatter is the
1/dt image of an error the BC loss already considers small.

And the BC loss cannot price it. It is per-timestep, so a student that sits a
steady 1.6 deg off and one that alternates +-1.6 deg score IDENTICALLY. This is
not a convergence problem; more of the same objective cannot fix it.

A REWARD can. `joint_vel_hinge`, `action_saturation` and `action_rate_l2` are
already in this env's reward manager -- all ten terms are computed on every
distillation step and thrown away, because `Distillation.update` never reads
`storage.rewards`. They are exactly what made the TEACHER smooth. So the fix is
not a new mechanism, it is running the mechanism that already worked on the
policy that never got it.

WHY THIS IS NOT `s_g1_ramp`. That arm added the same penalties as a
DIFFERENTIABLE term on top of the BC loss and reached |acc| 2.8 -- the smoothest
student measured, near the teacher's 0.7 -- while success collapsed to 64.8%.
The penalties were not too strong; the objective had nothing to push back with.
BC has no notion of task success, so "stop moving" is a free win. Under PPO the
task reward is in the same objective and pays for the motion that earns it.

Read that result the other way and it is the reason to try this at all: a
camera-only student CAN represent a near-teacher-smooth policy. It was never
affordable, not never reachable.

THE THREE WAYS THIS RUN CAN DIE, and what is done about each:

  1. A RANDOM CRITIC. PPO starts with an untrained value function, so the first
     advantages are noise with enough magnitude to destroy a 97.7% policy within
     a few hundred steps. `critic_warmup_iters` holds the actor still while the
     critic fits the returns of a policy that is not moving.

  2. ENTROPY INFLATING THE STD. The distilled student's std is 0.02 and PPO's
     default `entropy_coef` 0.005 rewards making it larger. The executed-std
     sweep in `flexiv_two_finger_distill_runner_cfg` is unambiguous about what
     that costs here -- 0.02 -> 97.8% lifted, 0.05 -> 42.5% -- so entropy is set
     to 0 for fine-tuning. There is nothing left to explore for; the policy is
     already on-task.

  3. DRIFT. Nothing in PPO's objective remembers the distilled policy, and the
     reward has known hackable directions (see the wrist-roll finding in the
     pose line). The adaptive-KL learning rate bounds each step but not the
     cumulative walk. Not addressed here beyond a low learning rate; if the run
     survives the first two and still degrades, a KL term against a frozen copy
     of the distilled actor is the next lever, and it requires copying
     `PPO.update` the way `SmoothedDaggerDistillation` copies its base.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.manipulation.rl.runner import _OnnxExportOnSaveMixin


class FinetunePPO(PPO):
  """PPO with a critic warm-up, for starting from a pretrained actor.

  Args:
    critic_warmup_iters: Number of updates during which the actor is held fixed
      and only the critic learns. 0 disables.

  The freeze is done with ``requires_grad_(False)`` rather than by editing the
  optimizer: Adam skips parameters whose grad is None, so the actor's moment
  estimates stay untouched and resume clean.

  Note what this does NOT freeze: the actor's ``EmpiricalNormalization`` keeps
  its statistics in buffers, which no gradient flag reaches, and PPO updates
  them from every rollout whether or not the weights are learning. That is not
  a harmless detail -- it is what destroyed job 68179 DURING this warm-up. See
  ``ManipulationFinetuneRunner._freeze_obs_normalizer``, which is where the
  buffers are actually pinned.
  """

  def __init__(
    self,
    *args,
    critic_warmup_iters: int = 0,
    std_max: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.critic_warmup_iters = critic_warmup_iters
    self.std_max = std_max
    self._updates = 0

  def update(self) -> dict[str, float]:
    self._updates += 1
    frozen = self._updates <= self.critic_warmup_iters
    if frozen:
      for p in self.actor.parameters():
        p.requires_grad_(False)
    try:
      out = super().update()
    finally:
      if frozen:
        for p in self.actor.parameters():
          p.requires_grad_(True)
    out["actor_frozen"] = float(frozen)
    # The std is a learned parameter again here, unlike in distillation where it
    # gets no gradient. It is the quantity most likely to move first and the one
    # whose cliff is measured, so it is logged every update.
    param = getattr(self._raw_actor.distribution, "std_param", None)
    if param is not None:
      # MEASURED, on the 12-iteration smoke (job 67480): exactly 0.0200 for all
      # six frozen updates, then 0.0201 / 0.0202 / 0.0206 / 0.0210 / 0.0213 /
      # 0.0216 -- monotone over every unfrozen update, and `entropy_coef` was
      # already 0. So the pressure is the policy gradient itself, not entropy,
      # and turning entropy off is not sufficient on its own.
      #
      # What makes this worth a hard clamp rather than a watch: the executed-std
      # cliff on this task is measured and steep (0.02 -> 97.8% lifted,
      # 0.05 -> 42.5%), the drift is toward it, and the resulting failure looks
      # like "fine-tuning made the policy worse" rather than like an exploration
      # setting. Clamping the std costs nothing a fine-tune needs: PPO's
      # gradient still flows through the MEAN, which is the only thing being
      # asked to change here.
      if self.std_max > 0.0:
        param.data.clamp_(max=self.std_max)
      out["exec_std"] = float(param.mean().item())
    return out


class ManipulationFinetuneRunner(_OnnxExportOnSaveMixin, MjlabOnPolicyRunner):
  """PPO runner that can warm-start its actor from a DISTILLATION checkpoint.

  A distillation checkpoint stores ``student_state_dict``/``teacher_state_dict``;
  PPO wants ``actor_state_dict``/``critic_state_dict``. The student and the
  fine-tuned actor are the same architecture on the same observation groups, so
  the student's weights load into the actor unchanged -- but only if the actor's
  model config here matches the one the student was distilled under, including
  ``obs_normalization`` (the normalizer's running statistics are in the
  checkpoint and are what make the CNN's inputs mean what they meant).

  The critic is deliberately NOT initialized from anything. The distillation run
  never had one, and a critic that has not seen this policy's returns is exactly
  what ``critic_warmup_iters`` exists to fix.
  """

  env: RslRlVecEnvWrapper

  def _freeze_obs_normalizer(self) -> None:
    """Stop the actor's ``EmpiricalNormalization`` from learning. THE fix.

    MEASURED, job 68179 -- the second fine-tune, which was healthy for 60
    iterations (reward 50, `lift_hold` 0.95, better than the distilled student)
    and then lost the cube entirely at iteration 65: `pad_touch` 0.21 ->
    0.0001, `grasp` -> 0, `lift_hold` -> 0, and it never came back over the
    remaining 1400 iterations.

    It collapsed while ``critic_warmup_iters`` still held the actor. Comparing
    model_0 against model_100 confirms the freeze worked -- every parameter
    differs by 7.4e-4, exactly one unfrozen Adam update -- and `exec_std` sat at
    0.0200 throughout. The weights did not move; the behaviour did.

    What moved is this normalizer, whose statistics are BUFFERS and so are
    untouched by ``requires_grad_(False)``. The student observation is
    ``joint_pos(11) + joint_vel(11) + actions(11) + goal_height(1)``, and dims
    22-32 -- the raw action fed back as an observation -- ran away:

      dim   at distill   iter 0   iter 500   iter 1499
       29       1.5171   5.4436    11.4350     19.9874
       27       0.4682   1.6355     3.7712      7.5230
       22       1.0164   1.3987     3.5151      5.4905
      (joint_pos and joint_vel, dims 0-21, did not move at all)

    Normalizing DIVIDES by that std, so the policy's own action-feedback channel
    was progressively flattened toward zero -- the input was being removed from
    under a warm-started policy that depends on it. Iteration 64 is where it
    fell off the cliff.

    And it cannot recover: ``count`` is 73.7M inherited from distillation
    against 24.6k new samples per iteration, so a poisoned statistic decays at
    0.03% per iteration.

    Setting ``until`` to the current count makes ``update`` return immediately.
    The distilled statistics are the ones the student's weights were fitted
    against, so they are the correct ones to keep, not a liability to adapt away
    from. The CRITIC's normalizer is deliberately left learning -- it starts
    from nothing and has no weights to be consistent with.
    """
    norm = getattr(self.alg.actor, "obs_normalizer", None)
    if norm is None or not hasattr(norm, "count"):
      # obs_normalization off, so the normalizer is a torch.nn.Identity.
      return
    n = int(norm.count.item())
    if n == 0:
      print("[finetune] WARNING: obs normalizer has count 0, refusing to freeze")
      return
    norm.until = n
    print(f"[finetune] actor obs normalizer frozen at count {n}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if "student_state_dict" not in loaded or "actor_state_dict" in loaded:
      infos = super().load(path, load_cfg, strict, map_location)
      self._freeze_obs_normalizer()
      return infos

    print(f"[finetune] warm-starting the actor from the student in {path}")
    # Critic, optimizer and iteration all start fresh -- none of them exist in a
    # distillation checkpoint, and asking for them raises a KeyError that reads
    # like a corrupt file rather than a missing stage. `iteration` False also
    # keeps the fine-tune's own counter at 0, so `max_iterations` and the
    # warm-up window are counted in fine-tuning steps and not continued from
    # the distillation run's 3000.
    self.alg.load(
      {"actor_state_dict": loaded["student_state_dict"]},
      {"actor": True, "critic": False, "optimizer": False, "iteration": False},
      strict,
    )
    self._freeze_obs_normalizer()
    infos = loaded.get("infos")
    # The curricula are driven by this counter, so a fine-tune that dropped it
    # would silently restart every schedule the distilled policy grew up under.
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
