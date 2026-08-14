"""PPO and distillation runner configs for the Flexiv two-finger grasp tasks."""

from mjlab.rl import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Shared CNN image encoder for the vision tasks (RGB / depth / RGB-D). A small
# conv stack with spatial-softmax pooling; the first conv adapts to the input
# channel count (3 for RGB, 1 for depth, 4 for RGB-D), so the same cfg works for
# every modality.
_VISION_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def _base(
  experiment_name: str,
  init_std: float = 1.5,
  entropy_coef: float = 0.02,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": init_std,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


def flexiv_two_finger_grasp_ppo_runner_cfg(
  init_std: float = 1.5,
  entropy_coef: float = 0.02,
) -> RslRlOnPolicyRunnerCfg:
  """PPO config; the two exploration knobs are exposed because the fingers need
  different ones from the arm.

  Measured on run 46846's checkpoint, the LEARNED per-dimension action std after
  3000 iterations:

    joint1-6   0.16-0.61      joint7  0.06
    left_1/2, right_1/2       1.95-1.97

  The arm dims fell from the 1.5 they start at because they got a usable
  gradient; the finger dims ROSE, which is what entropy_coef does to a dimension
  whose advantage does not correlate with its action. At std 1.95 and scale 0.6
  that is a +-1.17 rad random target on each finger EVERY control step, and the
  position servo (kp=5, 2 N.m limit) cannot follow it -- replaying that flailing
  moves the cube only 17.5 mm, i.e. the fingers never actually close on it even
  though they command the pinch angle constantly. Nothing a single step's action
  does changes the outcome, so the advantage stays ~0 and entropy keeps inflating
  the std. Lowering both knobs breaks that loop.
  """
  return _base("flexiv_two_finger_wide_grasp", init_std, entropy_coef)


def flexiv_two_finger_vision_ppo_runner_cfg(
  experiment_name: str = "flexiv_two_finger_wide_grasp_vision",
) -> RslRlOnPolicyRunnerCfg:
  """Vision runner: CNN image encoder + proprio MLP.

  ``obs_groups`` routes the proprio group ("actor"/"critic") and the image group
  ("camera") into each network; the CNN encodes the image, its features are
  concatenated with proprio, then the MLP heads run.
  """
  cfg = _base(experiment_name)
  cfg.actor.cnn_cfg = _VISION_CNN_CFG
  cfg.actor.class_name = _VISION_MODEL_CLS
  cfg.critic.cnn_cfg = _VISION_CNN_CFG
  cfg.critic.class_name = _VISION_MODEL_CLS
  cfg.obs_groups = {
    "actor": ("actor", "camera"),
    "critic": ("critic", "camera"),
  }
  return cfg


# Deeper than _VISION_CNN_CFG: the student has to recover the cube's YAW from a
# 50 mm box in the frame, and two stride-2 convs leave a receptive field too
# small to see a whole cube face. Three layers put the feature map at 20x15 on
# this task's 160x120 render (16x9 on the old 128x72 one) with 64 channels, so
# spatial softmax returns 128 keypoint coordinates either way -- the channel
# count sets the output width, not the resolution.
_STUDENT_CNN_CFG = {
  "output_channels": [32, 64, 64],
  "kernel_size": [5, 3, 3],
  "stride": [2, 2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}

# Same trunk with the FIRST stride halved, so the map ends at 40x30 here (32x18
# on the old render). Motivated by where the student stalls rather than by
# theory: at 600 iterations it lifts the cube to a 78.7 mm median but clears the
# 100 mm bar only 3.1% of the time, i.e. it finds the cube and misses on
# precision. The cube was 9-15 px wide in a 128x72 frame
# (`scripts/diag_student_view.py`), and three stride-2 layers collapse that to
# roughly ONE cell of the final map -- so a keypoint can only place it to within
# ~8 input pixels. Quadrupling the output cells costs one layer's worth of
# compute and no parameters.
#
# The 160x120 render already buys back part of this (300 cells against 144), so
# on this task it is an arm rather than the default. See __init__.py.
_STUDENT_CNN_FINE_CFG = {**_STUDENT_CNN_CFG, "stride": [1, 2, 2]}


def flexiv_two_finger_distill_runner_cfg(
  experiment_name: str = "flexiv_two_finger_wide_grasp_distill",
  student_std: float = 0.02,
  learning_rate: float = 3.0e-4,
  max_iterations: int = 2_000,
  student_sees_state: bool = False,
  beta_decay_iters: int = 0,
  fine_cnn: bool = False,
  relabel_achievable: bool = False,
  saturation_weight: float = 0.0,
  action_rate_weight: float = 0.0,
  penalty_ramp_start: int = 0,
  penalty_ramp_end: int = 0,
) -> RslRlDistillationRunnerCfg:
  """DAgger: fit a depth/RGB student to the frozen state teacher's actions.

  The ``teacher`` model config must match the PPO checkpoint being loaded
  bit for bit -- same hidden dims, same activation, same ``obs_normalization``
  (the normalizer's running mean/var live in the checkpoint), and the same
  distribution type so ``distribution.std_param`` has somewhere to land. These
  mirror ``flexiv_two_finger_grasp_ppo_runner_cfg``; changing one without the
  other loads a teacher that silently is not the teacher.

  ``student_std`` is not annealed and gets no gradient -- the loss regresses the
  student's MEAN onto the teacher's action, so this is a fixed exploration
  noise whose only job is to widen the state distribution being labelled. DAgger
  EXECUTES that noise, so it has to stay under what the task can absorb. Swept
  on a student that scores 98.6% deterministically
  (``scripts/diag_success_metric.py ... std=0.0,0.01,0.02,0.05,0.1``):

    exec std   0.0     0.01    0.02    0.05    0.10
    lifted    98.6%   98.6%   97.8%   42.5%   21.4%

  The cliff sits between 0.02 and 0.05, so 0.1 -- the first value tried here --
  was destroying five sixths of the rollouts before the teacher ever labelled
  them, and the resulting off-task states were then the entire training set.
  This is the same trap as the PPO cfg's finger-noise story: noise the actuator
  and the task cannot absorb looks like a policy that cannot learn.

  ``student_sees_state`` is the control condition, not a deployable policy: the
  student is handed the teacher's own observation and no image. It should fit
  almost perfectly and fast, because it is the same input through the same
  architecture. If THAT does not converge, the fault is in the distillation
  machinery -- optimizer, loss scale, data plumbing -- and not in perception,
  which is otherwise impossible to tell apart from a flat loss curve.
  """
  return RslRlDistillationRunnerCfg(
    student=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=None
      if student_sees_state
      else (_STUDENT_CNN_FINE_CFG if fine_cnn else _STUDENT_CNN_CFG),
      class_name="MLPModel" if student_sees_state else _VISION_MODEL_CLS,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": student_std,
        "std_type": "scalar",
      },
    ),
    teacher=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    ),
    algorithm=RslRlDistillationAlgorithmCfg(
      # THE GRADIENT-STEP BUG, and it cost two 1200-iteration runs. RSL-RL's
      # `Distillation.update()` walks a generator that yields ONE BATCH PER
      # ROLLOUT TIMESTEP and only calls `optimizer.step()` every
      # `gradient_length` batches. At epochs=2 / gradient_length=8 over
      # num_steps_per_env=24 that is 48/8 = SIX optimizer steps per iteration --
      # 7,200 across 1200 iterations, nowhere near enough to train a CNN from
      # scratch. Runs 50273/50274 showed exactly that: behaviour loss
      # oscillating 0.5-1.5 with no trend (a constant-action student scores
      # 0.84) and `episode_success` at 0.0000 throughout.
      #
      # `gradient_length` exists to accumulate BPTT chunks for RECURRENT
      # students. This student is feedforward, so there is nothing to backprop
      # through time and no reason to batch timesteps together -- 1 gives a step
      # per timestep, and each batch is still num_envs samples wide.
      # 5 epochs x 24 steps = 120 optimizer steps per iteration, 20x more
      # learning from the same collected data.
      #
      # The oracle control (`student_sees_state=True`) hid this: an MLP on 43
      # privileged inputs fits the teacher to MSE 0.01 within 150 iterations
      # even at six steps each, so the plumbing looked healthy.
      num_learning_epochs=5,
      gradient_length=1,
      learning_rate=learning_rate,
      max_grad_norm=1.0,
      loss_type="mse",
      # DAgger mixing, off by default. The student drives from iteration 0
      # otherwise, and on this task it knocks the cube off the table in the
      # first few steps and never recovers -- so every later label is the
      # teacher's opinion about a cube on the floor.
      # The smoothed class is a strict superset of DaggerDistillation -- with
      # all three knobs at their defaults it IS DaggerDistillation -- but it is
      # only selected when one of them is on, so an unrelated distillation run
      # keeps the exact code path it had.
      class_name=(
        "mjlab.tasks.manipulation.rl.distillation:SmoothedDaggerDistillation"
        if (relabel_achievable or saturation_weight or action_rate_weight)
        else "mjlab.rl.distillation:DaggerDistillation"
        if beta_decay_iters > 0
        else "Distillation"
      ),
      beta_decay_iters=beta_decay_iters,
      relabel_achievable=relabel_achievable,
      saturation_weight=saturation_weight,
      action_rate_weight=action_rate_weight,
      penalty_ramp_start=penalty_ramp_start,
      penalty_ramp_end=penalty_ramp_end,
    ),
    obs_groups={
      "student": ("actor",) if student_sees_state else ("student", "camera"),
      "teacher": ("actor",),
    },
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )


def flexiv_two_finger_finetune_runner_cfg(
  experiment_name: str = "flexiv_two_finger_wide_grasp_finetune",
  learning_rate: float = 1.0e-4,
  entropy_coef: float = 0.0,
  critic_warmup_iters: int = 100,
  std_max: float = 0.03,
  max_iterations: int = 1_500,
  student_history: int = 0,
  fine_cnn: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """PPO on the DISTILLED student: camera actor, privileged critic.

  The actor config must match `flexiv_two_finger_distill_runner_cfg`'s student
  BIT FOR BIT -- same hidden dims, same CNN, same ``obs_normalization``, same
  distribution type -- or the distilled weights do not load into it. The one
  field that legitimately differs is ``init_std``, and only because it is
  immediately overwritten by the checkpoint's own value.

  ASYMMETRIC ACTOR-CRITIC, which is the reason this is affordable at all. The
  actor sees what the robot sees (``student`` + ``camera``); the critic sees the
  43-dim privileged state the teacher was trained on. Estimating a value
  function from a noisy 160x120 depth image is a harder problem than the control
  one, and there is no reason to solve it -- the critic is a training-time
  object and is thrown away at deployment.

  ``entropy_coef`` DEFAULTS TO ZERO, unlike every other PPO cfg here. The
  distilled student arrives with std 0.02 and entropy rewards enlarging it,
  while the executed-std sweep in `flexiv_two_finger_distill_runner_cfg` puts
  the cliff between 0.02 and 0.05 (97.8% -> 42.5% lifted). A fine-tune has
  nothing to discover: the policy is already on-task, and every unit of extra
  exploration is spent on rollouts that fail for reasons unrelated to what is
  being learned.

  ``learning_rate`` is 10x below the from-scratch value for the same reason.
  """
  cfg = _base(experiment_name, init_std=0.02, entropy_coef=entropy_coef)
  cfg.actor = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=_STUDENT_CNN_FINE_CFG if fine_cnn else _STUDENT_CNN_CFG,
    class_name=_VISION_MODEL_CLS,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.02,
      "std_type": "scalar",
    },
  )
  cfg.critic = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
  )
  cfg.obs_groups = {
    "actor": ("student", "camera"),
    "critic": ("critic",),
  }
  cfg.algorithm.class_name = "mjlab.tasks.manipulation.rl.finetune:FinetunePPO"
  cfg.algorithm.learning_rate = learning_rate
  cfg.algorithm.entropy_coef = entropy_coef
  cfg.algorithm.critic_warmup_iters = critic_warmup_iters
  # Setting entropy to 0 is NOT sufficient -- the policy gradient drifts the std
  # up on its own (job 67480). 0.03 leaves the fine-tune a little room above the
  # distilled 0.02 while staying clear of the measured 0.05 cliff.
  cfg.algorithm.std_max = std_max
  # Fixed, not adaptive. The adaptive schedule multiplies the rate by 1.5 as
  # soon as the measured KL is small, and the first thing a frozen actor
  # produces is a KL of exactly zero -- so the warm-up would hand the actor back
  # a learning rate several times the one chosen for it.
  cfg.algorithm.schedule = "fixed"
  cfg.experiment_name = experiment_name
  cfg.max_iterations = max_iterations
  cfg.num_steps_per_env = 24
  # 25, not the 100 used everywhere else. A fine-tune starts good and can only
  # be measured against where it started, so the interesting checkpoints are
  # early and close together. Job 68179 was at its best over iterations 35-63
  # and there is no checkpoint of it: model_0 is the warm start and model_100 is
  # already 35 iterations past the collapse.
  cfg.save_interval = 25
  del student_history  # the env carries it; kept for signature symmetry
  return cfg
