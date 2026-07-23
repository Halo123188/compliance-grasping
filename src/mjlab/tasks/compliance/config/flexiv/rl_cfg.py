"""PPO + GRU runner config for the Stage-1 compliance task (plan §5.1, §8).

GRU (hidden 128) actor/critic so the policy can infer the hidden human state
(``K_h`` / push direction / released) from the τ history.  Hyperparameters are
the plan §8/E2 starting values.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _gru(distribution: bool) -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(256, 128),
    activation="elu",
    obs_normalization=True,
    rnn_type="gru",
    rnn_hidden_dim=128,
    rnn_num_layers=1,
    class_name="RNNModel",
    distribution_cfg=(
      {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
      if distribution
      else None
    ),
  )


def _mlp(distribution: bool) -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(256, 128, 128),
    activation="elu",
    obs_normalization=True,
    class_name="MLPModel",
    distribution_cfg=(
      {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
      if distribution
      else None
    ),
  )


def flexiv_reach_ppo_runner_cfg(mlp: bool = False) -> RslRlOnPolicyRunnerCfg:
  make = _mlp if mlp else _gru
  return RslRlOnPolicyRunnerCfg(
    actor=make(distribution=True),
    critic=make(distribution=False),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="compliance_reach_flexiv",
    save_interval=100,
    num_steps_per_env=32,  # BPTT truncation length (plan §8)
    max_iterations=15_000,
  )
