"""PPO + recurrent actor / asymmetric critic runner config (spec §3.1).

The actor is a GRU because it has to solve an estimation problem before it can
solve a control problem: external force is observable only as a signature in the
joint-torque history, and a single frame of ``tau_meas`` cannot distinguish "a
hand is pulling me" from "I am accelerating". The critic is a GRU too, but its
input includes the privileged perturbation state, so its recurrence is only
smoothing rather than inference.
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


def flexiv_tracking_ppo_runner_cfg(
  experiment_name: str = "compliance_tracking_flexiv",
  max_iterations: int = 15_000,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=_gru(distribution=True),
    critic=_gru(distribution=False),
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
      normalize_advantage_per_mini_batch=False,
    ),
    experiment_name=experiment_name,
    save_interval=200,
    num_steps_per_env=32,  # BPTT truncation length
    max_iterations=max_iterations,
  )
