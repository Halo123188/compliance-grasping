"""PPO + recurrent actor / asymmetric critic runner config (spec §3.1).

The actor is a GRU because it has to solve an estimation problem before it can
solve a control problem: external force is observable only as a signature in the
joint-torque history, and a single frame of ``tau_meas`` cannot distinguish "a
hand is pulling me" from "I am accelerating". The critic is a GRU too, but its
input includes the privileged perturbation state, so its recurrence is only
smoothing rather than inference.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

_AUX_ACTOR = "mjlab.tasks.compliance_tracking.rl_aux:AuxRNNModel"
_AUX_PPO = "mjlab.tasks.compliance_tracking.rl_aux:AuxPPO"


@dataclass
class RslRlPpoAuxAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO cfg + the force-head loss weight, resolved to ``AuxPPO``."""

  aux_coef: float = 1.0
  """Weight on the per-step force-estimation MSE added to the PPO objective."""
  class_name: str = _AUX_PPO


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


def flexiv_tracking_ppo_aux_runner_cfg(
  experiment_name: str = "compliance_tracking_flexiv",
  max_iterations: int = 15_000,
  aux_coef: float = 1.0,
) -> RslRlOnPolicyRunnerCfg:
  """Runner cfg for the direct-torque policy with the auxiliary force loss.

  Same PPO/GRU as the base tracking runner, but the actor resolves to
  ``AuxRNNModel`` (adds the force head) and the algorithm to ``AuxPPO`` (adds the
  regression term). Only the *actor* gets the head — the critic already sees
  ``F_ext`` and needs no estimator. See ``rl_aux``.
  """
  cfg = flexiv_tracking_ppo_runner_cfg(experiment_name, max_iterations)
  cfg.actor = replace(cfg.actor, class_name=_AUX_ACTOR)
  base_alg = cfg.algorithm
  cfg.algorithm = RslRlPpoAuxAlgorithmCfg(
    **{
      f.name: getattr(base_alg, f.name)
      for f in fields(base_alg)
      if f.name != "class_name"
    },
    aux_coef=aux_coef,
  )
  return cfg
