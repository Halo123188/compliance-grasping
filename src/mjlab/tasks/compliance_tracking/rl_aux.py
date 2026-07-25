"""Auxiliary force-estimation loss for the direct-torque compliance policy.

The direct-torque policy has to solve an *estimation* problem before it can
solve a control one: the external push is observable only as a signature in the
joint-torque history, and the tracking reward is far too weak a signal to make
the recurrent encoder learn that inference on its own — hence the earlier
proprio-torque runs came out compliant only intermittently (steady-state
``k_∥`` swinging 390–650 across seeds).

This module supplies the missing gradient without feeding the answer in.  At
train time we *have* the ground-truth ``F_ext`` (privileged), so we hang a small
head off the actor's recurrent latent and regress it onto ``F_ext`` with a
per-step MSE.  The head is discarded at deployment; what survives is a GRU
encoder that has been explicitly shaped to carry a force estimate in its state,
which the control head can then read.  This is the deployable counterpart to the
``pull_direction`` diagnostic — same information, but *learned* from
proprioception rather than handed in.

Both pieces plug into rsl_rl through the config's dotted ``class_name`` fields
(``AuxRNNModel`` for the actor, ``AuxPPO`` for the algorithm); nothing in rsl_rl
is forked.  The label rides the env's observation pipeline as a dedicated
``ext_force`` group that no model set consumes (see ``mdp.ext_force``), so it
reaches ``RolloutStorage`` and is available in every update batch.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import RNNModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

AUX_LABEL_GROUP = "ext_force"
"""Observation group holding the privileged ``F_ext`` regression target."""


class AuxRNNModel(RNNModel):
  """GRU actor with a force-estimation head on the recurrent latent.

  Identical to :class:`RNNModel` for acting; it additionally exposes
  :meth:`predict_force`, which reads the *same* post-RNN latent the control head
  consumes and maps it to a 3-vector force estimate.  The latent is stashed on
  every ``get_latent`` call so the update step can query the head against the
  exact tensor produced by the forward it just ran (no recompute, identical
  graph), and — crucially — the stashed latent is the already-unpadded recurrent
  output, so it lines up row-for-row with an ``unpad_trajectories`` of the label.
  """

  def __init__(self, *args, aux_hidden_dim: int = 64, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.force_head = nn.Sequential(
      nn.Linear(self.latent_dim, aux_hidden_dim),
      nn.ELU(),
      nn.Linear(aux_hidden_dim, 3),
    )
    self._aux_latent: torch.Tensor | None = None

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    latent = super().get_latent(obs, masks, hidden_state)
    self._aux_latent = latent
    return latent

  def predict_force(self) -> torch.Tensor:
    """Force estimate from the most recent ``get_latent`` (matches its shape)."""
    assert self._aux_latent is not None, "call the actor forward before predict_force"
    return self.force_head(self._aux_latent)


class AuxPPO(PPO):
  """PPO plus a per-step force-estimation MSE on the actor's force head.

  The only change from :class:`PPO.update` is a single extra loss term added to
  the PPO objective before the (unchanged) backward: the actor's force
  prediction against the privileged ``F_ext`` label, unpadded with the *same*
  trajectory masks the recurrent forward used so the two align exactly.  The head
  shares the actor's optimizer (its parameters are already in
  ``self.actor.parameters()``), so its gradient flows back through the GRU and
  shapes the encoder — that shaping, not the head itself, is the point.
  """

  def __init__(self, *args, aux_coef: float = 1.0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.aux_coef = aux_coef

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_aux_loss = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      # Recompute log-prob and entropy under the current parameters. This forward
      # also stashes the actor's post-RNN latent (see AuxRNNModel.get_latent),
      # which the force head reads below.
      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,  # type: ignore
            distribution_params,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      # --- auxiliary force-estimation loss ------------------------------------
      # force_pred rides the actor's unpadded recurrent latent; the label is the
      # same F_ext group put through the identical unpad, so both are (T, N_valid,
      # 3) in the same order. Added to the PPO loss so its gradient shares the one
      # backward and reaches the GRU.
      force_pred = cast(AuxRNNModel, self._raw_actor).predict_force()
      force_label = batch.observations[AUX_LABEL_GROUP]  # type: ignore
      if batch.masks is not None:
        force_label = unpad_trajectories(force_label, batch.masks)  # type: ignore
      aux_loss = (force_pred - cast(torch.Tensor, force_label)).pow(2).mean()
      loss = loss + self.aux_coef * aux_loss

      rnd_loss = (
        self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore
        if self.rnd
        else None
      )

      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(
          self.actor, batch, original_batch_size
        )
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_aux_loss += aux_loss.item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_aux_loss /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "aux_force": mean_aux_loss,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss

    self.storage.clear()
    return loss_dict
