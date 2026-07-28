"""Reward terms for the Stage-1 compliance task (plan §5.2).

Rewards may use privileged ground truth (``F_ext`` straight from sim, the
decoded stiffness); observations may not.  Weights (set in the env cfg) and their
target per-episode magnitudes:

  progress    +10     (≈ +4)     ‖x_ee−x_g‖ decrease
  success     +1      (+1)       dwell satisfied (also ends the episode)
  time        −0.005  (≈ −2.5)   constant
  force       −0.001  (≈ −3)     −max(0, ‖F_ext‖ − 5N)
  K           −0.002  (≈ −0.5)   −mean(log K / log K_max)
  action_rate −0.01              −‖a_t − a_{t−1}‖²  (framework action_rate_l2)

The K penalty is not optional: without it RL converges to "stiff everywhere"
(zero-cost in sim).  Rewards are configured with ``scale_rewards_by_dt=False`` so
these per-step weights match the magnitudes above over a 500-step episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.compliance.mdp.human import HumanDisturbance

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_LOG_K_MAX = torch.log(torch.tensor(2000.0))


def _human(env: ManagerBasedRlEnv, event_name: str) -> HumanDisturbance:
  return env.event_manager.get_term_cfg(event_name).func


class progress:
  """Δ distance-to-goal per step (positive when approaching)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._command_name = cfg.params.get("command_name", "reach")
    self._event_name = cfg.params.get("event_name", "human_disturbance")
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._need_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "reach",
    event_name: str = "human_disturbance",
  ) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    dist = torch.norm(cmd.command - cmd.ee_pos_w(), dim=-1)
    prog = self._prev - dist
    # Freshly reset envs: no progress yet, just seed prev with the new goal.
    prog = torch.where(self._need_init, torch.zeros_like(prog), prog)
    # No progress credit/penalty while the human is pulling the wrist: otherwise
    # every cm the hand drags the arm off-goal is a penalty, so the arm learns to
    # resist the pull, never reaches the hand target, and never gets released.
    # We still advance ``_prev`` so the return trip after release is scored from
    # wherever the arm ended up.  This is the key to the yield -> release ->
    # return-to-goal loop (the human is transient; plan §4.3).
    human = self._env.event_manager.get_term_cfg(self._event_name).func
    prog = torch.where(human.is_pushing(), torch.zeros_like(prog), prog)
    self._need_init[:] = False
    self._prev = dist.detach()
    return prog

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._need_init[env_ids] = True


def success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str = "reach",
  event_name: str = "human_disturbance",
) -> torch.Tensor:
  """Per-step dwell bonus, suppressed while the human is pulling the wrist.

  Ungated this is a reward exploit: the bonus is per-step and the success
  *termination* is gated on ``~is_pushing``, while the human only releases once
  the arm yields.  So an arm that braces at the goal is never released, never
  terminates, and farms +1/step for the rest of the episode (e14: +328/episode
  against a total bracing cost of -103).  Gating it here matches ``progress``
  and ``reached_goal``: holding the goal under load earns nothing, so the only
  route back to earning is yield -> be released -> return.
  """
  cmd = env.command_manager.get_term(command_name)
  human = _human(env, event_name)
  return (cmd.success() & ~human.is_pushing()).float()


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def force_penalty(
  env: ManagerBasedRlEnv, event_name: str = "human_disturbance", deadband: float = 5.0
) -> torch.Tensor:
  """``max(0, ‖F_ext‖ − deadband)`` — penalise sustained large human force."""
  f = _human(env, event_name).external_force()
  return torch.clamp(torch.norm(f, dim=-1) - deadband, min=0.0)


def stiffness_penalty(
  env: ManagerBasedRlEnv, action_name: str = "impedance"
) -> torch.Tensor:
  """``mean(log K / log K_max)`` over the 3 Cartesian axes."""
  k = env.action_manager.get_term(action_name).stiffness  # (N, 3)
  log_k_max = _LOG_K_MAX.to(k.device)
  return (torch.log(k) / log_k_max).mean(dim=-1)


def joint_acc_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg=None,
  cap: float = 50000.0,
) -> torch.Tensor:
  """Clamped sum-of-squared joint acceleration, for suppressing the reset whip.

  Raw ``joint_acc_l2`` is dominated ~700000x by the reset-step numerical
  transient (static arm -> large torque -> instantaneous huge acceleration), so
  an unclamped penalty chases that spike and blows up the reward variance.
  Clamping the per-step value at ``cap`` bounds the reset artefact while still
  penalising the elevated (but finite) acceleration of a hard "whip to the
  goal", which is the surgical target: steady reach / yield / return motion is
  low-acceleration and barely touched.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  asset = env.scene[asset_cfg.name]
  acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  return torch.clamp(torch.sum(acc**2, dim=-1), max=cap)


def torque_effort_penalty(
  env: ManagerBasedRlEnv, action_name: str = "torque"
) -> torch.Tensor:
  """``mean((tau / tau_limit)**2)`` over the arm joints.

  The direct-torque analogue of ``stiffness_penalty``: discourage the policy
  from slamming maximum torque (bang-bang) and reward economy of effort.  Reads
  the final commanded torque (post gravity comp / clip) so it penalises the same
  quantity in the pure and residual variants.
  """
  term = env.action_manager.get_term(action_name)
  tau = term.processed_torque  # (N, J)
  limit = term._limit  # (J,)
  return ((tau / limit) ** 2).mean(dim=-1)


def _k_par_perp(env, event_name, action_name):
  human = _human(env, event_name)
  k = env.action_manager.get_term(action_name).stiffness  # (N, 3)
  u = human.push_dir()  # (N, 3), zero when not pushing
  k_par = (u**2 * k).sum(dim=-1)  # (N,)
  k_perp = (k.sum(dim=-1) - k_par) / 2.0  # (N,)
  return human, k_par, k_perp


def push_softness(
  env: ManagerBasedRlEnv,
  event_name: str = "human_disturbance",
  action_name: str = "impedance",
) -> torch.Tensor:
  """Primary directional-compliance term: penalise the ABSOLUTE stiffness along
  the push direction (log K_par / log K_max) while the hand pulls, so the arm
  genuinely yields (gets pulled).  Absolute, not the K_par/K_perp ratio: the
  ratio is zero at "stiff everywhere" (K_par=K_perp) -- the brace we want to
  avoid -- and can be gamed by raising K_perp.  See push_anisotropy for the
  (secondary) shape term.
  """
  human, k_par, _ = _k_par_perp(env, event_name, action_name)
  log_k_max = _LOG_K_MAX.to(k_par.device)
  return (torch.log(k_par.clamp(min=1.0)) / log_k_max) * human.is_pushing().float()


def push_anisotropy(
  env: ManagerBasedRlEnv,
  event_name: str = "human_disturbance",
  action_name: str = "impedance",
) -> torch.Tensor:
  """Secondary shape term: mildly reward K_par < K_perp (soft along push, stiff
  orthogonal) so the orthogonal axes stay firm (hold the goal / reject the A1
  load) as push_softness drives K_par down.  log(K_par/K_perp) while pulled.
  """
  human, k_par, k_perp = _k_par_perp(env, event_name, action_name)
  log_k_max = _LOG_K_MAX.to(k_par.device)
  aniso = (
    torch.log(k_par.clamp(min=1.0)) - torch.log(k_perp.clamp(min=1.0))
  ) / log_k_max
  return aniso * human.is_pushing().float()
