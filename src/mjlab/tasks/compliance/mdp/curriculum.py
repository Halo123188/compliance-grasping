"""Displacement curriculum for the human disturbance (plan §6).

Only ``d`` (the push displacement) is curriculum-controlled; ``K_h``, direction,
and ``t_push`` are full-range from step one.  The rule is global (not per-env),
based on the success rate over the last 100 finished episodes:

  level 0: d ~ U(0.02, 0.05)   level 2: d ~ U(0.03, 0.15)
  level 1: d ~ U(0.02, 0.10)   level 3: d ~ U(0.05, 0.20)   ← final

  success_rate > 0.7  ->  level += 1   (monotonic, never regresses)

This is *not* a substitute for the stubborn human: the curriculum only shapes
early training and leaves the objective untouched, whereas a timeout would
permanently poison it (plan §6).  Evaluation always runs at level 3 regardless
of the current training level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.compliance.mdp.human import HumanDisturbance

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_LEVELS: tuple[tuple[float, float], ...] = (
  (0.02, 0.05),
  (0.02, 0.10),
  (0.03, 0.15),
  (0.05, 0.20),
  (0.05, 0.30),  # e11: farther pulls (OOD sweep showed 0.20-0.30 dropped to .67)
)


class displacement_curriculum:
  """Global success-rate driven ``d`` curriculum (see module docstring)."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._event_name = cfg.params.get("event_name", "human_disturbance")
    self._command_name = cfg.params.get("command_name", "reach")
    self._window = int(cfg.params.get("window", 100))
    self._threshold = float(cfg.params.get("promote_threshold", 0.7))
    self._level = 0
    self._recent: list[float] = []
    self._set_level(0)

  def _human(self) -> HumanDisturbance:
    return self._env.event_manager.get_term_cfg(self._event_name).func

  def _set_level(self, level: int) -> None:
    self._level = level
    self._human().d_range = _LEVELS[level]

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **kwargs) -> dict:
    del kwargs
    # Record outcomes of the episodes finishing now (command not yet reset).
    success = env.command_manager.get_term(self._command_name).episode_success
    outcomes = success[env_ids].detach().cpu().tolist()
    self._recent.extend(outcomes)
    if len(self._recent) > self._window:
      self._recent = self._recent[-self._window :]

    rate = float(sum(self._recent) / len(self._recent)) if self._recent else 0.0
    if (
      self._level < len(_LEVELS) - 1
      and len(self._recent) >= self._window
      and rate > self._threshold
    ):
      self._set_level(self._level + 1)
      self._recent.clear()  # require a fresh window at the new level

    return {"level": float(self._level), "success_rate": rate}

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # Global state; nothing per-env to reset.
