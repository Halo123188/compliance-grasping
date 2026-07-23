"""Reach goal command for the Stage-1 compliance task.

Samples an absolute Cartesian goal ``x_g`` in a workspace box, rejecting points
outside the arm's reachable radial shell, and tracks success (EE within a
threshold of the goal, held for a dwell time).  The goal is the equilibrium
anchor the Cartesian impedance controller regulates toward.

This term owns *only* the reach goal.  The human disturbance lives in a separate
event term; success here is judged on EE-to-goal distance and is independent of
whether the human is currently pushing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class ReachCommand(CommandTerm):
  cfg: ReachCommandCfg

  def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.robot_name]
    site_local, _ = self.robot.find_sites(cfg.ee_site_name)
    self._site_id = int(self.robot.indexing.site_ids[site_local[0]].item())

    self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._dwell = torch.zeros(self.num_envs, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """World-frame goal position ``x_g`` (N, 3)."""
    return self.goal_pos

  def ee_pos_w(self) -> torch.Tensor:
    return self._env.sim.data.site_xpos[:, self._site_id]

  def _update_metrics(self) -> None:
    error = torch.norm(self.goal_pos - self.ee_pos_w(), dim=-1)
    within = error < self.cfg.success_threshold
    # Dwell timer: accumulate time inside the threshold, reset on exit.
    self._dwell = torch.where(
      within, self._dwell + self._env.step_dt, torch.zeros_like(self._dwell)
    )
    at_goal = self._dwell >= self.cfg.success_dwell_time
    self.episode_success = torch.maximum(self.episode_success, at_goal.float())

    self.metrics["position_error"] = error
    self.metrics["at_goal"] = at_goal.float()
    self.metrics["episode_success"] = self.episode_success

  def success(self) -> torch.Tensor:
    """Per-env success flag (dwell satisfied). Used for early termination."""
    return self._dwell >= self.cfg.success_dwell_time

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self._dwell[env_ids] = 0.0
    self.episode_success[env_ids] = 0.0

    origins = self._env.scene.env_origins[env_ids]
    lo = torch.tensor(self.cfg.box_min, device=self.device)
    hi = torch.tensor(self.cfg.box_max, device=self.device)

    # Rejection-sample until the goal falls inside the reachable radial shell
    # (a cheap proxy for an IK reachability check, plan §2).
    goal = sample_uniform(lo, hi, (n, 3), device=self.device)
    r_min, r_max = self.cfg.reach_radius
    for _ in range(16):
      radius = torch.norm(
        goal - torch.tensor(self.cfg.base_pos, device=self.device), dim=-1
      )
      bad = (radius < r_min) | (radius > r_max)
      if not bad.any():
        break
      goal[bad] = sample_uniform(lo, hi, (int(bad.sum()), 3), device=self.device)

    self.goal_pos[env_ids] = goal + origins

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      goal = self.goal_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=goal,
        radius=self.cfg.success_threshold,
        color=self.cfg.viz.goal_color,
        label=f"reach_goal_{batch}",
      )


@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
  robot_name: str = "robot"
  ee_site_name: str = "grasp_site"

  # Workspace box (metres, robot-base frame == env origin), 40x40x30 cm centred
  # near the home EE (~0.5, 0.0, 0.5); calibrated against the Rizon4S reach.
  box_min: tuple[float, float, float] = (0.30, -0.20, 0.35)
  box_max: tuple[float, float, float] = (0.70, 0.20, 0.65)

  base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  reach_radius: tuple[float, float] = (0.35, 0.80)
  """Reachable radial shell from the base; goals outside are resampled."""

  success_threshold: float = 0.02  # m (plan §5.2)
  success_dwell_time: float = 0.5  # s (plan §5.2)

  @dataclass
  class VizCfg:
    goal_color: tuple[float, float, float, float] = (0.1, 0.9, 0.2, 0.5)

  viz: VizCfg = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.viz is None:
      self.viz = ReachCommandCfg.VizCfg()

  def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
    return ReachCommand(self, env)
