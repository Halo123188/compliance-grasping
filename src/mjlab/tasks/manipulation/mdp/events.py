"""Event terms specific to the manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# Private, and imported deliberately: writing `geom_size` without refreshing
# `geom_rbound` / `geom_aabb` leaves the broadphase describing the old shape, and
# the 60 lines that do it correctly per primitive type already exist.
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields(
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "body_inertia",
  recompute=RecomputeLevel.set_const,
)
def object_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Scale an object uniformly, carrying its inertia but NOT its mass.

  ONE DRAW FOR ALL THREE AXES, because a randomized cube has to stay a cube.
  ``dr.geom_size``'s ``shared_random`` shares across GEOMS and still draws each
  axis separately, which turns the cube into a random cuboid -- a different and
  much harder task than the one the pad geometry was calibrated for.

  INERTIA MOVES WITH THE SIZE AND MASS DOES NOT. For a solid body at fixed mass
  ``I ~ m*a^2``, so the tensor scales by ``s^2``; leaving it alone would give a
  55 mm cube the rotational inertia of a 50 mm one, and the grasp is decided by
  exactly that coupling when the jaw tips it. Mass is left to the separate mass
  draw ON PURPOSE: a policy that learns "bigger means heavier" from sim is wrong
  on a bench that can hand it a foam cube and a steel one of the same size.

  MUST RUN AFTER any event that writes ``body_inertia`` from defaults --
  ``dr.pseudo_inertia`` does, absolutely, every reset -- because this multiplies
  whatever is already there. Registered after it in ``add_physics_dr``, and both
  are ``reset`` mode, so the dict's insertion order is the execution order.

  Args:
    scale_range: multiplier on every half-size, drawn once per environment.
    asset_cfg: the object. Its geoms are scaled and its bodies' inertia with
      them, so both selections must name the same physical object.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  lo, hi = scale_range
  s = torch.rand(len(env_ids), 1, 1, device=env.device) * (hi - lo) + lo

  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  # From the DEFAULT size, not the current one, so repeated resets do not
  # compound the scale. `mj_model` is the CPU model and its fields are float64;
  # the per-world model is float32, and an index-put across the two raises.
  live = env.sim.model.geom_size[env_grid, geom_grid]
  default_size = torch.as_tensor(
    env.sim.mj_model.geom_size[geom_ids.cpu().numpy()], device=env.device
  ).to(live.dtype)
  env.sim.model.geom_size[env_grid, geom_grid] = default_size.unsqueeze(0) * s.to(
    live.dtype
  )
  _recompute_geom_bounds(env, env_ids, asset_cfg)

  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  inertia = env.sim.model.body_inertia[env_grid, body_grid]
  env.sim.model.body_inertia[env_grid, body_grid] = inertia * (s * s).to(inertia.dtype)
