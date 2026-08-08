from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import mujoco
import torch
from conftest import get_test_device

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorData
from mjlab.tasks.manipulation.mdp.commands import MultiCubeLiftingCommand
from mjlab.tasks.manipulation.mdp.observations import (
  camera_depth,
  camera_segmentation,
  camera_target_cube_mask,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _make_env(segmentation: torch.Tensor, target_geom_ids: torch.Tensor):
  sensor = SimpleNamespace(data=CameraSensorData(segmentation=segmentation))

  command = object.__new__(MultiCubeLiftingCommand)
  command._padded_geom_ids = target_geom_ids
  command.target_selection = torch.arange(
    target_geom_ids.shape[0], device=target_geom_ids.device
  )

  command_manager = SimpleNamespace(get_term=lambda _: command)
  return SimpleNamespace(
    scene={"seg_cam": sensor},
    command_manager=command_manager,
  )


def test_camera_segmentation_returns_bchw():
  device = get_test_device()
  geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  seg = torch.tensor(
    [
      [[[1, geom], [2, geom], [-1, -1]], [[3, geom], [4, geom], [-1, -1]]],
      [[[5, geom], [6, geom], [-1, -1]], [[7, geom], [8, geom], [-1, -1]]],
    ],
    dtype=torch.int32,
    device=device,
  )
  env = _make_env(seg, torch.tensor([[1], [7]], dtype=torch.int32, device=device))
  env = cast("ManagerBasedRlEnv", env)

  obs = camera_segmentation(env, "seg_cam")

  assert obs.shape == (2, 2, 2, 3)
  assert obs.dtype == torch.int32
  assert torch.equal(obs[:, 0], seg[..., 0])
  assert torch.equal(obs[:, 1], seg[..., 1])


def test_camera_target_cube_mask_filters_to_geom_hits():
  device = get_test_device()
  geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  flex = int(mujoco.mjtObj.mjOBJ_FLEX)
  seg = torch.tensor(
    [
      [[[3, geom], [3, flex], [0, geom]], [[-1, -1], [4, geom], [3, geom]]],
      [[[5, geom], [7, flex], [5, geom]], [[7, geom], [-1, -1], [0, geom]]],
    ],
    dtype=torch.int32,
    device=device,
  )
  env = _make_env(seg, torch.tensor([[3], [7]], dtype=torch.int32, device=device))
  env = cast("ManagerBasedRlEnv", env)

  mask = camera_target_cube_mask(env, "seg_cam", "lift_height")

  expected = torch.tensor(
    [
      [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
      [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
    ],
    dtype=torch.float32,
    device=device,
  )
  assert mask.shape == (2, 1, 2, 3)
  assert torch.equal(mask, expected)


def _depth_env(depth: torch.Tensor, segmentation: torch.Tensor | None = None):
  """Minimal env exposing what `camera_depth` reads: a sensor and a clock."""
  sensor = SimpleNamespace(
    data=CameraSensorData(depth=depth, segmentation=segmentation)
  )
  num_envs = depth.shape[0]
  return SimpleNamespace(
    scene={"d435": sensor},
    num_envs=num_envs,
    device=depth.device,
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=depth.device),
  )


def test_occlusion_shadow_falls_left_of_the_step_and_is_as_wide_as_disparity():
  device = get_test_device()
  # One row: background at 1.0 m out to column 7, foreground at 0.5 m after it.
  # f*B*(1/0.5 - 1/1.0) = 100 * 0.05 * 1.0 = 5 px of hidden background, so
  # columns 3..7 go invalid and columns 0..2 and 8.. survive.
  depth = torch.ones(1, 1, 16, device=device)
  depth[..., 8:] = 0.5
  env = cast("ManagerBasedRlEnv", _depth_env(depth.unsqueeze(-1)))

  out = camera_depth(
    env,
    "d435",
    cutoff_distance=3.0,
    shadow_focal_px=100.0,
    shadow_baseline=(0.05, 0.05),
    shadow_fill=(1.0, 1.0),
    shadow_step=0.01,
  )

  invalid = (out[0, 0, 0] == 0.0).tolist()
  assert invalid == [False] * 3 + [True] * 5 + [False] * 8


def test_occlusion_shadow_ignores_a_foreground_to_background_step():
  """The near surface is on the LEFT here, so nothing is hidden."""
  device = get_test_device()
  depth = torch.full((1, 1, 16), 0.5, device=device)
  depth[..., 8:] = 1.0
  env = cast("ManagerBasedRlEnv", _depth_env(depth.unsqueeze(-1)))

  out = camera_depth(
    env,
    "d435",
    cutoff_distance=3.0,
    shadow_focal_px=100.0,
    shadow_baseline=(0.05, 0.05),
    shadow_fill=(1.0, 1.0),
  )

  assert bool((out > 0.0).all())


def test_surface_blinding_hits_only_its_geoms_at_the_asked_for_area_fraction():
  device = get_test_device()
  geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  b, h, w = 8, 60, 80
  depth = torch.full((b, h, w, 1), 0.6, device=device)
  seg = torch.full((b, h, w, 2), -1, dtype=torch.int32, device=device)
  # Geom 5 owns the left half of the frame; geom 9 is present but not targeted.
  seg[:, :, : w // 2, 0] = 5
  seg[:, :, : w // 2, 1] = geom
  seg[:, :, w // 2 :, 0] = 9
  seg[:, :, w // 2 :, 1] = geom

  env = _depth_env(depth, seg)
  entity = SimpleNamespace(
    indexing=SimpleNamespace(
      geom_ids=torch.tensor([5, 9], dtype=torch.int32, device=device)
    )
  )
  env.scene["robot"] = entity
  cfg = SceneEntityCfg("robot", geom_names=("g",), geom_ids=[0])

  out = camera_depth(
    cast("ManagerBasedRlEnv", env),
    "d435",
    cutoff_distance=3.0,
    blind_gripper_cfg=cfg,
    blind_gripper_prob=(0.20, 0.20),
  )

  blinded = out == 0.0
  # Nothing outside geom 5.
  assert not bool(blinded[:, :, :, w // 2 :].any())
  # ...and the blinded share of geom 5 is the fraction that was asked for. The
  # field is smooth, so a single frame is lumpy; the mean over 8 envs is not.
  frac = float(blinded[:, :, :, : w // 2].float().mean())
  assert 0.14 < frac < 0.26, frac


def test_surface_blinding_is_anchored_to_the_bounding_box_not_the_image():
  """The same patch of the object stays blind when the object moves."""
  device = get_test_device()
  geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  h, w = 40, 40

  def frame(shift: int):
    depth = torch.full((1, h, w, 1), 0.6, device=device)
    seg = torch.full((1, h, w, 2), -1, dtype=torch.int32, device=device)
    seg[:, 10:30, 5 + shift : 25 + shift, 0] = 5
    seg[:, 10:30, 5 + shift : 25 + shift, 1] = geom
    return depth, seg

  env = _depth_env(*frame(0))
  env.scene["robot"] = SimpleNamespace(
    indexing=SimpleNamespace(
      geom_ids=torch.tensor([5], dtype=torch.int32, device=device)
    )
  )
  cfg = SceneEntityCfg("robot", geom_names=("g",), geom_ids=[0])
  cenv = cast("ManagerBasedRlEnv", env)

  def blind() -> torch.Tensor:
    return camera_depth(
      cenv,
      "d435",
      cutoff_distance=3.0,
      blind_gripper_cfg=cfg,
      blind_gripper_prob=(0.3, 0.3),
    )

  a = blind()
  patch_a = a[0, 0, 10:30, 5:25] == 0.0

  # Move the object 10 px right WITHOUT restarting the episode, so the held
  # field is the same draw.
  depth, seg = frame(10)
  env.scene["d435"].data.depth = depth
  env.scene["d435"].data.segmentation = seg
  b_out = blind()
  patch_b = b_out[0, 0, 10:30, 15:35] == 0.0

  assert torch.equal(patch_a, patch_b)


def test_depth_dr_draws_are_held_for_the_episode_and_redrawn_on_reset():
  device = get_test_device()
  depth = torch.rand(4, 1, 12, 16, device=device).clamp_min(0.2).unsqueeze(-1)
  env = _depth_env(depth.squeeze(-1).permute(0, 2, 3, 1))
  cenv = cast("ManagerBasedRlEnv", env)

  def shot() -> torch.Tensor:
    return camera_depth(
      cenv,
      "d435",
      cutoff_distance=3.0,
      shadow_focal_px=100.0,
      shadow_baseline=(0.03, 0.09),
      shadow_fill=(1.0, 1.0),
    )

  a = shot()
  base = env._depth_dr_d435["baseline"].clone()
  shot()
  assert torch.equal(base, env._depth_dr_d435["baseline"])
  assert torch.equal(a, shot())

  env.episode_length_buf += 5
  shot()
  assert torch.equal(base, env._depth_dr_d435["baseline"])

  env.episode_length_buf.zero_()
  shot()
  assert not torch.equal(base, env._depth_dr_d435["baseline"])


def test_a_draw_first_asked_for_mid_episode_is_not_left_at_zero():
  """The silent no-op: `fresh` is empty, so a new buffer never gets filled.

  This is how the effect switches itself off without failing anything -- zero
  is a legal probability, and a term drawing zero looks exactly like a term
  that is working.
  """
  device = get_test_device()
  depth = torch.full((2, 8, 8, 1), 0.6, device=device)
  env = _depth_env(depth)
  cenv = cast("ManagerBasedRlEnv", env)

  # First call asks for the scale error only, which creates the store and marks
  # every env as seen.
  camera_depth(cenv, "d435", cutoff_distance=3.0, scale_err=0.01)
  assert not bool((env.episode_length_buf < env._depth_dr_d435["seen"]).any())

  # A second call now asks for the shadow. No env has reset in between.
  camera_depth(
    cenv,
    "d435",
    cutoff_distance=3.0,
    shadow_focal_px=100.0,
    shadow_baseline=(0.04, 0.06),
    shadow_fill=(0.7, 0.95),
  )
  assert bool((env._depth_dr_d435["baseline"] >= 0.04).all())
  assert bool((env._depth_dr_d435["fill"] >= 0.7).all())
