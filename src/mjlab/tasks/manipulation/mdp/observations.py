from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
)
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to object in base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  distance_vec_w = obj_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def object_orientation(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Object orientation (quaternion) in the robot base frame.

  Without this the policy cannot see which way the object is turned, only where
  it is (``ee_to_object_distance`` is a position vector). That is fatal for an
  antipodal gripper on a yaw-randomized box: the grasp requires squaring the jaw
  to a face, and the angle to square to is unobservable.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  base_quat_w = robot.data.root_link_quat_w
  return quat_mul(quat_inv(base_quat_w), obj.data.root_link_quat_w)


def object_to_goal_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from object to goal in base frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos
  distance_vec_w = goal_pos_w - obj_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def goal_height(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Commanded goal height above the environment origin. Shape (B, 1).

  This is the one part of the lift command that is NOT privileged: an operator
  asking for a lift supplies the target height, so a policy without cube pose
  still gets to see it. ``object_to_goal_distance`` bundles the height into a
  vector measured from the cube, which does leak the cube's position -- hence a
  separate term rather than reusing that one.

  Measured from the env origin (not the world) so it is identical across the
  tiled environments.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  height = command.target_pos[:, 2] - env.scene.env_origins[:, 2]
  return height.unsqueeze(-1)


def ee_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """EE linear velocity in EE frame."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 6)
  ee_vel_linear_w = ee_vel_w[:, :3]
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_linear_ee = quat_apply(quat_inv(ee_quat_w), ee_vel_linear_w)
  return ee_vel_linear_ee


def target_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Target position in EE frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (LiftingCommand, MultiCubeLiftingCommand)):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand or "
      f"MultiCubeLiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  target_pos_w = command.target_pos
  target_pos_rel_w = target_pos_w - ee_pos_w
  target_pos_ee = quat_apply(quat_inv(ee_quat_w), target_pos_rel_w)
  return target_pos_ee


def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """RGB observation in CNN-compatible format (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb  # (B, H, W, 3)
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2)  # (B, 3, H, W)
  return rgb_data.float() / 255.0


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
  range_noise: float = 0.0,
  dropout_prob: float = 0.0,
  patch_prob: float = 0.0,
  patch_size: tuple[int, int] = (8, 8),
  scale_err: float = 0.0,
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W).

  The optional arguments model the three ways a real depth stream differs from
  a rendered one that per-pixel Gaussian noise does not cover. All default to
  zero, i.e. the plain rendered depth.

  They are parameters here rather than a ``noise`` term on the cfg because
  their ORDER matters: dropout has to be applied after the range noise, so an
  invalid pixel reads exactly 0.0 and not 0.0 plus a noise draw. A cfg-level
  noise term is applied to whatever this returns and would land on the wrong
  side of that.

  Args:
    range_noise: half-width of per-pixel uniform range noise, in units of the
      normalized output (so 0.01 is 30 mm at a 3 m cutoff).
    dropout_prob: fraction of pixels independently marked invalid. Invalid
      reads become 0.0, which is what a D435 writes and therefore what the
      policy should learn to treat as "no return" -- not a near-field surface.
    patch_prob: fraction of ``patch_size`` BLOCKS marked invalid. This is the
      one that matters: i.i.d. dropout is removed by any 3x3 median, so a
      student trained only against it is trained against nothing. Correlated
      blobs -- specular glare, an occlusion shadow, a surface past the
      stereo baseline -- survive filtering, and they are what actually appears
      on the bench.
    patch_size: (height, width) of the dropout block, in pixels. Must divide
      the frame.
    scale_err: half-width of a multiplicative range error, sampled per
      environment and held for the whole episode. A depth camera's scale error
      is a calibration constant, not per-frame flicker, so resampling it every
      step would model a different (and much easier) thing.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)

  if scale_err > 0.0:
    key = f"_depth_scale_{sensor_name}"
    gain = getattr(env, key, None)
    seen_key = f"{key}_seen"
    seen = getattr(env, seen_key, None)
    if gain is None or gain.shape[0] != env.num_envs:
      gain = torch.ones(env.num_envs, 1, 1, 1, device=depth_data.device)
      # Seeded ABOVE any real counter so the very first call counts as
      # "the episode restarted" and every env draws a gain immediately.
      seen = torch.full(
        (env.num_envs,), 1 << 30, dtype=env.episode_length_buf.dtype, device=env.device
      )
      setattr(env, key, gain)
      setattr(env, seen_key, seen)
    assert seen is not None
    # Resample when the episode counter goes BACKWARDS, not when it reads zero.
    # This function can be called more than once at the same step (the
    # observation manager caches, but direct callers and eval loops do not), and
    # `== 0` would hand out a different gain each time -- turning a calibration
    # constant back into the per-frame flicker this is written to avoid.
    fresh = env.episode_length_buf < seen
    if fresh.any():
      draw = torch.rand(int(fresh.sum()), 1, 1, 1, device=gain.device)
      gain[fresh] = 1.0 + (2.0 * draw - 1.0) * scale_err
    seen.copy_(env.episode_length_buf)
    depth_data = depth_data * gain

  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  out = torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)

  if range_noise > 0.0:
    out = out + (2.0 * torch.rand_like(out) - 1.0) * range_noise
    out = torch.clamp(out, 0.0, 1.0)
  if dropout_prob > 0.0:
    out = torch.where(torch.rand_like(out) < dropout_prob, 0.0, out)
  if patch_prob > 0.0:
    ph, pw = patch_size
    b, _, h, w = out.shape
    if h % ph or w % pw:
      raise ValueError(
        f"patch_size {patch_size} does not divide the {h}x{w} depth frame"
      )
    blocks = torch.rand(b, 1, h // ph, w // pw, device=out.device) < patch_prob
    out = torch.where(
      blocks.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3), 0.0, out
    )
  return out


def camera_segmentation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Per-pixel typed segmentation in (B, 2, H, W) format."""
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  return seg_data.permute(0, 3, 1, 2)  # (B, 2, H, W)


def camera_target_cube_mask(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Binary mask of the target cube selected by a MultiCubeLiftingCommand.

  Output shape: (B, 1, H, W) float32.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  obj_ids = seg_data[..., 0]  # (B, H, W)
  obj_types = seg_data[..., 1]  # (B, H, W)

  command = env.command_manager.get_term(command_name)
  assert isinstance(command, MultiCubeLiftingCommand)
  target_ids = command.target_geom_ids  # (B, K)

  # Only geom hits should participate in the target mask.
  is_geom = obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)
  mask = (obj_ids.unsqueeze(-1) == target_ids.unsqueeze(1).unsqueeze(1)).any(-1)
  mask = mask & is_geom
  return mask.float().unsqueeze(1)  # (B, 1, H, W)
