from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.manipulation.mdp.actions import SmoothedJointPositionAction
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


def smoothed_action_command(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """A `SmoothedJointPositionAction`'s internal command, relative to default.

  The limiter carries state the policy does not otherwise see. The original
  term left it out on the argument that it is nearly recoverable from
  `joint_pos` plus the last action, since a stiff servo tracks its target
  closely -- which is true, and beside the point once the limiter saturates.
  Under saturation the published command lags the REQUEST by an unbounded
  amount (16.7x on the SlewCurr teacher, job 61221), so `target - cmd` is not
  small, is not inferable from a tracking error, and is exactly the quantity a
  policy needs in order to stop winding up against a limit it cannot feel.

  Same convention as `joint_pos_rel`: default-relative, so a zero observation
  means "command sitting at the home pose".

  Adding this changes the observation dimension, so a checkpoint trained
  without it cannot be warm-started into a task that has it, in either
  direction. It has to go in the STUDENT group too, not just the teacher's --
  it is proprioception, available on hardware, and a student that cannot see
  the limiter inherits the same blindness.
  """
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, SmoothedJointPositionAction):
    raise TypeError(
      f"smoothed_action_command needs a SmoothedJointPositionAction, "
      f"got {type(term).__name__}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  default = robot.data.default_joint_pos[:, term.target_ids]
  return term.command - default


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


def _episode_store(env: ManagerBasedRlEnv, key: str) -> dict[str, torch.Tensor]:
  """Bag of draws held for a whole episode, hung off ``env`` under ``key``.

  Everything a real depth camera holds constant within one trial belongs here:
  the calibration scale error, the stereo baseline, how much of a given surface
  comes back invalid. Redrawing any of them per frame models a different and
  much easier thing, because a policy can average per-frame flicker away.
  """
  store = getattr(env, key, None)
  if store is None or store["seen"].shape[0] != env.num_envs:
    store = {
      # Seeded ABOVE any real counter so the very first call counts as "the
      # episode restarted" and every env draws immediately.
      "seen": torch.full(
        (env.num_envs,), 1 << 30, dtype=env.episode_length_buf.dtype, device=env.device
      )
    }
    setattr(env, key, store)
  return store


def _held(
  store: dict[str, torch.Tensor],
  fresh: torch.Tensor,
  name: str,
  shape: tuple[int, ...],
  lo: float,
  hi: float,
  device: torch.device,
) -> torch.Tensor:
  """Uniform draw in ``[lo, hi]``, redrawn only for the envs in ``fresh``."""
  buf = store.get(name)
  if buf is None:
    # A buffer appearing part-way through an episode is filled for EVERY env,
    # not just the ones `fresh` names. `fresh` is false for an env that has not
    # reset since the store was created, so seeding with zeros and waiting for
    # the reset would leave the effect switched off -- and switched off in the
    # one way nothing complains about, since zero is a legal draw.
    buf = torch.rand(shape, device=device) * (hi - lo) + lo
    store[name] = buf
    return buf
  if fresh.any():
    buf[fresh] = torch.rand_like(buf[fresh]) * (hi - lo) + lo
  return buf


def _blind_threshold_table(bins: int = 257, samples: int = 1 << 18) -> np.ndarray:
  """Quantiles of a bilinearly-upsampled i.i.d. uniform field.

  ``p`` in :func:`_surface_blind` is an AREA FRACTION -- the measured share of
  a surface that comes back invalid -- so the field has to be cut at its own
  ``p``-quantile and not at ``p`` itself. Bilinear interpolation averages four
  uniforms and the result piles up around 0.5: thresholding the upsampled field
  at 0.05 blinds roughly 0.1% of it, fifty times short of what was asked for.
  The marginal distribution depends on neither the draw nor the object, so one
  table computed once inverts it for every call.
  """
  rng = np.random.default_rng(0)
  a = rng.random(samples)
  b = rng.random(samples)
  u = rng.random((samples, 4))
  g = (
    (1.0 - a) * (1.0 - b) * u[:, 0]
    + a * (1.0 - b) * u[:, 1]
    + (1.0 - a) * b * u[:, 2]
    + a * b * u[:, 3]
  )
  return np.quantile(g, np.linspace(0.0, 1.0, bins))


_BLIND_TAU = _blind_threshold_table()
_BLIND_TAU_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _blind_threshold(prob: torch.Tensor) -> torch.Tensor:
  """Field cut-off giving an expected blinded area fraction of ``prob``."""
  key = (prob.device, prob.dtype)
  table = _BLIND_TAU_CACHE.get(key)
  if table is None:
    table = torch.as_tensor(_BLIND_TAU, dtype=prob.dtype, device=prob.device)
    _BLIND_TAU_CACHE[key] = table
  n = table.numel()
  x = prob.clamp(0.0, 1.0) * (n - 1)
  lo = x.floor().long().clamp(0, n - 2)
  return table[lo] + (x - lo.to(x.dtype)) * (table[lo + 1] - table[lo])


def _occlusion_shadow(
  depth_m: torch.Tensor,
  focal_px: float,
  baseline: torch.Tensor,
  step_m: float,
  max_px: int,
) -> torch.Tensor:
  """(B, 1, H, W) bool: pixels a stereo pair's second imager cannot see.

  A stereo depth camera triangulates against a second imager ``baseline`` metres
  to the side, so wherever a near surface hides the background from that second
  view there is a band of no return -- and on ONE side only, because the two
  imagers sit on one axis. Measured on the bench the right/left density ratio is
  0.09-0.16, so the band goes to the LEFT of every foreground edge and nowhere
  else.

  Width follows from disparity alone. A point at ``z`` shifts ``f*B/z`` pixels
  between the two views, so a step from ``z_bg`` down to ``z_fg`` hides
  ``f*B*(1/z_fg - 1/z_bg)`` pixels of background. Nothing here knows what it is
  looking at: claw, cube and table edge shadow identically, and the only input
  is the rendered depth -- no segmentation, no object list.
  """
  zb = depth_m[..., :-1]  # column u   -- the background side of a step
  zf = depth_m[..., 1:]  # column u+1 -- the foreground side
  width = focal_px * baseline * (1.0 / zf - 1.0 / zb)
  width = torch.where(zb - zf > step_m, width, torch.zeros_like(width))

  shadow = torch.zeros_like(depth_m, dtype=torch.bool)
  n = width.shape[-1]
  # A pixel k columns left of a step is shadowed iff that step is wider than k.
  # The loop is over the SHIFT, not over pixels, so the whole frame costs
  # `max_px` shifted comparisons.
  for k in range(min(max_px, n)):
    shadow[..., : n - k] |= width[..., k:] > k
  return shadow


def _sample_in_bbox(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Bilinearly stretch ``field`` (B, 1, fh, fw) across each mask's bbox.

  Separable on purpose. ``grid_sample`` wants an explicit (B, H, W, 2)
  coordinate tensor, which at a few thousand envs is hundreds of megabytes
  spent describing something that varies along one axis at a time; going along
  the width and then the height is the same arithmetic with only a
  (B, 1, fh, W) intermediate.
  """
  b, _, fh, fw = field.shape
  _, h, w = mask.shape
  ar_w = torch.arange(w, device=mask.device)
  ar_h = torch.arange(h, device=mask.device)
  cols = mask.any(1)  # (B, W)
  rows = mask.any(2)  # (B, H)
  # An empty mask collapses to a degenerate box; those envs are masked off by
  # the caller, and clamp_min keeps the division finite.
  u0 = torch.where(cols, ar_w, w - 1).amin(1).view(b, 1, 1, 1)
  u1 = torch.where(cols, ar_w, 0).amax(1).view(b, 1, 1, 1)
  v0 = torch.where(rows, ar_h, h - 1).amin(1).view(b, 1, 1, 1)
  v1 = torch.where(rows, ar_h, 0).amax(1).view(b, 1, 1, 1)

  # Pixel -> field-node coordinates, clamped at the bbox edge so pixels outside
  # it hold the border value rather than extrapolating.
  pu = ((ar_w.view(1, 1, 1, w) - u0) / (u1 - u0).clamp_min(1) * (fw - 1)).clamp(
    0, fw - 1
  )
  pv = ((ar_h.view(1, 1, h, 1) - v0) / (v1 - v0).clamp_min(1) * (fh - 1)).clamp(
    0, fh - 1
  )
  iu = pu.floor().long().clamp(max=fw - 2)
  iv = pv.floor().long().clamp(max=fh - 2)
  fu = pu - iu
  fv = pv - iv

  col = torch.take_along_dim(field, iu, 3) * (1.0 - fu) + (
    torch.take_along_dim(field, iu + 1, 3) * fu
  )  # (B, 1, fh, W)
  return torch.take_along_dim(col, iv, 2) * (1.0 - fv) + (
    torch.take_along_dim(col, iv + 1, 2) * fv
  )


def _surface_blind(
  seg: torch.Tensor,
  geom_ids: torch.Tensor,
  field: torch.Tensor,
  prob: torch.Tensor,
) -> torch.Tensor:
  """(B, 1, H, W) bool: holes punched in one geom group's rendered surface.

  A real D435 does not return the whole of a surface it can see. Printed
  plastic and a matte cube both scatter enough of the projected IR away from
  the imager that patches come back invalid, and those patches sit ON the
  object and travel with it.

  That is why ``field`` is sampled in the object's own BOUNDING BOX rather than
  in image coordinates. A blob pinned to the image lands on a different part of
  the claw every time the claw moves, and if it is also redrawn each frame a
  policy averages it out over a handful of steps and has been trained against
  nothing at all. Anchored to the bounding box and held for the episode, the
  same patch of the claw stays blind while the claw moves -- which is what the
  bench does.
  """
  mask = (seg[..., 0].unsqueeze(-1) == geom_ids).any(-1) & (
    seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
  )  # (B, H, W)
  g = _sample_in_bbox(field, mask)  # (B, 1, H, W)
  return mask.unsqueeze(1) & (g < _blind_threshold(prob))


def _global_geom_ids(env: ManagerBasedRlEnv, cfg: SceneEntityCfg) -> torch.Tensor:
  """Model-wide geom IDs for a resolved ``SceneEntityCfg``.

  ``SceneEntityCfg.geom_ids`` are ENTITY-LOCAL indices while the segmentation
  buffer stores model-wide MuJoCo IDs, so comparing the two directly matches
  the wrong geoms and does so silently.
  """
  entity: Entity = env.scene[cfg.name]
  return entity.indexing.geom_ids[cfg.geom_ids].to(torch.int64)


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
  range_noise: float = 0.0,
  dropout_prob: float | tuple[float, float] = 0.0,
  scale_err: float = 0.0,
  shadow_focal_px: float = 0.0,
  shadow_baseline: tuple[float, float] = (0.044, 0.056),
  shadow_fill: tuple[float, float] = (0.70, 0.95),
  shadow_step: float = 0.01,
  shadow_max_px: int = 16,
  blind_gripper_cfg: SceneEntityCfg | None = None,
  blind_gripper_prob: tuple[float, float] = (0.0, 0.0),
  blind_object_cfg: SceneEntityCfg | None = None,
  blind_object_prob: tuple[float, float] = (0.0, 0.0),
  blind_field_shape: tuple[int, int] = (12, 16),
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W).

  The optional arguments model the ways a real depth stream differs from a
  rendered one that per-pixel Gaussian noise does not cover. All default to
  zero or ``None``, i.e. the plain rendered depth.

  They are parameters here rather than a ``noise`` term on the cfg because
  their ORDER matters: every invalidating effect has to be applied after the
  range noise, so an invalid pixel reads exactly 0.0 and not 0.0 plus a noise
  draw. A cfg-level noise term is applied to whatever this returns and would
  land on the wrong side of that.

  The two structured effects are computed from the CLEAN rendered depth and
  segmentation but written into the NORMALIZED output. That split is forced
  from both ends: the shadow width is a function of metric ``z``, so it cannot
  be computed after normalization, and a zero written before the ``min_depth``
  clamp comes back out as ``min_depth / cutoff_distance`` instead of 0.0.
  Detecting the depth step on the clean render also keeps ``range_noise``, whose
  half-width is the step threshold, from manufacturing steps on flat surfaces.

  Args:
    range_noise: half-width of per-pixel uniform range noise, in units of the
      normalized output (so 0.01 is 30 mm at a 3 m cutoff).
    dropout_prob: fraction of pixels independently marked invalid, as a scalar
      or a ``(lo, hi)`` band drawn per episode. Invalid reads become 0.0, which
      is what a D435 writes and therefore what the policy should learn to treat
      as "no return" -- not a near-field surface. This is the salt-and-pepper
      component only; the correlated blobs are the two terms below.
    scale_err: half-width of a multiplicative range error, sampled per
      environment and held for the whole episode. A depth camera's scale error
      is a calibration constant, not per-frame flicker, so resampling it every
      step would model a different (and much easier) thing.
    shadow_focal_px: horizontal focal length in pixels of the RENDERED frame.
      Non-zero switches the occlusion shadow on; see :func:`_occlusion_shadow`.
    shadow_baseline: band for the stereo baseline in metres, drawn per episode.
      It is not a fitted quantity: inverting measured shadow widths gives
      50 +- 6 mm, which is the D435's nominal baseline.
    shadow_fill: band for the probability that a pixel inside a shadow band is
      actually invalid, drawn per episode.
    shadow_step: metric depth step, in metres, that counts as a background ->
      foreground edge.
    shadow_max_px: widest shadow modelled, and hence the number of shifted
      comparisons the effect costs.
    blind_gripper_cfg: geoms of the gripper (fingers and pads) to punch holes
      in; ``None`` disables. See :func:`_surface_blind`.
    blind_gripper_prob: band for the blinded area fraction, drawn per episode.
    blind_object_cfg: geoms of the manipulated object to punch holes in.
    blind_object_prob: band for the object's blinded area fraction.
    blind_field_shape: (rows, cols) of the low-resolution random field that is
      bilinearly upsampled across each object's bounding box.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
  device = depth_data.device

  drop_lo, drop_hi = (
    dropout_prob if isinstance(dropout_prob, tuple) else (dropout_prob, dropout_prob)
  )
  shadow_on = shadow_focal_px > 0.0
  blind_on = blind_gripper_cfg is not None or blind_object_cfg is not None

  # Every episode-held draw is taken here, in one place, because they share a
  # single "has this env restarted" counter. Resample when the episode counter
  # goes BACKWARDS, not when it reads zero: this function can be called more
  # than once at the same step (the observation manager caches, but direct
  # callers and eval loops do not), and `== 0` would hand out a different draw
  # each time -- turning calibration constants back into the per-frame flicker
  # holding them is meant to avoid.
  gain = drop = baseline = fill = grip_p = obj_p = grip_f = obj_f = None
  if scale_err > 0.0 or drop_hi > 0.0 or shadow_on or blind_on:
    n, fh, fw = env.num_envs, *blind_field_shape
    store = _episode_store(env, f"_depth_dr_{sensor_name}")
    fresh = env.episode_length_buf < store["seen"]
    if scale_err > 0.0:
      gain = _held(
        store, fresh, "gain", (n, 1, 1, 1), 1 - scale_err, 1 + scale_err, device
      )
    if drop_hi > 0.0:
      drop = _held(store, fresh, "drop", (n, 1, 1, 1), drop_lo, drop_hi, device)
    if shadow_on:
      baseline = _held(store, fresh, "baseline", (n, 1, 1, 1), *shadow_baseline, device)
      fill = _held(store, fresh, "fill", (n, 1, 1, 1), *shadow_fill, device)
    if blind_gripper_cfg is not None:
      grip_p = _held(store, fresh, "grip_p", (n, 1, 1, 1), *blind_gripper_prob, device)
      grip_f = _held(store, fresh, "grip_f", (n, 1, fh, fw), 0.0, 1.0, device)
    if blind_object_cfg is not None:
      obj_p = _held(store, fresh, "obj_p", (n, 1, 1, 1), *blind_object_prob, device)
      obj_f = _held(store, fresh, "obj_f", (n, 1, fh, fw), 0.0, 1.0, device)
    store["seen"].copy_(env.episode_length_buf)

  depth_clean = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  if gain is not None:
    depth_data = depth_data * gain

  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  out = torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)

  if range_noise > 0.0:
    out = out + (2.0 * torch.rand_like(out) - 1.0) * range_noise
    out = torch.clamp(out, 0.0, 1.0)

  if shadow_on:
    assert baseline is not None and fill is not None
    band = _occlusion_shadow(
      depth_clean, shadow_focal_px, baseline, shadow_step, shadow_max_px
    )
    out = torch.where(band & (torch.rand_like(out) < fill), 0.0, out)

  if blind_on:
    seg = sensor.data.segmentation  # (B, H, W, 2)
    assert seg is not None, (
      f"Camera '{sensor_name}' has no segmentation data; surface blinding needs "
      "'segmentation' in the sensor's data_types"
    )
    for cfg, field, prob in (
      (blind_gripper_cfg, grip_f, grip_p),
      (blind_object_cfg, obj_f, obj_p),
    ):
      if cfg is None:
        continue
      assert field is not None and prob is not None
      out = torch.where(
        _surface_blind(seg, _global_geom_ids(env, cfg), field, prob), 0.0, out
      )

  if drop is not None:
    out = torch.where(torch.rand_like(out) < drop, 0.0, out)
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


def object_half_extent(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """The object's half-edge, in metres. [B, 1]

  REQUIRED once the size range is wide enough that one closing depth cannot
  serve both ends of it. Today the policy is size-BLIND and gets away with it:
  it commands full close and the cube stops the fingers wherever it stops them,
  because the finger range's lower bound (-0.0754, a 40 mm gap) is only 10 mm
  inside a 50 mm cube. Open that bound up far enough to pinch a 15 mm object
  (-0.32) and the same full-close command becomes ~0.24 rad of over-travel on a
  50 mm cube -- order 350 N, against the 459 N forced-closure incident that
  threw the cube 607 mm off the table. The policy has to know which object it is
  holding before it is allowed to close that far.

  Read from the live model rather than from a config constant, so it follows
  `dr_cube_scale`'s per-env draw instead of reporting the nominal size.
  """
  obj: Entity = env.scene[object_name]
  gid = int(obj.indexing.geom_ids[0])
  return env.sim.model.geom_size[:, gid, 0].unsqueeze(-1)
