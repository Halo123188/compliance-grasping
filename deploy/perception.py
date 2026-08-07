"""D435 depth -> the exact tensor the student was trained on.

The sim camera was sized so this is a three-line recipe with no retrain (see the
CameraSensorCfg comment in env_cfgs.py): the modelled intrinsics are the D435's
848x480 depth profile, and mujoco-warp's 4:3 render centre-crops that stream to
exactly 640x480, giving a 73.53 x 58.53 deg field. So stream 848x480, centre-crop
to 640x480, resize to 160x120, and the student sees its training field of view.

Streaming a 16:9 profile instead would squash 89.42 deg into pixels the student
learned as 73.53 -- which does not raise, it just makes everything look nearer
the optical axis than it is.

Order matters in the normalisation. Sim clamps to [MIN, CUTOFF] and divides;
a pixel the sensor could not resolve must end up at exactly 0.0, which is what
the student learned "no data" looks like, so invalid pixels are zeroed AFTER the
divide rather than clamped up to the MIN floor.
"""

from __future__ import annotations

import numpy as np

from . import calib


def to_observation(depth_m: np.ndarray) -> np.ndarray:
  """(H,W) metric depth in metres -> (1,120,160) float32 in [0,1].

  Zeros in the input are treated as INVALID (the RealSense convention) and stay
  zero, rather than being clamped up to DEPTH_MIN_M and read as "something 1 cm
  from the lens".
  """
  invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
  d = depth_m.astype(np.float32) + calib.DEPTH_AXIS_OFFSET_M
  d = np.clip(d, calib.DEPTH_MIN_M, calib.DEPTH_CUTOFF_M) / calib.DEPTH_CUTOFF_M
  d[invalid] = 0.0
  return d.reshape(1, *d.shape)


def centre_crop_resize(depth_m: np.ndarray) -> np.ndarray:
  """(480,848) -> (120,160), via the 640x480 centre crop the sim field assumes."""
  h, w = depth_m.shape
  cw, ch = calib.D435_CROP_WH
  if (w, h) != calib.D435_STREAM_WH:
    raise ValueError(
      f"expected an {calib.D435_STREAM_WH} depth frame, got {(w, h)}. "
      "The field of view the student was trained on comes from THIS profile; "
      "see the note in this module."
    )
  x0, y0 = (w - cw) // 2, (h - ch) // 2
  crop = depth_m[y0 : y0 + ch, x0 : x0 + cw]

  # Area-average downsample by the exact integer factor (640/160 = 480/120 = 4),
  # which is what the renderer's box filter does. Nearest-neighbour instead
  # would alias the cube's edges, and the cube is 16-25 px.
  th, tw = calib.DEPTH_HW
  fy, fx = ch // th, cw // tw
  if fy * th != ch or fx * tw != cw:
    raise ValueError(f"crop {(cw, ch)} is not an integer multiple of {calib.DEPTH_HW}")
  blocks = crop.reshape(th, fy, tw, fx)
  # Average only over VALID pixels in each block; a block that is entirely
  # invalid stays 0 and to_observation() will keep it there.
  valid = np.isfinite(blocks) & (blocks > 0.0)
  n = valid.sum(axis=(1, 3))
  s = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
  return np.where(n > 0, s / np.maximum(n, 1), 0.0).astype(np.float32)


class RealSenseDepth:
  """The D435 in exactly the profile the policy expects. Requires pyrealsense2."""

  def __init__(self, fps: int = 30):
    import pyrealsense2 as rs  # local: the cluster has no camera and no driver

    self._rs = rs
    w, h = calib.D435_STREAM_WH
    self.pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    profile = self.pipeline.start(cfg)
    self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

  def read(self) -> np.ndarray:
    """Blocking. -> (1,120,160) float32 observation, ready for the policy."""
    frames = self.pipeline.wait_for_frames()
    frame = frames.get_depth_frame()
    if not frame:
      raise RuntimeError("no depth frame")
    raw = np.asanyarray(frame.get_data()).astype(np.float32) * self.depth_scale
    return to_observation(centre_crop_resize(raw))

  def close(self) -> None:
    self.pipeline.stop()
