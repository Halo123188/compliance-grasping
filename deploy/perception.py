"""D435 depth -> the exact tensor the student was trained on.

The sim camera was sized so this is a three-line recipe with no retrain (see the
CameraSensorCfg comment in env_cfgs.py): the modelled intrinsics are the D435's
848x480 depth profile, and mujoco-warp's 4:3 render centre-crops that stream to
exactly 640x480, giving a 73.53 x 58.53 deg field. So stream 848x480, centre-crop
to 640x480, resize to 160x120, and the student sees its training field of view.

THE LAST STEP IS THE ONE THAT COULD VARY, and so far never has: every checkpoint
through round 3 reads 160x120. Round 3 does look at the crop more closely, but
it does so with a stride-1 first convolution rather than a bigger frame, which is
a property of the network and not of this file. The target size is still a policy
property (`profiles.Profile.depth_hw`, asserted against the exported graph)
threaded in here rather than a constant, because the day it does move this is the
step that has to move with it. The FIELD OF VIEW -- what a mistake here would
silently change -- comes from the stream and the crop above and is the same for
every checkpoint.

Streaming a 16:9 profile instead would squash 89.42 deg into pixels the student
learned as 73.53 -- which does not raise, it just makes everything look nearer
the optical axis than it is.

Order matters in the normalisation. Sim clamps to [MIN, CUTOFF] and divides;
a pixel the sensor could not resolve must end up at exactly 0.0, which is what
the student learned "no data" looks like, so invalid pixels are zeroed AFTER the
divide rather than clamped up to the MIN floor.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import calib


def to_observation(depth_m: np.ndarray) -> np.ndarray:
  """(H,W) metric depth in metres -> (1,H,W) float32 in [0,1].

  Zeros in the input are treated as INVALID (the RealSense convention) and stay
  zero, rather than being clamped up to DEPTH_MIN_M and read as "something 1 cm
  from the lens".
  """
  invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
  d = depth_m.astype(np.float32) + calib.DEPTH_AXIS_OFFSET_M
  d = np.clip(d, calib.DEPTH_MIN_M, calib.DEPTH_CUTOFF_M) / calib.DEPTH_CUTOFF_M
  d[invalid] = 0.0
  return d.reshape(1, *d.shape)


def centre_crop_resize(
  depth_m: np.ndarray, out_hw: tuple[int, int] = calib.DEPTH_HW
) -> np.ndarray:
  """(480,848) -> `out_hw`, via the 640x480 centre crop the sim field assumes.

  `out_hw` is the checkpoint's `profiles.Profile.depth_hw`, which is (120,160)
  for every checkpoint to date. The crop is the same whatever it is; only the
  block size changes.
  """
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
  # which is what the renderer's box filter does. Nearest-neighbour instead would
  # alias the object's edges, and at 120x160 the cube is only 16-25 px.
  th, tw = out_hw
  fy, fx = ch // th, cw // tw
  if fy * th != ch or fx * tw != cw:
    raise ValueError(f"crop {(cw, ch)} is not an integer multiple of {out_hw}")
  blocks = crop.reshape(th, fy, tw, fx)
  # Average only over VALID pixels in each block; a block that is entirely
  # invalid stays 0 and to_observation() will keep it there.
  valid = np.isfinite(blocks) & (blocks > 0.0)
  n = valid.sum(axis=(1, 3))
  s = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
  return np.where(n > 0, s / np.maximum(n, 1), 0.0).astype(np.float32)


class RealSenseDepth:
  """The D435 in exactly the profile the policy expects. Requires pyrealsense2.

  Two things here are control-loop parameters wearing a camera's clothes, and
  both were set by a measurement rather than a preference.

  THE FRAME RATE. `wait_for_frames()` blocks until the sensor has a new frame,
  so streaming at 30 fps puts a 33.3 ms floor under a loop whose whole budget is
  20 ms. Measured on the bench at 30 fps: 44 ms per step, i.e. 23 Hz against the
  50 Hz the policy was trained at, with EVERY step late. The 848x480 depth
  profile runs at up to 90 fps and the resolution is what the trained field of
  view depends on -- so when this will not start, change the RATE, never the
  resolution.

  AND THE THREAD. Even at 90 fps a blocking read couples the loop to the
  sensor's phase: finish 1 ms after a frame boundary and you wait a whole frame
  period for the next one, so the loop aliases down to a divisor of the frame
  rate instead of running at its own. The reader thread keeps the most recent
  frame in hand and `read()` returns it without blocking. The cost is that a
  frame can be up to one frame period old -- 11 ms at 90 fps, against the 20 ms
  the observation is fresh for anyway -- and `repeats` counts how often the same
  frame was handed out twice, so the cost is measured rather than assumed.
  """

  def __init__(
    self,
    fps: int = 90,
    threaded: bool = True,
    first_frame_s: float = 5.0,
    preset: str | None = None,
    out_hw: tuple[int, int] = calib.DEPTH_HW,
  ):
    """`out_hw` is the policy's `Profile.depth_hw`; see the module docstring."""
    import pyrealsense2 as rs  # local: the cluster has no camera and no driver

    self._rs = rs
    self.out_hw = out_hw
    w, h = calib.D435_STREAM_WH
    self.fps = fps
    self.pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    try:
      profile = self.pipeline.start(cfg)
    except RuntimeError as exc:
      raise RuntimeError(
        f"could not start the D435 depth stream at {w}x{h} @ {fps} fps: {exc}\n"
        "The RESOLUTION is what the student's field of view comes from and is "
        "not negotiable; the rate is. Try --camera-fps 60, then 30 -- but 30 "
        "cannot sustain the 50 Hz control loop, so treat it as a diagnosis, "
        "not a fix."
      ) from exc
    sensor = profile.get_device().first_depth_sensor()
    self.depth_scale = sensor.get_depth_scale()
    if preset is not None:
      self.set_preset(sensor, preset)

    self._lock = threading.Lock()
    self._latest: np.ndarray | None = None
    self._seq = 0  # frames the sensor has delivered
    self._served = -1  # _seq of the frame the last read() handed out
    self.repeats = 0  # times read() handed out a frame it had already served
    self._error: BaseException | None = None
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    if threaded:
      self._thread = threading.Thread(target=self._pump, name="d435", daemon=True)
      self._thread.start()
      self._await_first_frame(first_frame_s)

  # The preset decides how aggressively the stereo matcher rejects a pixel it is
  # unsure of, i.e. how much of the frame comes back as 0 = no return. That is
  # not a picture-quality setting here: measured on run5, the pixels the sensor
  # blanks and the renderer fills are 2.4% of the frame, and patching just those
  # took the policy's command from 0.205 rad away from its sim-frame behaviour
  # to 0.085. HIGH_ACCURACY blanks the most, HIGH_DENSITY the least.
  PRESETS = {
    "custom": 0,
    "default": 1,
    "hand": 2,
    "high_accuracy": 3,
    "high_density": 4,
    "medium_density": 5,
  }

  def set_preset(self, sensor, name: str) -> None:
    """Apply a D400 visual preset by name. Raises on an unknown name."""
    if name not in self.PRESETS:
      raise ValueError(f"unknown preset {name!r}; choose from {sorted(self.PRESETS)}")
    opt = self._rs.option.visual_preset
    if not sensor.supports(opt):
      print(f"[cam] WARNING: this device has no visual_preset option; {name} ignored")
      return
    sensor.set_option(opt, float(self.PRESETS[name]))
    got = int(sensor.get_option(opt))
    print(f"[cam] visual preset -> {name} (value {got})")

  def _grab(self) -> np.ndarray:
    """One blocking sensor read -> the (1, *out_hw) observation."""
    frames = self.pipeline.wait_for_frames()
    frame = frames.get_depth_frame()
    if not frame:
      raise RuntimeError("no depth frame")
    raw = np.asanyarray(frame.get_data()).astype(np.float32) * self.depth_scale
    return to_observation(centre_crop_resize(raw, self.out_hw))

  def _pump(self) -> None:
    while not self._stop.is_set():
      try:
        obs = self._grab()
      except BaseException as exc:  # noqa: BLE001 -- re-raised in read()
        # close() stops the pipeline out from under a blocked wait_for_frames,
        # which raises here. That is the shutdown path, not a failure, and
        # recording it would make every clean exit look like a camera fault.
        if not self._stop.is_set():
          with self._lock:
            self._error = exc
        return
      with self._lock:
        self._latest = obs
        self._seq += 1

  def _await_first_frame(self, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
      with self._lock:
        if self._error is not None:
          raise RuntimeError("the D435 reader thread died") from self._error
        if self._latest is not None:
          return
      time.sleep(0.005)
    raise RuntimeError(
      f"no depth frame within {timeout_s:.1f}s of starting the stream. "
      "The pipeline started, so the profile is valid and the sensor is not "
      "delivering -- check the USB3 link before anything else."
    )

  def read(self) -> np.ndarray:
    """-> (1, *out_hw) float32 observation. Non-blocking once threaded."""
    if self._thread is None:
      return self._grab()
    with self._lock:
      if self._error is not None:
        raise RuntimeError("the D435 reader thread died") from self._error
      if self._latest is None:
        raise RuntimeError("no depth frame yet")
      if self._seq == self._served:
        self.repeats += 1
      self._served = self._seq
      return self._latest

  def close(self) -> None:
    # Stop the thread BEFORE the pipeline. The other order tears down the
    # pipeline under a blocked wait_for_frames, which is a use-after-free in the
    # SDK rather than an exception. If the join times out, stopping anyway is
    # still the least bad thing available on the way out.
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)
    self.pipeline.stop()
