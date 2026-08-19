"""The D435 reader thread, against a fake pyrealsense2.

The threading is the part of `deploy/` that cannot be exercised on the bench
without also exercising the robot, and its failure modes are all silent: a
shutdown race that reads a stopped pipeline, an exception on the reader thread
that never reaches the control loop, a "fresh" frame that is the same one
served twice. So the sensor is faked and the thread is tested directly.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest
from deploy import calib


class _FakeFrame:
  def __init__(self, arr: np.ndarray):
    self._arr = arr

  def get_data(self) -> np.ndarray:
    return self._arr


class _FakeFrames:
  def __init__(self, arr: np.ndarray):
    self._frame = _FakeFrame(arr)

  def get_depth_frame(self) -> _FakeFrame:
    return self._frame


class _FakePipeline:
  """Delivers one frame every `interval` s, numbered so a repeat is visible."""

  def __init__(self, interval: float = 0.005, fail_after: int | None = None):
    self.interval = interval
    self.fail_after = fail_after
    self.n = 0
    self.stopped = False
    self._lock = threading.Lock()

  def start(self, _cfg):
    return _FakeProfile()

  def wait_for_frames(self) -> _FakeFrames:
    time.sleep(self.interval)
    with self._lock:
      if self.stopped:
        raise RuntimeError("pipeline stopped")  # what the SDK does on teardown
      self.n += 1
      n = self.n
    if self.fail_after is not None and n > self.fail_after:
      raise RuntimeError("sensor exploded")
    w, h = calib.D435_STREAM_WH
    # Depth in millimetres, so depth_scale 0.001 turns it into metres. The
    # frame number is encoded in the value, which is how the test tells a
    # fresh frame from a re-served one.
    return _FakeFrames(np.full((h, w), 500 + n, dtype=np.uint16))

  def stop(self) -> None:
    with self._lock:
      self.stopped = True


class _FakeSensor:
  def get_depth_scale(self) -> float:
    return 0.001


class _FakeDevice:
  def first_depth_sensor(self) -> _FakeSensor:
    return _FakeSensor()


class _FakeProfile:
  def get_device(self) -> _FakeDevice:
    return _FakeDevice()


class _FakeConfig:
  def enable_stream(self, *_args) -> None:
    pass


@pytest.fixture
def fake_rs(monkeypatch):
  """Install a fake `pyrealsense2` and hand back the pipeline it will build."""
  pipeline = _FakePipeline()
  rs = types.ModuleType("pyrealsense2")
  rs.pipeline = lambda: pipeline  # type: ignore[attr-defined]
  rs.config = _FakeConfig  # type: ignore[attr-defined]
  rs.stream = types.SimpleNamespace(depth=0)  # type: ignore[attr-defined]
  rs.format = types.SimpleNamespace(z16=0)  # type: ignore[attr-defined]
  monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
  return pipeline


def _frame_number(obs: np.ndarray) -> int:
  """Invert the observation's normalisation to recover the encoded number."""
  metres = float(obs.flat[0]) * calib.DEPTH_CUTOFF_M
  return round(metres * 1000) - 500


def test_threaded_read_is_fresh_and_shaped(fake_rs):
  from deploy.perception import RealSenseDepth

  cam = RealSenseDepth(fps=90)
  try:
    first = cam.read()
    assert first.shape == (1, *calib.DEPTH_HW)
    assert first.dtype == np.float32
    # Wait past a frame period and the reader must have moved on by itself,
    # which is the whole point of the thread.
    n0 = _frame_number(first)
    time.sleep(0.05)
    assert _frame_number(cam.read()) > n0
  finally:
    cam.close()


def test_repeats_count_the_loop_outrunning_the_sensor(fake_rs):
  from deploy.perception import RealSenseDepth

  fake_rs.interval = 0.05  # a slow sensor, so back-to-back reads must repeat
  cam = RealSenseDepth(fps=90)
  try:
    cam.read()
    for _ in range(5):
      cam.read()
    assert cam.repeats == 5
  finally:
    cam.close()


def test_close_is_not_reported_as_a_camera_fault(fake_rs):
  from deploy.perception import RealSenseDepth

  cam = RealSenseDepth(fps=90)
  cam.close()
  assert cam._thread is not None
  assert not cam._thread.is_alive()
  # The teardown raises inside wait_for_frames; recording it would make every
  # clean exit look like a failure.
  assert cam._error is None


def test_reader_thread_error_surfaces_from_read(fake_rs):
  from deploy.perception import RealSenseDepth

  fake_rs.fail_after = 2
  cam = RealSenseDepth(fps=90)
  try:
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
      try:
        cam.read()
      except RuntimeError as exc:
        assert "reader thread died" in str(exc)
        return
      time.sleep(0.005)
    pytest.fail("the thread's exception never reached read()")
  finally:
    cam.close()


def test_blocking_mode_still_works(fake_rs):
  from deploy.perception import RealSenseDepth

  cam = RealSenseDepth(fps=90, threaded=False)
  try:
    assert cam._thread is None
    assert _frame_number(cam.read()) < _frame_number(cam.read())
    assert cam.repeats == 0
  finally:
    cam.close()


# --- the frame size is the checkpoint's ---------------------------------------


def test_the_crop_is_the_same_at_either_resolution():
  """The output size is a sampling choice; the FIELD is not up for grabs.

  The trained field of view comes from the 848x480 stream and the 640x480
  centre crop, both of which are fixed, and only the block size moves with
  `out_hw`. So a finer output has to average down to the coarser one exactly --
  if it does not, the crop moved, which is the mistake this guards.
  """
  from deploy.perception import centre_crop_resize

  w, h = calib.D435_STREAM_WH
  rng = np.random.default_rng(0)
  depth = rng.uniform(0.3, 1.0, size=(h, w)).astype(np.float32)

  small = centre_crop_resize(depth, (120, 160))
  large = centre_crop_resize(depth, (240, 320))
  assert (small.shape, large.shape) == ((120, 160), (240, 320))
  np.testing.assert_allclose(
    small, large.reshape(120, 2, 160, 2).mean(axis=(1, 3)), rtol=1e-5
  )


def test_a_target_the_crop_does_not_divide_by_is_refused():
  """Resampling by a non-integer factor is a different filter, so it raises."""
  from deploy.perception import centre_crop_resize

  w, h = calib.D435_STREAM_WH
  with pytest.raises(ValueError, match="integer multiple"):
    centre_crop_resize(np.ones((h, w), np.float32), (100, 130))


def test_the_reader_delivers_the_size_the_policy_asked_for(fake_rs):
  """The camera takes its output size from the checkpoint, not from calib.py.

  Every checkpoint to date asks for calib.DEPTH_HW; a size that is not the
  default is used here precisely so the plumbing is exercised rather than the
  constant.
  """
  from deploy.perception import RealSenseDepth

  cam = RealSenseDepth(fps=90, out_hw=(240, 320))
  try:
    assert cam.read().shape == (1, 240, 320)
  finally:
    cam.close()
