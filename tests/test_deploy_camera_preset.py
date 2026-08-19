"""`RealSenseDepth` must accept the preset its own CLI has always advertised.

`live_view.py` has offered ``--camera-preset`` since it was written, documented
as "high_density blanks the fewest pixels, high_accuracy the most", and passed
it as ``RealSenseDepth(fps=..., preset=...)``. ``RealSenseDepth.__init__`` only
ever took ``fps``, so every run of the live view WITH a camera died on
``TypeError: unexpected keyword argument 'preset'``. Nothing caught it because
the cluster has no camera and no driver, so the import is local and the class is
never constructed in CI.

These tests fake just enough of ``pyrealsense2`` to construct the class, which is
the only way to cover this without hardware. They pin the signature and the
behaviours that matter: a valid preset reaches the sensor, an invalid one raises
instead of silently leaving the camera on its factory default, and a device
without the option says so rather than pretending. A wrong preset is invisible
in the depth image and would show up only as a policy that behaves differently
from the one that was tuned.
"""

import sys
from unittest.mock import MagicMock

import pytest


def _install_fake_pyrealsense2(
  monkeypatch, *, supports=True
) -> tuple[MagicMock, MagicMock]:
  """A pyrealsense2 stub with the surface RealSenseDepth touches.

  A MagicMock rather than a real ModuleType: `import pyrealsense2 as rs` only
  looks the name up in `sys.modules`, so it does not have to be a module, and a
  mock takes attribute assignment without the type checker objecting.
  """
  rs = MagicMock(name="pyrealsense2")

  sensor = MagicMock()
  sensor.get_depth_scale.return_value = 0.001
  sensor.supports.return_value = supports
  # `set_preset` reads the value back to report what the device took; the real
  # option is a float and `int()` of a bare MagicMock raises.
  sensor.get_option.side_effect = lambda _opt: float(
    sensor.set_option.call_args[0][1] if sensor.set_option.call_args else 0.0
  )

  profile = MagicMock()
  profile.get_device.return_value.first_depth_sensor.return_value = sensor
  pipeline = MagicMock()
  pipeline.start.return_value = profile

  rs.pipeline = MagicMock(return_value=pipeline)
  rs.config = MagicMock()
  rs.stream = MagicMock()
  rs.format = MagicMock()
  rs.option = MagicMock()

  monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
  return rs, sensor


def _make(preset=None, **kw):
  """Construct without the reader thread -- these tests never call read()."""
  from deploy.perception import RealSenseDepth

  return RealSenseDepth(preset=preset, threaded=False, **kw)


def test_constructing_with_a_preset_does_not_raise(monkeypatch):
  """The regression: this call is exactly what live_view.py makes."""
  _install_fake_pyrealsense2(monkeypatch)

  _make(fps=90, preset="high_density")


def test_the_preset_actually_reaches_the_sensor(monkeypatch):
  rs, sensor = _install_fake_pyrealsense2(monkeypatch)

  _make(preset="high_accuracy")
  sensor.set_option.assert_called_once_with(rs.option.visual_preset, 3.0)


def test_no_preset_leaves_the_camera_alone(monkeypatch):
  """The default path must not touch visual_preset at all."""
  _, sensor = _install_fake_pyrealsense2(monkeypatch)

  _make()
  sensor.set_option.assert_not_called()


def test_an_unknown_preset_raises_rather_than_being_ignored(monkeypatch):
  _install_fake_pyrealsense2(monkeypatch)

  with pytest.raises(ValueError, match="unknown preset"):
    _make(preset="hgh_density")


def test_preset_names_are_case_insensitive(monkeypatch):
  """The driver spells them High_Density; the CLI documents high_density."""
  rs, sensor = _install_fake_pyrealsense2(monkeypatch)

  _make(preset="HIGH_DENSITY")
  sensor.set_option.assert_called_once_with(rs.option.visual_preset, 4.0)


def test_a_device_without_the_option_is_skipped_not_crashed(monkeypatch):
  """Not every D400 exposes visual_preset; that is a warning, not a failure."""
  _, sensor = _install_fake_pyrealsense2(monkeypatch, supports=False)

  _make(preset="high_density")
  sensor.set_option.assert_not_called()
