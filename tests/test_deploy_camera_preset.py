"""`RealSenseDepth` must accept the preset its own CLI has always advertised.

`live_view.py` has offered ``--camera-preset`` since it was written, documented
as "high_density blanks the fewest pixels, high_accuracy the most", and passed
it as ``RealSenseDepth(fps=..., preset=...)``. ``RealSenseDepth.__init__`` only
ever took ``fps``, so every run of the live view WITH a camera died on
``TypeError: unexpected keyword argument 'preset'``. Nothing caught it because
the cluster has no camera and no driver, so the import is local and the class is
never constructed in CI.

These tests fake just enough of ``pyrealsense2`` to construct the class, which is
the only way to cover this without hardware. They pin the signature and the two
behaviours that matter: a valid preset reaches the sensor, and an invalid one
raises instead of silently leaving the camera on its factory default -- a wrong
preset is invisible in the depth image and would show up only as a policy that
behaves differently from the one that was tuned.
"""

import sys
from unittest.mock import MagicMock

import pytest


class _FakePreset:
  """Stands in for one member of rs.rs400_visual_preset."""

  def __init__(self, name: str, value: int):
    self.name = name
    self._value = value

  def __int__(self) -> int:
    return self._value


def _install_fake_pyrealsense2(monkeypatch) -> tuple[MagicMock, MagicMock]:
  """A pyrealsense2 stub with the surface RealSenseDepth touches.

  A MagicMock rather than a real ModuleType: `import pyrealsense2 as rs` only
  looks the name up in `sys.modules`, so it does not have to be a module, and a
  mock takes attribute assignment without the type checker objecting.
  """
  rs = MagicMock(name="pyrealsense2")

  sensor = MagicMock()
  sensor.get_depth_scale.return_value = 0.001

  profile = MagicMock()
  profile.get_device.return_value.first_depth_sensor.return_value = sensor
  pipeline = MagicMock()
  pipeline.start.return_value = profile

  rs.pipeline = MagicMock(return_value=pipeline)
  rs.config = MagicMock()
  rs.stream = MagicMock()
  rs.format = MagicMock()
  rs.option = MagicMock()
  rs.rs400_visual_preset = MagicMock()
  rs.rs400_visual_preset.__members__ = {
    "default": _FakePreset("Default", 0),
    "high_accuracy": _FakePreset("High_Accuracy", 3),
    "high_density": _FakePreset("High_Density", 4),
  }

  monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
  return rs, sensor


def test_constructing_with_a_preset_does_not_raise(monkeypatch):
  """The regression: this call is exactly what live_view.py makes."""
  rs, _ = _install_fake_pyrealsense2(monkeypatch)
  from deploy.perception import RealSenseDepth

  RealSenseDepth(fps=90, preset="high_density")


def test_the_preset_actually_reaches_the_sensor(monkeypatch):
  rs, sensor = _install_fake_pyrealsense2(monkeypatch)
  from deploy.perception import RealSenseDepth

  RealSenseDepth(preset="high_accuracy")
  sensor.set_option.assert_called_once_with(rs.option.visual_preset, 3)


def test_no_preset_leaves_the_camera_alone(monkeypatch):
  """The default path must not touch visual_preset at all."""
  _, sensor = _install_fake_pyrealsense2(monkeypatch)
  from deploy.perception import RealSenseDepth

  RealSenseDepth()
  sensor.set_option.assert_not_called()


def test_an_unknown_preset_raises_rather_than_being_ignored(monkeypatch):
  _install_fake_pyrealsense2(monkeypatch)
  from deploy.perception import RealSenseDepth

  with pytest.raises(ValueError, match="unknown D400 visual preset"):
    RealSenseDepth(preset="hgh_density")


def test_preset_names_are_case_insensitive(monkeypatch):
  """The driver spells them High_Density; the CLI documents high_density."""
  rs, sensor = _install_fake_pyrealsense2(monkeypatch)
  from deploy.perception import RealSenseDepth

  RealSenseDepth(preset="HIGH_DENSITY")
  sensor.set_option.assert_called_once_with(rs.option.visual_preset, 4)
