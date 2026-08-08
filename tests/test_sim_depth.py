"""The render-to-observation recipe, which has to agree with deploy/perception.py.

These two paths describe the same camera and were out of step for a day: the
scripts rendered 640x480 believing `mujoco.Renderer` applies mujoco-warp's
aspect crop, which it does not, so every sim reference image carried a 848/640 =
1.325x horizontal magnification relative to the real stream. The parity test
below is the one that would have caught it.
"""

import numpy as np
import pytest
from deploy import calib
from deploy.perception import centre_crop_resize
from scripts.sim_depth import RENDER_WH, to_observation


def test_renders_at_the_sensor_size_not_the_crop():
  # The whole fix: render where the viewport aspect equals the sensor aspect,
  # so neither renderer's crop branch has anything to do.
  assert RENDER_WH == calib.D435_STREAM_WH
  assert RENDER_WH[0] / RENDER_WH[1] == pytest.approx(4.77 / 2.70, rel=1e-3)


def test_agrees_with_the_deploy_path_pixel_for_pixel():
  rng = np.random.default_rng(0)
  w, h = RENDER_WH
  frame = rng.uniform(0.2, 1.5, (h, w))
  np.testing.assert_allclose(to_observation(frame), centre_crop_resize(frame))


def test_past_the_cutoff_reads_as_no_return():
  w, h = RENDER_WH
  frame = np.full((h, w), 10.0)  # the renderer's far plane, not a distance
  assert (to_observation(frame) == 0.0).all()


def test_a_render_at_the_wrong_size_is_refused():
  # Silently accepting 640x480 is exactly the bug this module exists to fix.
  with pytest.raises(ValueError, match="expected a"):
    to_observation(np.zeros((480, 640)))


def test_the_crop_keeps_the_optical_centre():
  # A centred crop must not move the principal point, or the calibrated
  # intrinsics stop describing the cropped image.
  w, h = RENDER_WH
  cw, ch = calib.D435_CROP_WH
  assert (w - cw) % 2 == 0 and (h - ch) % 2 == 0
  frame = np.full((h, w), 1.0)
  frame[:, (w - cw) // 2 + cw // 2] = 0.5  # a column at the crop's centre
  out = to_observation(frame)
  assert out[:, calib.DEPTH_HW[1] // 2].mean() < out[:, 0].mean()
