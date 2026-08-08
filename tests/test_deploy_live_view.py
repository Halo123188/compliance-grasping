"""The two comparison images and the shift fit behind deploy/live_view.py."""

import numpy as np
from deploy.live_view import best_shift, diff_rgb, overlay_rgb


def _frame(value: float = 0.4) -> np.ndarray:
  return np.full((12, 16), value)


def test_diff_is_blue_when_real_is_nearer():
  real, sim = _frame(0.30), _frame(0.40)
  out = diff_rgb(real, sim)
  assert (out[..., 2] > 0).all()
  assert (out[..., 0] == 0).all()


def test_a_pixel_valid_in_only_one_frame_saturates_rather_than_getting_a_third_colour():
  real, sim = _frame(), _frame()
  real[3, 4] = 0.0  # sensor blank, renderer solid -> "further than anything"
  sim[7, 9] = 0.0  # the reverse
  out = diff_rgb(real, sim)
  assert tuple(out[3, 4]) == (255, 0, 0)
  assert tuple(out[7, 9]) == (0, 0, 255)
  # green is gone: it used to mark these and drowned out the alignment signal
  assert (out[..., 1] == 0).all()


def test_overlay_puts_sim_on_red_and_real_on_cyan():
  real, sim = _frame(0.30), _frame(0.90)
  out = overlay_rgb(real, sim)
  # near is bright, so the nearer frame is the brighter channel
  assert out[..., 1].mean() > out[..., 0].mean()
  assert (out[..., 1] == out[..., 2]).all()


def test_overlay_of_identical_frames_is_neutral_grey():
  f = _frame(0.5)
  out = overlay_rgb(f, f)
  assert (out[..., 0] == out[..., 1]).all()
  assert (out[..., 1] == out[..., 2]).all()


def test_best_shift_recovers_a_known_translation():
  rng = np.random.default_rng(0)
  sim = rng.uniform(0.25, 0.6, (40, 60))
  real = np.roll(sim, 3, axis=1)  # the real scene sits 3 px to the right
  du, dv, err = best_shift(real, sim, slice(0, 40))
  assert (du, dv) == (3, 0)
  assert err < 1e-9


def test_best_shift_ignores_pixels_that_are_invalid_in_either_frame():
  rng = np.random.default_rng(1)
  sim = rng.uniform(0.25, 0.6, (40, 60))
  real = np.roll(sim, -2, axis=1)
  real[::5] = 0.0
  du, dv, _ = best_shift(real, sim, slice(0, 40))
  assert (du, dv) == (-2, 0)
