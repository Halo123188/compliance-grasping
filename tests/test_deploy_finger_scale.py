"""The counts-per-radian fit, without a gripper on the desk.

The bench half of `deploy/calibrate_finger_scale.py` cannot be tested here, but
the part that turns readings into a number can -- and it is the part that has to
tell a direct drive from a four-bar, because a four-bar has no counts-per-radian
at all and the fit will happily return one anyway.
"""

from __future__ import annotations

import numpy as np
import pytest
from deploy.calibrate_finger_scale import fit_scale

OFFSETS = np.array([-600, -300, 0, 300, 600], dtype=float)


def test_it_recovers_a_known_scale():
  """A direct drive at the scale calib.py currently guesses."""
  k = 651.9
  rad = OFFSETS / k
  fit = fit_scale(OFFSETS, rad)
  assert fit["counts_per_rad"] == pytest.approx(k, rel=1e-9)
  assert fit["counts_at_zero_rad"] == pytest.approx(0.0, abs=1e-6)
  assert fit["resid_max"] < 1e-6


def test_it_recovers_the_scale_the_taught_endpoints_imply():
  """805 counts/rad -- what f1j1's +221 taught open against 179 predicted means."""
  k = 805.0
  fit = fit_scale(OFFSETS, OFFSETS / k)
  assert fit["counts_per_rad"] == pytest.approx(k, rel=1e-9)


def test_a_gauge_that_reads_a_tilted_axis_looks_like_a_scale_error():
  """The failure the procedure exists to prevent, stated as a test.

  An inclinometer on an axis tilted by `t` reads cos(t) of the true rotation, so
  the fit returns a cleanly affine, entirely wrong scale -- residuals cannot
  catch it, only mounting the axis horizontally can.
  """
  k, tilt = 805.0, np.radians(20.0)
  fit = fit_scale(OFFSETS, (OFFSETS / k) * np.cos(tilt))
  assert fit["counts_per_rad"] == pytest.approx(k / np.cos(tilt), rel=1e-9)
  assert fit["resid_max"] < 1e-6  # indistinguishable from a good measurement


def test_a_nonlinear_linkage_shows_up_in_the_residual():
  """A four-bar's angle is not affine in the motor's; the residual must say so."""
  k = 805.0
  rad = OFFSETS / k
  counts = k * rad + 4.0e3 * rad**2  # a few counts of curvature over the sweep
  fit = fit_scale(counts, rad)
  assert fit["resid_max"] > 3.0, "curvature this size must not be reported as affine"
  assert abs(fit["curvature"]) == pytest.approx(4.0e3, rel=1e-6)


def test_it_refuses_a_fit_with_no_residual_to_report():
  with pytest.raises(ValueError, match="at least 3"):
    fit_scale(np.array([0.0, 300.0]), np.array([0.0, 0.4]))


# --- the caliper route --------------------------------------------------------


def test_the_neutral_gap_is_the_model_s_own_number():
  """0 counts on all four motors is 50.00 mm between the flat inner faces."""
  from deploy.calibrate_finger_scale import GAP_TABLE, NEUTRAL_GAP_MM, gap_to_rad

  assert dict(GAP_TABLE)[0.0] == NEUTRAL_GAP_MM == 50.00
  assert gap_to_rad(50.00) == pytest.approx(0.0, abs=1e-9)


def test_gap_inverts_the_table_both_ways():
  from deploy.calibrate_finger_scale import GAP_TABLE, gap_to_rad

  for q, mm in GAP_TABLE:
    assert gap_to_rad(mm) == pytest.approx(q, abs=1e-6)


def test_gap_refuses_to_extrapolate():
  """Off the table means a distal is not at zero, not a small error."""
  from deploy.calibrate_finger_scale import gap_to_rad

  for bad in (0.5, 120.0):
    with pytest.raises(ValueError, match="outside the table"):
      gap_to_rad(bad)


def test_a_caliper_sweep_recovers_the_scale():
  """Readings taken from the table at a true 805 must fit back to 805."""
  from deploy.calibrate_finger_scale import GAP_TABLE, fit_scale, gap_to_rad

  k = 805.0
  counts = np.array([-150.0, -75.0, 0.0, 60.0, 120.0, 180.0])
  # what the caliper would read if the joint really turned at k
  gaps = [
    np.interp(c / k, [p[0] for p in GAP_TABLE], [p[1] for p in GAP_TABLE])
    for c in counts
  ]
  fit = fit_scale(counts, np.array([gap_to_rad(g) for g in gaps]))
  assert fit["counts_per_rad"] == pytest.approx(k, rel=0.02)


def test_the_outside_width_is_what_the_scale_sweep_reads():
  """Neutral is 90.00 mm, and the table inverts on the opening side."""
  from deploy.calibrate_finger_scale import (
    MIN_USABLE_OUTER_MM,
    NEUTRAL_OUTER_MM,
    OUTER_TABLE,
    outer_to_rad,
  )

  assert dict(OUTER_TABLE)[0.0] == NEUTRAL_OUTER_MM == 90.00
  for q, mm in OUTER_TABLE:
    if mm >= MIN_USABLE_OUTER_MM:  # below the kink the inversion is refused
      assert outer_to_rad(mm) == pytest.approx(q, abs=1e-6)


def test_the_outside_width_refuses_the_shallow_branch():
  """A reading at or below the kink is worse than no reading.

  The 2026-08-11 sweep proved it: one point at 89.12 mm moved the answer from
  1028 to 976 counts/rad and manufactured a curvature term out of its own noise.
  """
  from deploy.calibrate_finger_scale import outer_to_rad

  for bad in (85.6, 89.12, 90.0, 91.9):
    with pytest.raises(ValueError, match="kink"):
      outer_to_rad(bad)


def test_an_outside_width_sweep_recovers_the_scale():
  from deploy.calibrate_finger_scale import (
    CALIPER_OFFSETS,
    OUTER_TABLE,
    fit_scale,
    outer_to_rad,
  )

  k = 805.0
  counts = np.array([float(c) for c in CALIPER_OFFSETS])
  q = [p[0] for p in OUTER_TABLE]
  w = [p[1] for p in OUTER_TABLE]
  widths = [np.interp(c / k, q, w) for c in counts]
  fit = fit_scale(counts, np.array([outer_to_rad(x) for x in widths]))
  assert fit["counts_per_rad"] == pytest.approx(k, rel=0.02)


def test_the_width_offset_is_applied_before_the_lookup():
  """A jaw built 3.12 mm narrow is a width constant, not an angle constant.

  Correcting it in the lookup is exact. Leaving it to the fit's intercept is
  only ALMOST right -- once the sweep is confined to the opening branch the map
  is nearly linear there, so the bias is well under a percent. This pins both
  facts, so nobody later "simplifies" the correction away believing it is free,
  and nobody panics about an old sweep that skipped it.
  """
  from deploy.calibrate_finger_scale import (
    CALIPER_OFFSETS,
    OUTER_TABLE,
    fit_scale,
    outer_to_rad,
  )

  k, narrow = 805.0, 3.12
  counts = np.array([float(c) for c in CALIPER_OFFSETS])
  q = [p[0] for p in OUTER_TABLE]
  w = [p[1] for p in OUTER_TABLE]
  reads = np.array([np.interp(c / k, q, w) - narrow for c in counts])

  corrected = fit_scale(counts, np.array([outer_to_rad(x + narrow) for x in reads]))
  assert corrected["counts_per_rad"] == pytest.approx(k, rel=0.02)

  uncorrected = fit_scale(counts, np.array([outer_to_rad(x) for x in reads]))
  bias = (uncorrected["counts_per_rad"] - k) / k
  assert 0.001 < bias < 0.02, f"expected a small positive bias, got {bias:+.4f}"


def test_the_offsets_all_land_on_the_usable_branch():
  """Whatever the true scale, the swept points must be invertible."""
  from deploy.calibrate_finger_scale import (
    CALIPER_OFFSETS,
    MIN_USABLE_OUTER_MM,
    OUTER_TABLE,
  )

  q = [p[0] for p in OUTER_TABLE]
  w = [p[1] for p in OUTER_TABLE]
  # 1028 is the measured scale; 651.9 the old guess. Both must stay on-branch,
  # since the sweep is chosen before the scale is known.
  for k in (651.9, 1028.0):
    widths = [np.interp(c / k, q, w) for c in CALIPER_OFFSETS]
    assert min(widths) >= MIN_USABLE_OUTER_MM, f"k={k} puts a point on the kink"
    assert max(widths) <= w[-1], f"k={k} runs off the end of the table"
