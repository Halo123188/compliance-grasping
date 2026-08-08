"""The camera check, against synthetic frames with a known answer.

The point of `deploy/check_camera.py` is to tell a correctly mounted camera
from one that is 180 degrees out, so the test is exactly that: synthesise a
bench with a cube where the model says it should be, and again with the frame
point-reflected, and require the verdict to name each one.
"""

from __future__ import annotations

import numpy as np
import pytest
from deploy import calib, check_camera


def test_projection_matches_the_compiled_model():
  # From the compiled wide-claw model: a cube resting on the foam at
  # base-frame (0.46, 0) lands here in the 160x120 observation. Regenerate
  # these by projecting through `cam_xpos`/`cam_xmat` of the compiled
  # "scene_cam" -- NOT through `check_camera.project`, which is the thing under
  # test. They moved on 2026-08-07 (from 78.28, 49.13, 0.3575) when the camera
  # extrinsic picked up its measured +19.94 mm lateral correction.
  u, v, rng = check_camera.project((0.46, 0.0, check_camera.RESTING_Z_BASE))
  assert (round(u, 2), round(v, 2), round(rng, 4)) == (84.25, 48.6, 0.3576)


def test_taking_the_foam_off_moves_the_cube_further_than_the_tolerance():
  """Why `--surface-z` exists rather than a hardcoded resting height.

  The two benches put the cube 50 mm apart in z, and if that is more than the
  verdict's own tolerance then guessing wrong reports a camera fault that is
  not there. Pinning it here so the flag cannot quietly be dropped again.
  """
  foam = check_camera.WORK_SURFACE_Z_BASE + check_camera.CUBE_HALF_M
  bare = check_camera.BARE_TABLE_Z_BASE + check_camera.CUBE_HALF_M
  assert round(foam - bare, 6) == 0.050
  assert round(foam, 6) == round(check_camera.RESTING_Z_BASE, 6)

  uf, vf, _ = check_camera.project((*check_camera.CHECK_CUBE_XY, foam))
  ub, vb, _ = check_camera.project((*check_camera.CHECK_CUBE_XY, bare))
  assert float(np.hypot(uf - ub, vf - vb)) > 8.0  # the verdict's tolerance


def test_world_y_moves_the_image_left():
  """The check the whole script exists for: +y must go LEFT, not right."""
  u0, _, _ = check_camera.project((0.46, 0.0, check_camera.RESTING_Z_BASE))
  u_left, _, _ = check_camera.project((0.46, 0.10, check_camera.RESTING_Z_BASE))
  assert u_left < u0 - 25  # ~30 px for 100 mm at this range


def test_largest_blob_picks_the_biggest_component():
  mask = np.zeros((10, 10), bool)
  mask[2:5, 3:6] = True  # 9 px
  mask[8, 8] = True  # 1 px, must lose
  blob = check_camera.largest_blob(mask)
  assert blob.sum() == 9
  rows, cols = np.nonzero(blob)
  assert (rows.mean(), cols.mean()) == (3.0, 4.0)
  assert check_camera.largest_blob(np.zeros((4, 4), bool)).sum() == 0


CHECK_XY = check_camera.CHECK_CUBE_XY
CUBE = (*CHECK_XY, check_camera.RESTING_Z_BASE)


def _bench_with_cube(u: float, v: float, rng: float) -> tuple[np.ndarray, np.ndarray]:
  """(empty bench, bench + a cube blob at u,v) as metric depth frames.

  The bench sits 50 mm BEHIND the cube face, so the difference clears the
  script's 20 mm threshold the way a real 50 mm cube on foam does.
  """
  base = np.full(calib.DEPTH_HW, rng + 0.05, dtype=float)
  frame = base.copy()
  r, c = int(round(v)), int(round(u))
  frame[r - 3 : r + 4, c - 3 : c + 4] = rng
  return base, frame


def _verdict(monkeypatch, capsys, frame, baseline, tmp_path, xy=CHECK_XY) -> str:
  path = tmp_path / "baseline.npz"
  np.savez_compressed(path, depth_m=baseline)
  monkeypatch.setattr(check_camera, "grab", lambda *_a, **_k: frame)
  monkeypatch.setattr(
    "sys.argv",
    ["check_camera", "--baseline", str(path), "--cube-xy", str(xy[0]), str(xy[1])],
  )
  check_camera.main()
  return capsys.readouterr().out


def test_correct_mounting_reads_as_ok(monkeypatch, capsys, tmp_path):
  u, v, rng = check_camera.project(CUBE)
  base, frame = _bench_with_cube(u, v, rng)
  out = _verdict(monkeypatch, capsys, frame, base, tmp_path)
  assert "OK:" in out
  assert "AS MODELLED" in out


def test_a_camera_mounted_upside_down_is_named(monkeypatch, capsys, tmp_path):
  u, v, rng = check_camera.project(CUBE)
  h, w = calib.DEPTH_HW
  base, frame = _bench_with_cube(w - 1 - u, h - 1 - v, rng)
  out = _verdict(monkeypatch, capsys, frame, base, tmp_path)
  assert "THE CAMERA IS ROTATED 180" in out
  assert "do NOT flip the image" in out


def test_a_mirrored_y_axis_is_named(monkeypatch, capsys, tmp_path):
  u, v, rng = check_camera.project(CUBE)
  _, w = calib.DEPTH_HW[0], calib.DEPTH_HW[1]
  base, frame = _bench_with_cube(w - 1 - u, v, rng)
  out = _verdict(monkeypatch, capsys, frame, base, tmp_path)
  assert "THE CAMERA IS MIRRORED LEFT-RIGHT" in out


def test_a_centreline_placement_refuses_to_rule_out_a_mirror(
  monkeypatch, capsys, tmp_path
):
  """Why CHECK_CUBE_XY is off-centre: at y=0 a mirror is 2.4 px from the truth.

  A correct camera and a left-right mirrored one predict almost the same pixel
  there, so "closest = AS MODELLED" carries no information. The check has to
  say so instead of reporting a confident OK.
  """
  centre = (0.46, 0.0, check_camera.RESTING_Z_BASE)
  u, v, rng = check_camera.project(centre)
  base, frame = _bench_with_cube(u, v, rng)
  out = _verdict(monkeypatch, capsys, frame, base, tmp_path, xy=centre[:2])
  assert "cannot rule out: mirrored left-right" in out
  assert "OK as far as this placement can tell" in out
  assert "OK:" not in out


def test_an_empty_difference_refuses_rather_than_guessing(
  monkeypatch, capsys, tmp_path
):
  base = np.full(calib.DEPTH_HW, 0.42, dtype=float)
  path = tmp_path / "baseline.npz"
  np.savez_compressed(path, depth_m=base)
  monkeypatch.setattr(check_camera, "grab", lambda *_a, **_k: base.copy())
  monkeypatch.setattr("sys.argv", ["check_camera", "--baseline", str(path)])
  assert check_camera.main() == 1
  assert "only 0 pixels changed" in capsys.readouterr().out


def test_a_point_behind_the_camera_raises():
  with pytest.raises(ValueError, match="behind the camera"):
    check_camera.project((-1.0, 0.0, 0.0))
