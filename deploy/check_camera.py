"""Is the camera pointing where the policy thinks it is? Check, don't assume.

The depth image is the one input to this policy that nothing else cross-checks.
The joint vector is verified by `--require-home`, the action feedback is the
policy's own output, and a wrong `goal_height` is range-checked. A camera that
is mounted 180 degrees out, or mirrored, or aimed 10 degrees off, produces a
perfectly plausible depth frame and a policy that reaches confidently at the
wrong place -- which reads on the bench as "the policy is bad".

So this projects a known point through the SIM's camera model and asks the real
sensor where it actually sees it.

  # 1. clear the bench, capture the reference
  .venv-deploy/bin/python -m deploy.check_camera --save baseline.npz

  # 2. put the cube down at a MEASURED spot, capture and compare
  .venv-deploy/bin/python -m deploy.check_camera --baseline baseline.npz \\
      --cube-xy 0.5207 0.1016 --surface-z -0.015

Both frames must be taken at the SAME ARM POSE. The cube is found by
differencing against the empty bench, so a claw that moved between the two is a
second blob competing with the first -- and at the policy home pose the claw
lands within 7 px of the default cube position on a bare table, which is close
enough to merge into one.

Differencing against an empty bench rather than hunting for "the nearest thing"
is deliberate: the claw and the camera mount are in frame too, and at the home
pose they are nearer than the cube. The difference image has exactly one thing
in it.

FRAME. Metres, in the ROBOT BASE frame -- the one RDK reports `tcp_pose` in.
Right-handed, and the sim's base body carries no rotation, so these are also
the sim world's axes with the origin moved:

  origin  the TOP of the arm's 15 mm mounting plate, i.e. the arm's own base
          flange. In the sim scene that is 0.365 m above the FLOOR.
  +x      forward, out over the bench (the cube spawns around x = 0.46)
  +y      to the arm's LEFT looking along +x
  +z      up

  z in this frame        table top -0.015 | ARM BASE 0.000 | foam top +0.035

  A 50 mm cube's CENTRE is 0.025 above whatever it rests on, so +0.060 on the
  foam and +0.010 on the bare table. Which one applies is `--surface-z`, and it
  is a measurement: fit the table plane in a `capture_sweep` run rather than
  assuming, because 50 mm is 11.9 px here and the verdict tolerates 8.

DO NOT confuse this with `goal_height`, which is measured from the FLOOR and so
runs 0.50-0.60 for the same bench. calib.py keeps heights in floor coordinates
because that is the frame the policy's lift command lives in; this file works
in base coordinates because that is the frame you can measure against the robot.

Confirmed against hardware at the home pose: RDK reported the TCP at
[0.465, 0.000, 0.260] where the sim's flange sits at [0.463, -0.001, ...], so
the origin and the two horizontal axes agree to about 2 mm. The z of the TCP
does NOT match the sim's hand mount (0.2295) by 30 mm -- a configured tool
offset, most likely -- which is worth knowing but does not touch this check,
since nothing here goes through the TCP.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from . import calib

# --- the sim camera, in the robot base frame --------------------------------- #
# Regenerate with `scripts/wide_calibrate.py`-style access to the compiled model:
# take `cam_xpos`/`cam_xmat` of "scene_cam" minus the "base" body's xpos (the
# base has identity rotation, so only the translation differs). Duplicated here
# for the same reason as everything in calib.py -- the robot host does not have
# mjlab -- and it is a claim about the REAL mount that this script is the test of.
#
# Rows of CAM_ROT_BASE are the world axes; the COLUMNS are the camera's own
# axes expressed in the base frame, i.e. column 0 is image-right, column 1 is
# image-up, column 2 is image-backward (MuJoCo cameras look down -z).
#
# y MOVED +19.94 mm on 2026-08-07 (was -0.005707), from the nine-pose sweep in
# `deploy/capture_sweep.py` + `scripts/fit_camera_pose.py`: the camera really
# sits that far to the image-left of where the CAD chain put it. This constant
# and `arm_cfg.CALIB_CAM_OFFSET` are the same measurement written in two frames
# and MUST move together -- see the note there for why it is a translation
# rather than a rotation, and for the joint1 ambiguity this script exists to
# settle. Anything that reads a mismatch here is a real disagreement.
CAM_POS_BASE = (0.126105, 0.014232, 0.194425)
CAM_ROT_BASE = (
  (0.000091213, 0.477064930, -0.878868047),
  (-0.999993765, 0.003146045, 0.001603944),
  (0.003530144, 0.878862421, 0.477062242),
)
# Intrinsics of the modelled 848x480 depth profile (arm_cfg.CAMERA_FOCALPIXEL /
# CAMERA_PRINCIPALPIXEL, the latter already converted out of MuJoCo's
# centre-relative convention).
CAM_FOCAL_PX = (428.3054, 428.3054)
CAM_PRINCIPAL_PX = (424.4773, 243.6254)

# Bench geometry in the base frame, from the sim scene: the foam top and the
# centre of a 50 mm cube resting on it. calib.py carries these relative to the
# FLOOR because that is the frame goal_height lives in; the camera check wants
# them relative to the arm, and ARM_BASE_Z is the difference.
WORK_SURFACE_Z_BASE = calib.WORK_SURFACE_Z - calib.ARM_BASE_Z  # 0.035
RESTING_Z_BASE = calib.RESTING_Z - calib.ARM_BASE_Z  # 0.060
# Half the 50 mm cube, i.e. how far its CENTRE sits above whatever it rests on.
CUBE_HALF_M = calib.RESTING_Z - calib.WORK_SURFACE_Z  # 0.025
# The bench with the foam TAKEN OFF. Not a guess: fitting the table plane in the
# nine frames of a `capture_sweep` run puts it at -15.0 mm, consistent to 0.1 mm
# across every pose, which is this number. The same fit reads +35.0 mm with the
# foam on, so the two are 50 mm apart and trivially distinguishable -- measure,
# do not assume, because at bench range 50 mm is 11.9 px and the verdict below
# tolerates 8.
BARE_TABLE_Z_BASE = -0.015
# DELIBERATELY OFF-CENTRE, and this is the whole design of the check. At the
# scene's nominal (0.46, 0) the cube lands at u = 84.3 in a 160-wide frame, so a
# LEFT-RIGHT MIRRORED camera predicts u = 75.8 and the two hypotheses are only
# 8.5 px apart -- within the blob centroid's own spread, so the measurement
# would happily report "OK" for a mirrored camera. At (0.52, 0.10) the nearest
# wrong hypothesis is 44.8 px away. Still inside the trained spawn range (x
# jitter +-0.08 about 0.46, y jitter +-0.10), so the policy has seen it.
CHECK_CUBE_XY = (0.52, 0.10)


def project(p_base) -> tuple[float, float, float]:
  """A point in the base frame -> (u, v, range) in the 160x120 observation.

  `range` is distance along the optical axis, which is what a depth camera
  reports -- not the euclidean distance to the point.
  """
  rot = np.asarray(CAM_ROT_BASE, float)
  rel = rot.T @ (np.asarray(p_base, float) - np.asarray(CAM_POS_BASE, float))
  fwd = -rel[2]
  if fwd <= 0:
    raise ValueError(f"{p_base} is behind the camera")
  u = CAM_PRINCIPAL_PX[0] + CAM_FOCAL_PX[0] * rel[0] / fwd
  v = CAM_PRINCIPAL_PX[1] - CAM_FOCAL_PX[1] * rel[1] / fwd
  # The same 848x480 -> centre-crop 640x480 -> /4 the observation goes through.
  sw, _ = calib.D435_STREAM_WH
  cw, _ = calib.D435_CROP_WH
  return (u - (sw - cw) / 2) / 4.0, v / 4.0, float(fwd)


def largest_blob(mask: np.ndarray) -> np.ndarray:
  """-> boolean mask of the largest 4-connected component, or all-False."""
  h, w = mask.shape
  seen = np.zeros_like(mask)
  best = np.zeros_like(mask)
  for start in zip(*np.nonzero(mask & ~seen), strict=True):
    if seen[start]:
      continue
    stack, comp = [start], []
    seen[start] = True
    while stack:
      r, c = stack.pop()
      comp.append((r, c))
      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
          seen[rr, cc] = True
          stack.append((rr, cc))
    if len(comp) > int(best.sum()):
      best = np.zeros_like(mask)
      best[tuple(np.array(comp).T)] = True
  return best


def ascii_map(depth_m: np.ndarray, rows: int = 30, cols: int = 80) -> str:
  """The frame as text, because over ssh that beats copying a PNG back.

  Nearer is denser. Invalid pixels are the only thing that renders as blank,
  so a hole in the frame is visible as a hole; the far end of the ramp is '.'
  rather than a space for exactly that reason.
  """
  h, w = depth_m.shape
  fy, fx = h // rows, w // cols
  block = depth_m[: rows * fy, : cols * fx].reshape(rows, fy, cols, fx)
  valid = block > 0
  n = valid.sum(axis=(1, 3))
  s = np.where(valid, block, 0.0).sum(axis=(1, 3))
  d = np.where(n > 0, s / np.maximum(n, 1), 0.0)
  finite = d[d > 0]
  if finite.size == 0:
    return "(no valid pixels)"
  lo, hi = finite.min(), finite.max()
  ramp = "@%#*+=-:,."
  span = max(hi - lo, 1e-6)
  out = []
  for row in d:
    line = "".join(
      " " if x <= 0 else ramp[min(int((x - lo) / span * len(ramp)), len(ramp) - 1)]
      for x in row
    )
    out.append(line)
  return "\n".join(out) + f"\n  near {lo:.3f} m ('@')  far {hi:.3f} m ('.')"


def encode_png(img: np.ndarray) -> bytes:
  """(H,W) uint8 greyscale or (H,W,3) uint8 RGB -> PNG bytes, stdlib only.

  The robot host has numpy, onnxruntime, pyserial and pyrealsense2 and nothing
  else -- no imageio, no PIL -- and adding one for a diagnostic image would be
  a dependency on the machine that most needs to stay minimal.
  """
  img = np.ascontiguousarray(img, dtype=np.uint8)
  h, w = int(img.shape[0]), int(img.shape[1])
  colour = 2 if img.ndim == 3 else 0
  rows = img.reshape(h, -1)
  raw = b"".join(b"\x00" + rows[r].tobytes() for r in range(h))

  def chunk(tag: bytes, data: bytes) -> bytes:
    return (
      struct.pack(">I", len(data))
      + tag
      + data
      + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

  return (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, colour, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw, 6))
    + chunk(b"IEND", b"")
  )


def depth_to_grey(depth_m: np.ndarray, lo: float = 0.0, hi: float = 0.0):
  """Metric depth -> (H,W) uint8, near BRIGHT. Invalid stays 0 (black).

  `lo`/`hi` pin the scale so two frames can be compared by eye; left at 0 they
  are taken from the frame, which is fine alone and misleading side by side.
  """
  good = depth_m > 0
  if hi <= lo:
    v = depth_m[good]
    lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
  out = np.zeros(depth_m.shape, np.uint8)
  out[good] = 255 - np.clip(
    (depth_m[good] - lo) / max(hi - lo, 1e-6) * 254, 0, 254
  ).astype(np.uint8)
  return out


def write_png(path: Path, depth_m: np.ndarray) -> None:
  """8-bit greyscale PNG of a metric depth frame."""
  path.write_bytes(encode_png(depth_to_grey(depth_m)))


def grab(fps: int, frames: int) -> np.ndarray:
  """-> (120,160) metric depth, averaged over valid pixels only. 0 == invalid."""
  from .perception import RealSenseDepth

  cam = RealSenseDepth(fps=fps)
  try:
    acc = np.zeros(calib.DEPTH_HW)
    n = np.zeros(calib.DEPTH_HW)
    got = 0
    while got < frames:
      # The reader thread hands out the latest frame, so a tight loop would
      # average one frame N times. Take one per frame period.
      obs = cam.read().reshape(calib.DEPTH_HW) * calib.DEPTH_CUTOFF_M
      valid = obs > 0
      acc[valid] += obs[valid]
      n[valid] += 1
      got += 1
      _sleep(1.0 / fps)
    return np.where(n > 0, acc / np.maximum(n, 1), 0.0)
  finally:
    cam.close()


def _sleep(s: float) -> None:
  import time

  time.sleep(s)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--camera-fps", type=int, default=90)
  ap.add_argument("--frames", type=int, default=10, help="frames to average")
  ap.add_argument("--save", default=None, help="write the frame to this .npz")
  ap.add_argument(
    "--baseline",
    default=None,
    help="an earlier --save of the EMPTY bench; enables the cube location check",
  )
  ap.add_argument(
    "--cube-xy",
    type=float,
    nargs=2,
    default=CHECK_CUBE_XY,
    metavar=("X", "Y"),
    help=(
      "where the cube actually is, base frame metres. MEASURE it -- this check "
      "is only as good as this number. The default is off-centre on purpose; a "
      "cube near the image centreline cannot distinguish a mirrored camera "
      "from a correct one."
    ),
  )
  ap.add_argument(
    "--surface-z",
    type=float,
    default=WORK_SURFACE_Z_BASE,
    metavar="Z",
    help=(
      f"top of what the cube is RESTING ON, base frame metres. Defaults to "
      f"{WORK_SURFACE_Z_BASE:+.3f}, the foam top, because that is the bench the "
      f"policy was trained against. Pass {BARE_TABLE_Z_BASE:+.3f} for the bare "
      "table with the foam taken off -- the 50 mm between them is 11.9 px at "
      "bench range, which is past this check's own 8 px tolerance, so getting "
      "it wrong reports a camera fault that is not there."
    ),
  )
  ap.add_argument("--png", default=None, help="also write a greyscale PNG here")
  ap.add_argument(
    "--min-blob", type=int, default=6, help="ignore changed regions smaller than this"
  )
  ap.add_argument(
    "--delta", type=float, default=0.02, help="metres nearer than the baseline to count"
  )
  args = ap.parse_args()

  depth = grab(args.camera_fps, args.frames)
  valid = depth > 0
  print(f"frame {depth.shape}, {100 * valid.mean():.1f}% valid")
  if valid.any():
    d = depth[valid]
    print(
      f"  range p05 {np.percentile(d, 5):.3f}  p50 {np.percentile(d, 50):.3f}  "
      f"p95 {np.percentile(d, 95):.3f} m"
    )
  print()
  print(ascii_map(depth))

  if args.save:
    np.savez_compressed(args.save, depth_m=depth)
    print(f"\nsaved {args.save}")
  if args.png:
    write_png(Path(args.png), depth)
    print(f"saved {args.png}")

  cube = (
    float(args.cube_xy[0]),
    float(args.cube_xy[1]),
    float(args.surface_z) + CUBE_HALF_M,
  )
  pu, pv, prange = project(cube)
  surf = float(args.surface_z)
  which = (
    "foam top"
    if abs(surf - WORK_SURFACE_Z_BASE) < 1e-6
    else ("bare table" if abs(surf - BARE_TABLE_Z_BASE) < 1e-6 else "custom")
  )
  print(
    f"\nresting on {surf:+.3f} m ({which}); cube centre {surf + CUBE_HALF_M:+.3f} m"
  )
  print(
    f"model says a cube at {tuple(round(c, 3) for c in cube)} lands at "
    f"({pu:.1f}, {pv:.1f}) px, range {prange:.3f} m"
  )

  if args.baseline is None:
    print(
      "\nNo --baseline, so there is nothing to locate the cube against. Capture "
      "the EMPTY bench with --save, put the cube down, and re-run with "
      "--baseline. The claw and the camera mount are in frame and nearer than "
      "the cube, so 'find the closest thing' does not work here."
    )
    return 0

  base = np.load(args.baseline)["depth_m"]
  if base.shape != depth.shape:
    print(f"!! baseline is {base.shape}, this frame is {depth.shape}")
    return 1
  # Nearer than the baseline by more than the noise, and valid in both.
  changed = (base > 0) & (depth > 0) & ((base - depth) > args.delta)
  blob = largest_blob(changed)
  size = int(blob.sum())
  if size < args.min_blob:
    print(
      f"\n!! only {size} pixels changed by more than {args.delta * 1000:.0f} mm "
      f"(need {args.min_blob}). Either the cube is not there, or it is not "
      "visible, or the baseline was taken with it already in place."
    )
    return 1

  rows, cols = np.nonzero(blob)
  mu, mv = float(cols.mean()), float(rows.mean())
  mrange = float(depth[blob].mean())
  print(f"\nfound a {size}-pixel change at ({mu:.1f}, {mv:.1f}) px, {mrange:.3f} m")

  # Which arrangement does the measurement actually match? The reflections are
  # about the frame centre rather than the principal point, which is half a
  # pixel off and nowhere near enough to confuse a 180-degree error.
  h, w = calib.DEPTH_HW
  candidates = {
    "AS MODELLED": (pu, pv),
    "rotated 180": (w - 1 - pu, h - 1 - pv),
    "mirrored left-right": (w - 1 - pu, pv),
    "mirrored up-down": (pu, h - 1 - pv),
  }
  errs = {k: float(np.hypot(mu - x, mv - y)) for k, (x, y) in candidates.items()}
  best = min(errs, key=lambda k: errs[k])
  print("\n  hypothesis            predicted px    error")
  for k, (x, y) in candidates.items():
    mark = " <-- closest" if k == best else ""
    print(f"  {k:21} ({x:5.1f},{y:5.1f})   {errs[k]:6.1f} px{mark}")

  # +-2.5 deg of pan is the trained camera-rotation DR, which at this range is
  # about 5 px in this frame. Anything inside that is calibration noise; a
  # 30-plus px error is a mounting mistake, not a calibration one.
  tol = 8.0

  # A verdict is only worth as much as this placement's ability to SEPARATE the
  # hypotheses. Near the image centreline a left-right mirror predicts almost
  # the same pixel as the model, so "closest = AS MODELLED" would mean nothing.
  # Say so rather than reporting a confident OK the geometry cannot support.
  seps = {
    k: float(np.hypot(x - pu, y - pv))
    for k, (x, y) in candidates.items()
    if k != "AS MODELLED"
  }
  blind = [k for k, s in seps.items() if s < 2 * tol]
  if blind:
    print(
      f"\n!! this cube position cannot rule out: {', '.join(blind)} -- it "
      f"predicts within {2 * tol:.0f} px of the modelled position, which is "
      "inside the measurement's own error.\n"
      f"   Move the cube to the default {CHECK_CUBE_XY} and re-run; there the "
      "nearest wrong hypothesis is 33 px away."
    )

  print()
  if best == "AS MODELLED" and errs[best] <= tol:
    verdict = "OK" if not blind else "OK as far as this placement can tell"
    print(
      f"{verdict}: within {tol:.0f} px of the model, which is the trained "
      "camera-rotation DR (+-2.5 deg of pan is ~5 px here). The extrinsic is "
      "not the problem."
    )
  elif best == "AS MODELLED":
    dz = mrange - prange
    print(
      f"AIM IS OFF by {errs[best]:.1f} px, but the arrangement is right -- this is "
      "a calibration error, not a mounting one.\n"
      f"  range error {dz * 1000:+.1f} mm along the optical axis. That axis is the "
      "expensive one: 1.4 mm costs nothing and 2.6 mm costs 26 points of success."
    )
  else:
    print(
      f"!! THE CAMERA IS {best.upper()} relative to the model "
      f"({errs[best]:.1f} px against {errs['AS MODELLED']:.1f} px for the "
      "modelled arrangement).\n"
      "   Everything the policy infers about WHERE the cube is is reflected. "
      "Fix the mounting -- do NOT flip the image in perception.py to compensate, "
      "because the depth VALUES would still be those of the real geometry while "
      "the pixel positions became those of a mirrored one."
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
