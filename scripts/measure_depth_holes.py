"""Measure the D435's holes well enough to model them, from a pose sweep.

A hole is a pixel where the renderer has solid geometry and the sensor returns
nothing. On this bench they are 7% of the frame and they are the largest
remaining sim-to-real gap: patching only the hole pixels of a recorded run took
the policy's command from 0.205 rad away from its sim-frame behaviour to 0.085,
while fixing the geometry everywhere else did nothing (0.202).

They are not random. A D435 is an active stereo camera and a pixel comes back
empty for one of three reasons, each with its own geometry:

  A  SURFACE BLANKING     a specular or IR-dark surface returns nothing, so a
                          whole face of an object is blank
  B  OCCLUSION SHADOW     the projector and the two imagers are separated by a
                          baseline, so a foreground object hides a STRIP of the
                          background beside it -- always on the same side, with
                          width f*B*(1/z_fg - 1/z_bg)
  C  EDGE FATTENING       the foreground bleeds a pixel or two over the
                          background

Random 8x8 patch dropout, which is what the training env models today, is none
of these: it is uncorrelated with geometry, it moves every frame, and a median
filter removes it. This script measures each mechanism separately so the
randomization can be built from numbers rather than from a guess.

  uv run python scripts/measure_depth_holes.py sweep.npz

Needs a sweep from `deploy/capture_sweep.py` and renders each pose itself, so
the sim and real frames being compared are the same scene at the same joints.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy import kinematics  # noqa: E402
from deploy.check_camera import CAM_POS_BASE, CAM_ROT_BASE  # noqa: E402
from fit_camera_pose import CLAW_RADIUS, build, cloud, fit_plane  # noqa: E402
from sim_depth import RENDER_WH, render_observation  # noqa: E402

FX = 428.3054 / 4.0  # focal length of the 160x120 observation, in pixels
D435_BASELINE_M = 0.050  # nominal IR stereo baseline
BAND_PX = 14  # how far out from a silhouette to look for a shadow
EDGE_STEP_M = 0.010  # depth step that counts as a silhouette edge


def masks(sim, real, q, hand, Rcam, cam_pos):
  """-> (claw silhouette in the render, holes, real-invalid)."""
  n, d = fit_plane(sim)
  fk = kinematics.forward(np.concatenate([q, hand]))
  claw = np.mean([fk["left_pad"], fk["right_pad"], fk["hand_base"]], axis=0)
  c = Rcam @ (claw - cam_pos)
  P = cloud(sim)
  fg = (sim > 0) & (P @ n + d > 0.02) & (((P - c) ** 2).sum(-1) < CLAW_RADIUS**2)
  return fg, (sim > 0) & (real <= 0), real <= 0


def shift(mask, du):
  out = np.zeros_like(mask)
  if du > 0:
    out[:, du:] = mask[:, :-du]
  elif du < 0:
    out[:, :du] = mask[:, -du:]
  else:
    out = mask.copy()
  return out


def median3(z):
  """3x3 median of a depth frame, zeros included as values."""
  h, w = z.shape
  pad = np.pad(z, 1, mode="edge")
  stack = np.stack(
    [pad[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=0
  )
  return np.median(stack, axis=0)


def components(mask):
  """Sizes of the 4-connected components of a boolean mask."""
  seen = np.zeros_like(mask)
  sizes = []
  for start in zip(*np.nonzero(mask), strict=True):
    if seen[start]:
      continue
    stack, n = [start], 0
    seen[start] = True
    while stack:
      r, c = stack.pop()
      n += 1
      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
          if mask[rr, cc] and not seen[rr, cc]:
            seen[rr, cc] = True
            stack.append((rr, cc))
    sizes.append(n)
  return np.array(sizes)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("sweep", help="npz from deploy/capture_sweep.py")
  ap.add_argument(
    "--cam-y",
    type=float,
    default=None,
    help=(
      "override the camera's base-frame y before rendering. The shadow WIDTH "
      "is measured in pixels beside a silhouette, so a registration error "
      "between the two frames biases it directly; scripts/fit_camera_pose.py "
      "measured +0.012943 against the nominal -0.005707."
    ),
  )
  args = ap.parse_args()

  z = np.load(args.sweep)
  names = [str(s) for s in z["names"]]
  Q, hand, real = z["joint_pos"], z["hand_pos"], z["depth_m"]

  R = np.asarray(CAM_ROT_BASE, float)
  Rcam = np.stack([R[:, 0], -R[:, 1], -R[:, 2]])
  cam_pos = np.asarray(CAM_POS_BASE, float)

  m, data, jadr = build()
  if args.cam_y is not None:
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
    # camera_mount's frame is the base rotated +90 deg about z, so its local x
    # is the base's y.
    m.cam_pos[cid][0] += args.cam_y - CAM_POS_BASE[1]
    print(f"camera base-frame y {CAM_POS_BASE[1]:+.6f} -> {args.cam_y:+.6f}\n")
  r = mujoco.Renderer(m, height=RENDER_WH[1], width=RENDER_WH[0])
  r.enable_depth_rendering()
  sims = []
  for i in range(len(names)):
    data.qpos[jadr] = np.concatenate([Q[i], hand])
    mujoco.mj_forward(m, data)
    sims.append(render_observation(r, data))
  r.close()

  H, W = real[0].shape
  print(f"{len(names)} poses, {H}x{W}\n")

  # --- 1. how much, and where ------------------------------------------- #
  print("1. HOW MUCH IS MISSING")
  print(f"{'pose':>13}{'invalid':>10}{'holes':>9}{'top 1/3':>10}{'claw blank':>12}")
  tot, top, claw_blank = [], [], []
  for i, nm in enumerate(names):
    fg, hole, inval = masks(sims[i], real[i], Q[i], hand, Rcam, cam_pos)
    t3 = hole[: H // 3]
    cb = (hole & fg).sum() / max(fg.sum(), 1)
    tot.append(inval.mean())
    top.append(t3.mean())
    claw_blank.append(cb)
    print(
      f"{nm:>13}{100 * inval.mean():9.2f}%{100 * hole.mean():8.2f}%"
      f"{100 * t3.mean():9.2f}%{100 * cb:11.1f}%"
    )
  print(
    f"{'MEDIAN':>13}{100 * np.median(tot):9.2f}%{'':9}"
    f"{100 * np.median(top):9.2f}%{100 * np.median(claw_blank):11.1f}%"
  )

  # --- 2. which side is the shadow on? ----------------------------------- #
  # Bands just outside the claw silhouette, left and right. The occlusion
  # shadow is on ONE side -- the baseline direction -- so an asymmetry here is
  # the measurement, and its sign is the parameter the DR needs.
  print("\n2. SHADOW SIDE  (hole density in a band beside the claw)")
  print(f"{'offset px':>11}{'LEFT':>10}{'RIGHT':>10}{'ratio R/L':>12}")
  for off in (1, 2, 3, 4, 6, 8, 11, 14):
    lo, ro, la, ra = 0, 0, 0, 0
    for i in range(len(names)):
      fg, hole, _ = masks(sims[i], real[i], Q[i], hand, Rcam, cam_pos)
      solid = sims[i] > 0
      right = shift(fg, off) & ~fg & solid
      left = shift(fg, -off) & ~fg & solid
      ro += (hole & right).sum()
      ra += right.sum()
      lo += (hole & left).sum()
      la += left.sum()
    dl = lo / max(la, 1)
    dr = ro / max(ra, 1)
    print(f"{off:11d}{100 * dl:9.1f}%{100 * dr:9.1f}%{dr / max(dl, 1e-9):12.2f}")

  # --- 3. is the band width what the baseline predicts? ------------------ #
  print("\n3. SHADOW WIDTH  vs  f*B*(1/z_fg - 1/z_bg)")
  widths, preds = [], []
  for i in range(len(names)):
    fg, hole, _ = masks(sims[i], real[i], Q[i], hand, Rcam, cam_pos)
    sim = sims[i]
    for row in range(H):
      cols = np.flatnonzero(fg[row])
      if cols.size < 3:
        continue
      for edge, direction in ((cols.max(), +1), (cols.min(), -1)):
        u = edge + direction
        run = 0
        while 0 <= u < W and hole[row, u]:
          run += 1
          u += direction
        if not (0 <= u < W) or sim[row, u] <= 0 or run == 0:
          continue
        z_fg, z_bg = sim[row, edge], sim[row, u]
        if z_bg - z_fg < EDGE_STEP_M:
          continue
        widths.append((direction, run))
        preds.append(FX * D435_BASELINE_M * (1 / z_fg - 1 / z_bg))
  widths = np.array(widths)
  preds = np.array(preds)
  for direction, label in ((+1, "right of claw"), (-1, "left of claw")):
    sel = widths[:, 0] == direction
    if sel.sum() < 5:
      print(f"  {label:>14}: {sel.sum()} samples, too few")
      continue
    obs = widths[sel, 1]
    print(
      f"  {label:>14}: n={sel.sum():4d}  observed {np.median(obs):5.1f} px"
      f"   predicted {np.median(preds[sel]):5.1f} px"
      f"   implied baseline {1000 * D435_BASELINE_M * np.median(obs) / max(np.median(preds[sel]), 1e-9):5.1f} mm"
    )

  # --- 4. does a median filter remove them? ------------------------------ #
  print("\n4. SURVIVES A 3x3 MEDIAN?  (i.i.d. dropout would not)")
  before, after = [], []
  for i in range(len(names)):
    b = (real[i] <= 0).mean()
    a = (median3(real[i]) <= 0).mean()
    before.append(b)
    after.append(a)
  b, a = np.median(before), np.median(after)
  print(
    f"  invalid {100 * b:.2f}% -> {100 * a:.2f}% after filtering "
    f"({100 * (1 - a / b):.0f}% removed)"
  )

  # --- 5. how blotchy? --------------------------------------------------- #
  print("\n5. BLOB SIZE  (sets the correlation length of the random field)")
  allsz = np.concatenate([components(real[i] <= 0) for i in range(len(names))])
  print(f"  {len(allsz)} components over {len(names)} frames")
  print(
    f"  size p50 {np.median(allsz):.0f} px   p90 {np.percentile(allsz, 90):.0f}"
    f"   max {allsz.max()}   mean sqrt(area) {np.sqrt(allsz.mean()):.1f} px"
  )
  print(
    f"  fraction of invalid pixels in blobs >= 16 px: "
    f"{100 * allsz[allsz >= 16].sum() / allsz.sum():.0f}%"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
