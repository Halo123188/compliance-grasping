"""Solve for the camera's real pose from a multi-pose sweep.

`deploy/capture_sweep.py` records a depth frame at each of several arm poses.
This renders the sim's view of each of those poses with the CURRENT extrinsic,
measures how far the real arm's point cloud sits from the rendered one, and
then asks which fault explains the whole set:

  camera position error      the same displacement at every pose
  camera rotation error      a displacement proportional to range, in the
                             direction the rotation sweeps
  joint1 zero-offset         a displacement about the BASE axis, whose
                             direction rotates as joint1 does

Those three are indistinguishable at a single pose, which is why the home-pose
measurement could only say "the arm is 20 mm to the right". Across poses they
predict different patterns, so a linear least-squares over the per-pose
displacements separates them -- and reports how well, which matters more than
the numbers: a sweep that did not move joint1 enough will still return an
answer, and the condition number is what says whether to believe it.

  uv run python scripts/fit_camera_pose.py sweep.npz

The output is the correction to apply to CAM_POS_BASE / CAM_ROT_BASE in
deploy/check_camera.py and to the camera in the sim scene. Nothing is written:
an extrinsic that changes under your feet between a diagnosis and a run is
worse than a wrong one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (  # noqa: E402
  ARM_BASE_Z,
  twofinger_arm_spec,
)
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (  # noqa: E402
  get_table_spec,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import calib, kinematics  # noqa: E402
from deploy.check_camera import CAM_POS_BASE, CAM_ROT_BASE  # noqa: E402
from sim_depth import RENDER_WH, render_observation  # noqa: E402

# The 160x120 observation's intrinsics: the 848x480 profile's focal length,
# centre-cropped to 640x480 and downsampled by 4.
FX = FY = 428.3054 / 4.0
CX = (424.4773 - (848 - 640) / 2) / 4.0
CY = 243.6254 / 4.0
# A point has to stand this far out of the table to count as "the arm". Below
# it the table itself dominates, and the table is a plane -- it constrains
# nothing this script is solving for.
ARM_MIN_H = 0.02
# How far from the FK-predicted claw centre a point may be and still count as
# claw. The claw itself is about 120 mm across, and the error being measured is
# tens of millimetres, so this has to be comfortably larger than both -- its job
# is to exclude the room, not to pick a side of the claw.
CLAW_RADIUS = 0.16
# Fewest claw pixels a pose may contribute. The claw is about 120 x 90 mm, so at
# the far end of a sweep (0.48 m, 107 px focal) its silhouette is only ~200 px --
# small, but the ICP residual per pose is printed, so a thin pose announces
# itself rather than quietly dominating.
MIN_POINTS = 120


def build():
  """The bench as it stands for a calibration sweep: arm and table, no cube."""
  spec = twofinger_arm_spec()
  for b in spec.worldbody.bodies:
    if b.name == "base":
      b.pos = [0.0, 0.0, ARM_BASE_Z]
  spec.attach(get_table_spec(), prefix="", frame=spec.worldbody.add_frame())
  m = spec.compile()
  d = mujoco.MjData(m)
  jadr = [
    m.jnt_qposadr[
      mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, n if n.startswith("joint") else f"{n}_tf"
      )
    ]
    for n in calib.JOINT_NAMES
  ]
  return m, d, jadr


def cloud(z: np.ndarray) -> np.ndarray:
  h, w = z.shape
  vv, uu = np.mgrid[0:h, 0:w]
  return np.stack([(uu - CX) * z / FX, (vv - CY) * z / FY, z], -1)


def fit_plane(z: np.ndarray) -> tuple[np.ndarray, float]:
  """The table, in camera coordinates. Normal points back at the camera."""
  P, good = cloud(z), z > 0
  h = z.shape[0]
  m = good.copy()
  m[: h // 3] = False
  m &= z < 0.9
  pts = P[m]
  n = np.array([0.0, -1.0, 0.0])
  d = 0.0
  for _ in range(8):
    c = pts.mean(0)
    _, _, V = np.linalg.svd(pts - c, full_matrices=False)
    n = V[-1] * (-1 if V[-1][2] > 0 else 1)
    d = float(-n @ c)
    m2 = m & good & (np.abs(P @ n + d) < 0.006)
    if m2.sum() < 500:
      break
    pts, m = P[m2], m2
  return n, d


def arm_cloud(
  z: np.ndarray, n: np.ndarray, d: float, centre: np.ndarray, radius: float
) -> np.ndarray:
  """Points that are the CLAW: above the table and near where FK says it is.

  "Above the table plane" alone is not a claw detector. At the home pose it
  happens to work, because everything else above the plane is past the 3 m
  cutoff -- but raise the claw and the far wall climbs into the top of the
  frame and joins the cloud. That is what wrecked the first run of this fit:
  the real clouds carried 1.8x the points of the rendered ones and ICP happily
  slid the claw 130 mm sideways to sit on a wall the render does not contain.

  So the claw is gated on the FORWARD KINEMATICS, which are known to 1.4 mm
  against RDK's own tool pose. The gate is the same for both frames, and it is
  wide enough (`radius`) that it cannot manufacture the answer: it excludes the
  room, not one side of the claw.
  """
  P = cloud(z)
  m = (z > 0) & (P @ n + d > ARM_MIN_H) & (z < 0.9)
  m &= ((P - centre) ** 2).sum(-1) < radius**2
  return P[m]


def silhouette_width(P: np.ndarray) -> float:
  """How wide the claw's silhouette is, in metres across the image.

  This is the sanity check that has to be read BEFORE the displacement fit,
  because if the two silhouettes are not the same object then "how far apart are
  they" is not a well posed question. Measured on the first sweep: the real claw
  came back 25-30% wider than the rendered one at every single pose, which is
  either the depth sensor fattening the foreground at a depth edge (a standard
  D435 artifact) or the modelled pads being thinner than the real fingers. Both
  bias an edge- or centroid-based displacement, and by an amount that depends on
  where in the frame the claw is -- which is exactly the spurious position
  dependence the first run of this script reported as a camera error.

  3rd and 97th percentile rather than min and max: one flying pixel should not
  define an edge.
  """
  return float(np.percentile(P[:, 0], 97) - np.percentile(P[:, 0], 3))


def trimmed_nn(X: np.ndarray, Y: np.ndarray, keep: float = 0.85) -> np.ndarray:
  """Nearest-neighbour distances from X into Y, worst (1-keep) dropped.

  The real cloud has holes the render does not, so a fraction of its points
  have no counterpart at all and would otherwise pull the fit.
  """
  out = np.empty(len(X))
  for i in range(0, len(X), 512):
    c = X[i : i + 512]
    out[i : i + 512] = np.sqrt(((c[:, None] - Y[None]) ** 2).sum(-1).min(1))
  out.sort()
  return out[: int(keep * len(out))]


def fit_translation(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
  """Translation taking the real cloud A onto the sim cloud B, by trimmed ICP."""
  t = np.zeros(3)
  for _ in range(14):
    X = A + t
    idx = np.empty(len(X), int)
    for i in range(0, len(X), 512):
      c = X[i : i + 512]
      idx[i : i + 512] = ((c[:, None] - B[None]) ** 2).sum(-1).argmin(1)
    Y = B[idx]
    r = np.linalg.norm(Y - X, axis=1)
    keep = r <= np.quantile(r, 0.85)
    t = t + (Y[keep] - X[keep]).mean(0)
  return t, float(trimmed_nn(A + t, B).mean())


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("sweep", help="npz from deploy/capture_sweep.py")
  args = ap.parse_args()

  z = np.load(args.sweep)
  names = [str(s) for s in z["names"]]
  Q = z["joint_pos"]
  hand = z["hand_pos"]
  real = z["depth_m"]

  m, data, jadr = build()
  w, h = RENDER_WH
  renderer = mujoco.Renderer(m, height=h, width=w)
  renderer.enable_depth_rendering()

  # The camera's own axes in the base frame. MuJoCo cameras look down -z, so
  # the optical axis is -column 2, and image-right/up are columns 0 and 1.
  R = np.asarray(CAM_ROT_BASE, float)
  cam_right, cam_up, cam_back = R[:, 0], R[:, 1], R[:, 2]
  cam_pos = np.asarray(CAM_POS_BASE, float)
  # Rows map a base-frame vector onto the axes `cloud()` uses: x image-right,
  # y image-DOWN, z along the optical axis. The image-down flip is the whole
  # reason this is spelled out rather than reusing CAM_ROT_BASE directly --
  # MuJoCo's camera y is image-UP, and getting that backwards silently mirrors
  # every pitch estimate.
  Rcam = np.stack([cam_right, -cam_up, -cam_back])

  print(
    f"{'pose':>13}{'real pts':>10}{'sim pts':>9}{'dx':>8}{'dy':>8}{'dz':>8}"
    f"{'rms':>7}{'width sim':>11}{'real':>7}   (mm, camera axes)"
  )
  info = []
  widths: list[tuple[float, float]] = []
  for i, name in enumerate(names):
    data.qpos[jadr] = np.concatenate([Q[i], hand])
    mujoco.mj_forward(m, data)
    sim = render_observation(renderer, data)
    n, d = fit_plane(sim)
    fk = kinematics.forward(np.concatenate([Q[i], hand]))
    claw_base = np.mean([fk["left_pad"], fk["right_pad"], fk["hand_base"]], axis=0)
    centre = Rcam @ (claw_base - cam_pos)
    A = arm_cloud(real[i], n, d, centre, CLAW_RADIUS)
    B = arm_cloud(sim, n, d, centre, CLAW_RADIUS)
    if len(A) < MIN_POINTS or len(B) < MIN_POINTS:
      print(f"{name:>13}{len(A):10d}{len(B):9d}   too few arm points; skipped")
      continue
    t, rms = fit_translation(A, B)
    ws, wr = silhouette_width(B), silhouette_width(A)
    widths.append((ws, wr))
    print(
      f"{name:>13}{len(A):10d}{len(B):9d}"
      f"{t[0] * 1000:8.1f}{t[1] * 1000:8.1f}{t[2] * 1000:8.1f}{rms * 1000:7.1f}"
      f"{ws * 1000:11.0f}{wr * 1000:7.0f}"
    )
    info.append((name, A.mean(0), t, rms))

  renderer.close()
  if len(info) < 4:
    print("\n!! fewer than four usable poses; cannot separate the hypotheses")
    return 1

  # The model. `t` is the translation taking the REAL cloud onto the SIM cloud
  # in camera axes, t = p_sim - p_real, and each unknown is a TRUE-MINUS-NOMINAL
  # correction:
  #
  #   camera at cam_pos + dp  ->  p_real = p_sim - dp     ->  t = +dp
  #   camera rotated by w     ->  p_real = p_sim - w x p  ->  t = +w x p
  #
  # A JOINT1 OFFSET IS DELIBERATELY NOT IN HERE, and no sweep of arm poses can
  # put it there. A joint1 error dj moves a point at p_base by dj*(z x p_base).
  # Substitute p_base = Rcam^T p_cam + cam_pos and it splits exactly into
  #
  #     dj*(z x Rcam^T p_cam)   -- a camera rotation of dj about the base's z
  #   + dj*(z x cam_pos)        -- a camera translation
  #
  # with no remainder, for every pose. So joint1 is an exact linear combination
  # of the six camera columns: the design matrix is singular by construction,
  # not for want of data, and the first version of this script duly returned a
  # condition number of 2.6e16 and a nonsense -8.7 deg. The reason is physical
  # rather than numerical -- everything joint1 moves is the same rigid body, and
  # a rigid body cannot tell you what it is rigid WITH RESPECT TO. Separating
  # them needs a feature that does NOT turn with joint1: a cube at a measured
  # place on the table (deploy/check_camera.py), or the base structure itself.
  #
  # What is fitted here is therefore the EFFECTIVE camera pose -- the one that
  # makes the render match the arm. Correcting the camera by it is right for the
  # arm either way; it is right for the table and the cube only if the camera
  # really is the cause. The equivalence is printed below so the choice is
  # visible instead of buried.
  A_rows, b_rows = [], []
  for _name, centroid, t, _rms in info:
    for k in range(3):
      row = np.zeros(6)
      row[k] = 1.0
      for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        row[3 + j] = np.cross(e, centroid)[k]
      A_rows.append(row)
      b_rows.append(t[k])
  M = np.asarray(A_rows)
  y = np.asarray(b_rows)
  sol, *_ = np.linalg.lstsq(M, y, rcond=None)
  resid = M @ sol - y
  _, sv, _ = np.linalg.svd(M, full_matrices=False)

  rms = lambda r: float(np.sqrt((r**2).mean())) * 1000  # noqa: E731

  dp, dw = sol[:3], sol[3:6]
  dp_base = Rcam.T @ dp
  print("\nEFFECTIVE CAMERA POSE over all poses")
  print(
    f"  position   right {dp[0] * 1000:+7.2f}  down {dp[1] * 1000:+7.2f}"
    f"  along axis {dp[2] * 1000:+7.2f}  mm"
  )
  print(
    f"  rotation   pitch {np.degrees(dw[0]):+6.2f}  yaw {np.degrees(dw[1]):+6.2f}"
    f"  roll {np.degrees(dw[2]):+6.2f}  deg  (about right / down / optical axis)"
  )
  print(f"  residual   rms {rms(resid):.2f} mm over {len(y)} equations")
  print(f"  condition  {sv[0] / sv[-1]:.1f}  (over ~1e3 and the poses were too alike)")
  print(
    "  CAM_POS_BASE would become "
    f"({cam_pos[0] + dp_base[0]:.6f}, {cam_pos[1] + dp_base[1]:.6f}, "
    f"{cam_pos[2] + dp_base[2]:.6f})"
  )

  # The joint1 equivalence, as numbers rather than as a claim.
  jz = Rcam @ np.array([0.0, 0.0, 1.0])
  j_shift = Rcam @ np.cross(np.array([0.0, 0.0, 1.0]), cam_pos)
  scale = np.radians(1.0)
  print("\nWHAT ONE DEGREE OF JOINT1 LOOKS LIKE FROM HERE (the degenerate direction)")
  print(
    f"  == camera yaw {np.degrees(scale * jz[1]):+.3f} deg about image-down"
    f", pitch {np.degrees(scale * jz[0]):+.3f}, roll {np.degrees(scale * jz[2]):+.3f}"
  )
  print(
    f"  +  camera shift right {scale * j_shift[0] * 1000:+.3f} mm"
    f", down {scale * j_shift[1] * 1000:+.3f}"
    f", along axis {scale * j_shift[2] * 1000:+.3f}"
  )
  print("  Any multiple of this can be moved between the two without changing the")
  print("  fit. To pin it down, measure a CUBE at a known place on the table --")
  print("  it does not turn with joint1. deploy/check_camera.py does exactly that.")

  ws = np.array([w[0] for w in widths])
  wr = np.array([w[1] for w in widths])
  excess = float(np.median(wr / ws) - 1.0)
  print("\nSILHOUETTE WIDTH, real vs rendered")
  print(
    f"  real is {100 * excess:+.0f}% wider (median over {len(ws)} poses, "
    f"spread {100 * (wr / ws).std():.0f} pt)"
  )
  if abs(excess) > 0.08:
    print("  !! READ THIS BEFORE THE NUMBERS ABOVE. The two silhouettes are not")
    print("     the same shape, so the displacement between them is not a pose")
    print("     error -- an asymmetric width excess moves a centroid by an amount")
    print("     that depends on where in the frame the claw sits, and that shows")
    print("     up in the fit as a camera error that is not there. Fix the width")
    print("     (sensor foreground-fattening, or the modelled pad thickness)")
    print("     before believing any extrinsic correction printed above.")

  print("\nWHAT EACH PIECE BUYS, alone")
  for label, cols in (
    ("position only", [0, 1, 2]),
    ("rotation only", [3, 4, 5]),
    ("both", [0, 1, 2, 3, 4, 5]),
  ):
    s, *_ = np.linalg.lstsq(M[:, cols], y, rcond=None)
    print(f"  {label:<28} residual rms {rms(M[:, cols] @ s - y):6.2f} mm")
  print(f"  {'nothing (current extrinsic)':<28} residual rms {rms(y):6.2f} mm")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
