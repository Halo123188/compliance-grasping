"""Turn a raw metric depth frame into something a human can read.

Shared by `render_student.py` (the depth panel beside the rollout) and
`wide_scene_shot.py` (the still of the training scene), so both show the policy's
input under exactly the same mapping.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps

_TURBO = colormaps["turbo"]


def colorize_depth(depth_m: np.ndarray, span: float = 0.30) -> np.ndarray:
  """Depth -> RGB coloured by height above the fitted table plane. HxWx3 in [0,1].

  No linear map of ABSOLUTE depth can show this scene. Two failed attempts,
  both measured on a rendered frame:

    min/max stretch          median panel pixel 0, mean 6.6/255 -- near-black.
                             The fingers sit centimetres from the lens and the
                             background is clamped at the 3 m cutoff, so those
                             two own the entire range.
    2-98 percentile stretch  mean 59/255 but the table saturates to one flat
                             red. The table plane recedes across ~110 mm of
                             depth while the cube stands only 50 mm proud of
                             it, so any window wide enough to hold the table is
                             six times too wide to resolve the cube.

  The table is a plane, so fit it and colour the height above it. That is the
  same data the network receives, re-referenced for display -- not extra
  information -- and it puts the cube and fingers on a scale where they show up.

  Fit in INVERSE depth, not depth. Under a pinhole camera a planar surface is
  linear in (u, v) only in 1/z; fitting z directly leaves a curved residual that
  saturates half the panel. ``sensor.data.depth`` is raw metres -- the /3.0
  normalisation belongs to the observation term, not here.

  The fit is least squares run three times with 2-sigma rejection, so the cube
  and the gripper (exactly what we want to see) cannot drag the plane toward
  themselves.
  """
  h, w = depth_m.shape
  valid = (depth_m > 0.05) & (depth_m < 2.0) & np.isfinite(depth_m)  # drop background
  disp = 1.0 / np.clip(depth_m, 0.05, None)

  vv, uu = np.mgrid[0:h, 0:w]
  A = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], axis=1)
  dd = disp.ravel()
  keep = valid.ravel().copy()
  coef = np.zeros(3)
  for _ in range(3):
    if keep.sum() < 16:
      break
    coef, *_ = np.linalg.lstsq(A[keep], dd[keep], rcond=None)
    resid = dd - A @ coef
    keep = valid.ravel() & (np.abs(resid) < 2.0 * resid[keep].std())

  z_plane = 1.0 / np.clip((A @ coef).reshape(h, w), 1e-3, None)
  height = z_plane - depth_m  # metres along the ray; positive = above the table

  rgb = _TURBO(np.clip(height / span, 0.0, 1.0))[..., :3]
  rgb[height < 0.004] = 0.10  # the table itself reads as flat dark
  rgb[~valid] = 0.03  # background
  return rgb
