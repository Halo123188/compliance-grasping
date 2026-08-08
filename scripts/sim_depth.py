"""Render the scene camera the way the STUDENT was trained to see it.

`mujoco.Renderer` and `mujoco_warp` disagree about a camera with calibrated
intrinsics, and the difference is a 1.325x horizontal magnification -- big
enough to have been mistaken, for most of a day, for a camera extrinsic error.

mujoco-warp crops. `mujoco_warp/_src/render_util.py` compares the render's
aspect with the sensor's and shrinks whichever sensor dimension is too large:
the D435's sensor is 4.77 x 2.70 mm (16:9) and the student renders 160x120
(4:3), so the sensor WIDTH is cut 4.77 -> 3.60 mm. That is exactly a 640-column
centre-crop of the 848-wide stream (3.60/4.77 x 848 = 640.0000), and it is why
`deploy/perception.py` centre-crops the real D435 to 640x480 before resizing --
so the student sees the 73.53 deg it was trained on.

`mujoco.Renderer` does NOT crop. It fits the FULL 89.42 deg sensor field into
whatever viewport it is given, so rendering 640x480 and calling it "the crop"
squashes 89.42 deg into pixels that mean 73.53. Measured directly, by putting
the cube at five known bench positions and comparing the column it lands on
with the column the calibrated projection predicts:

    cube y     project u    render u
     -0.18        125.47      114.35
     -0.09        102.00       96.61
      0.00         78.51       78.16
      0.09         55.01       60.12
      0.18         31.49       42.38
                render_u = 0.768 * project_u + 18.0

0.768, against the 0.755 that 640/848 predicts and the 1.000 a true crop would
give. (The 1.7% is the cube's visible face turning with the viewing angle; the
claw silhouette, which has no such bias, measured 1.30.)

So: render at the sensor's OWN resolution and aspect, where "no crop" and "the
right crop" coincide, then apply the same centre-crop and box filter that
`deploy/perception.py` applies to the real stream. Both paths then describe the
same 73.53 x 58.53 deg.
"""

from __future__ import annotations

import numpy as np
from deploy import calib

# The modelled sensor, not the crop. Rendering at this size is the point: the
# viewport aspect equals the sensor aspect, so there is nothing for either
# renderer's crop branch to do and the two agree.
RENDER_WH = calib.D435_STREAM_WH  # (848, 480)


def to_observation(depth_m: np.ndarray) -> np.ndarray:
  """(480,848) rendered depth -> the (120,160) the student sees, in metres.

  Deliberately the same three steps as `deploy.perception.centre_crop_resize`,
  in the same order: crop to the trained field, then box-filter by the exact
  integer factor. Anything past the cutoff becomes 0, matching the sensor's
  "no return" rather than a very distant reading.
  """
  h, w = depth_m.shape
  if (w, h) != RENDER_WH:
    raise ValueError(
      f"expected a {RENDER_WH} render, got {(w, h)}. The whole point of this "
      "module is that the render size must equal the modelled sensor size; "
      "see the note above before changing it."
    )
  cw, ch = calib.D435_CROP_WH
  x0, y0 = (w - cw) // 2, (h - ch) // 2
  crop = depth_m[y0 : y0 + ch, x0 : x0 + cw]
  th, tw = calib.DEPTH_HW
  fy, fx = ch // th, cw // tw
  small = crop.reshape(th, fy, tw, fx).mean(axis=(1, 3))
  small = np.where(small >= calib.DEPTH_CUTOFF_M, 0.0, small)
  return small


def render_observation(renderer, data, camera: str = "scene_cam") -> np.ndarray:
  """One depth observation from an already-configured depth renderer."""
  renderer.update_scene(data, camera=camera)
  return to_observation(np.asarray(renderer.render(), dtype=np.float64))
