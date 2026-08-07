"""Dump exactly what the student sees: the camera observation TENSOR, not a render.

  uv run python scripts/diag_student_view.py [TASK] [OUT.png]

`scripts/render_scene.py` renders the D435 through MuJoCo's offscreen renderer
at full resolution. That is not the student's input. The student gets the
observation manager's output: 128x72, depth clipped to 3 m and divided by it,
or RGB divided by 255. A cube that is obvious in the render can be four pixels
and half a millimetre of depth contrast in the tensor, which is the difference
between a distillation that works and one that plateaus.

Prints the cube's on-image size and its depth contrast against the table, since
those are the two numbers that decide whether yaw is recoverable at all.
"""

import sys

import imageio.v2 as imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = (
  sys.argv[1] if len(sys.argv) > 1 else "Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "videos/student_view.png"
N, DEV = 8, "cuda:0"

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)

torch.manual_seed(0)
obs, _ = env.reset()
# Step a little so the arm is at its hover pose rather than mid-reset.
for _ in range(5):
  obs = env.step(torch.zeros(N, env.action_manager.total_action_dim, device=DEV))[0]

img = obs["camera"].float().cpu().numpy()  # (N, C, H, W)
print(f"task={TASK}")
print(f"camera obs: shape={img.shape} dtype={obs['camera'].dtype}")
print(f"            min={img.min():.4f} max={img.max():.4f} mean={img.mean():.4f}")
print(f"student obs dim: {obs['student'].shape[-1]}")
print(f"teacher obs dim: {obs['actor'].shape[-1]}")

# Depth is the last channel for rgbd, the only one for depth-only.
depth = img[:, -1] if img.shape[1] in (1, 4) else None
if depth is not None:
  # Isolate the cube WITHOUT camera intrinsics: at reset every env holds the
  # same arm pose over the same table, and only the cube's pose is randomized.
  # So the per-pixel median across the batch is the background, and each env's
  # deviation from it is that env's cube. Thresholding against the frame's own
  # median instead just re-detects the arm, which fills half the image.
  bg = np.median(depth, axis=0)
  print("\ncube footprint in the student's actual input (background subtracted):")
  for i in range(N):
    resid = bg - depth[i]  # positive where the cube sits in front of the table
    mask = resid > 0.002  # 6 mm at the 3 m cutoff
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
      print(f"  env {i}: NOT VISIBLE")
      continue
    print(
      f"  env {i}: {mask.sum():3d} px, bbox {xs.max() - xs.min() + 1}x"
      f"{ys.max() - ys.min() + 1}, peak contrast {resid.max():.4f} "
      f"({resid.max() * 3.0 * 1000:.0f} mm), centre ({xs.mean():.0f}, {ys.mean():.0f})"
    )
  seen = sum(float((bg - depth[i]).max()) > 0.002 for i in range(N))
  print(f"  visible in {seen}/{N} envs")

# Tile the batch into one contrast-stretched strip per channel, then a second
# strip of the background-subtracted depth so the cube is actually legible.
tiles = []
for c in range(img.shape[1]):
  row = np.concatenate([img[i, c] for i in range(N)], axis=1)
  lo, hi = row.min(), row.max()
  tiles.append((row - lo) / max(hi - lo, 1e-6))
if depth is not None:
  resid = np.clip(np.median(depth, axis=0)[None] - depth, 0, None)
  row = np.concatenate([resid[i] for i in range(N)], axis=1)
  tiles.append(row / max(row.max(), 1e-6))
strip = (np.concatenate(tiles, axis=0) * 255).astype(np.uint8)
imageio.imwrite(OUT, strip)
print(f"wrote {OUT}  ({strip.shape[1]}x{strip.shape[0]}, per-channel contrast stretch)")
env.close()
