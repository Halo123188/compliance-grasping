"""Render what the depth DR actually does to the student's observation.

  uv run python scripts/wide_depth_dr_view.py OUT.png [envs=N]

Three rows, and each answers a different question:

  1. WHAT THE STUDENT SEES -- the full observation, several environments at once.
     Every draw is per-episode, so the spread across the row is the spread the
     student trains against, cube size included.
  2. WHERE IT COMES FROM -- one environment, one effect added at a time, so a
     term that is doing nothing shows up as a panel identical to its neighbour.
  3. WHICH EFFECT KILLED WHICH PIXEL -- the same frame with every invalid pixel
     painted by cause, plus the frame the policy is actually acting on once
     camera latency is applied.

Invalid pixels are drawn BLACK and the valid depth ramp starts above black, so
"no return" is never confusable with "very close" -- which is the whole reason
the DR writes exactly 0.0 rather than a small number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide import (  # noqa: E402
  env_cfgs as ec,
)
from mjlab.tasks.manipulation.mdp.observations import (  # noqa: E402
  _global_geom_ids,
  camera_depth,
)
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

TASK = "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr"

# Categorical slots 1-3 of the validated default palette, in fixed order. Three
# and not four because the two blinding targets are ONE mechanism; splitting them
# would put a fourth hue on screen that does not clear the all-pairs floors.
CAUSE_COLORS = {
  "occlusion shadow": "#2a78d6",
  "surface blinding": "#eb6834",
  "speckle dropout": "#1baf7a",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"

# Valid depth rides a single-hue ramp that STARTS ABOVE BLACK, leaving pure black
# to mean "no return" alone.
DEPTH_CMAP = LinearSegmentedColormap.from_list(
  "depth", ["#e8e8e6", "#2e2e2c"]
)  # near -> far, light -> dark
INVALID = "#000000"


def _shot(env: ManagerBasedRlEnv, **kw) -> np.ndarray:
  return camera_depth(env, "d435", cutoff_distance=3.0, **kw)[:, 0].cpu().numpy()


def _rgb(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
  """Depth frame -> RGB, with invalid pixels forced to black."""
  norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
  out = DEPTH_CMAP(norm)[..., :3]
  out[depth == 0.0] = 0.0
  return out


def main() -> None:
  out = Path(sys.argv[1] if len(sys.argv) > 1 else "videos/depth_dr.png")
  n = next(
    (int(a.split("=")[1]) for a in sys.argv[2:] if a.startswith("envs=")),
    6,
  )

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = n
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  env.reset()
  zero = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
  # A few steps so the arm has settled out of its reset transient and the cube
  # has come to rest on the foam at whatever size it was drawn.
  for _ in range(8):
    env.step(zero)

  grip = SceneEntityCfg("robot", geom_names=ec._CLAW_VIS_GEOMS)
  grip.resolve(env.scene)
  cube = SceneEntityCfg("cube", geom_names=("cube",))
  cube.resolve(env.scene)

  noise_kw = dict(range_noise=ec._DEPTH_NOISE)
  shadow_kw = dict(
    shadow_focal_px=ec._DEPTH_SHADOW_FOCAL_PX,
    shadow_baseline=ec._DEPTH_SHADOW_BASELINE,
    shadow_fill=ec._DEPTH_SHADOW_FILL,
    shadow_step=ec._DEPTH_SHADOW_STEP,
  )
  blind_kw = dict(
    blind_gripper_cfg=grip,
    blind_gripper_prob=ec._DEPTH_BLIND_GRIPPER,
    blind_object_cfg=cube,
    blind_object_prob=ec._DEPTH_BLIND_OBJECT,
  )
  drop_kw = dict(dropout_prob=ec._DEPTH_DROPOUT)
  scale_kw = dict(scale_err=ec._DEPTH_SCALE_ERR)
  full_kw = {**noise_kw, **shadow_kw, **blind_kw, **drop_kw, **scale_kw}

  clean = _shot(env)
  full = _shot(env, **full_kw)
  # Attribution: each cause evaluated alone, against the same rendered frame.
  cause_masks = {
    "occlusion shadow": _shot(env, **shadow_kw) == 0.0,
    "surface blinding": _shot(env, **blind_kw) == 0.0,
    "speckle dropout": _shot(env, **drop_kw) == 0.0,
  }

  valid = clean[clean > 0.0]
  vmin, vmax = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))

  # ---- figure ---------------------------------------------------------------
  cols = max(n, 6)
  fig = plt.figure(figsize=(2.05 * cols, 7.6), facecolor=SURFACE)
  gs = fig.add_gridspec(
    3, cols, hspace=0.42, wspace=0.06, top=0.885, bottom=0.105, left=0.02, right=0.98
  )

  def show(ax, img, title, sub=""):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
      s.set_edgecolor("#d7d7d3")
    ax.set_title(title, fontsize=8.5, color=INK, pad=3)
    if sub:
      ax.text(
        0.5,
        -0.06,
        sub,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=INK_2,
      )

  # Row 1 -- the observation, across environments.
  for i in range(n):
    ax = fig.add_subplot(gs[0, i])
    zero_pct = 100.0 * float((full[i] == 0.0).mean())
    show(ax, _rgb(full[i], vmin, vmax), f"env {i}", f"{zero_pct:.1f}% no return")

  # Row 2 -- one environment, one effect at a time.
  ladder = [
    ("rendered", {}),
    ("+ range noise", noise_kw),
    ("+ occlusion shadow", {**noise_kw, **shadow_kw}),
    ("+ surface blinding", {**noise_kw, **shadow_kw, **blind_kw}),
    ("+ speckle dropout", {**noise_kw, **shadow_kw, **blind_kw, **drop_kw}),
    ("+ scale error = full", full_kw),
  ]
  for j, (label, kw) in enumerate(ladder):
    ax = fig.add_subplot(gs[1, j])
    img = _shot(env, **kw)[0]
    show(ax, _rgb(img, vmin, vmax), label, f"{100.0 * (img == 0.0).mean():.1f}% zero")

  # Row 3 -- attribution and latency.
  ax = fig.add_subplot(gs[2, 0:2])
  rgb = _rgb(clean[0], vmin, vmax) * 0.45 + 0.55  # wash out, so causes read
  for name, color in CAUSE_COLORS.items():
    m = cause_masks[name][0]
    rgb[m] = to_rgb(color)
  show(ax, rgb, "which effect killed which pixel", "env 0, each cause alone")
  ax.legend(
    handles=[
      Patch(
        facecolor=c,
        edgecolor="none",
        label=f"{k}  {100 * cause_masks[k][0].mean():.2f}%",
      )
      for k, c in CAUSE_COLORS.items()
    ],
    loc="upper left",
    bbox_to_anchor=(1.03, 1.0),
    frameon=False,
    fontsize=7.6,
    labelcolor=INK,
  )

  # The claw and cube are a few percent of a 160x120 frame, so the blinding is
  # only legible zoomed. Box drawn on the panel above.
  # Bounded by the BLINDING TARGETS themselves, not by a depth threshold: the
  # near strip on the left is the arm's own link and would swallow the box.
  seg = env.scene["d435"].data.segmentation
  assert seg is not None
  ids = torch.cat([_global_geom_ids(env, grip), _global_geom_ids(env, cube)])
  obj = (seg[0, ..., 0].unsqueeze(-1) == ids).any(-1).cpu().numpy()
  ys, xs = np.nonzero(obj)
  y0, y1 = max(ys.min() - 3, 0), min(ys.max() + 4, clean.shape[1])
  x0, x1 = max(xs.min() - 3, 0), min(xs.max() + 4, clean.shape[2])
  ax.add_patch(
    Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, lw=1.0, ls=":")
  )

  ax = fig.add_subplot(gs[2, 3])
  show(
    ax,
    _rgb(full[0][y0:y1, x0:x1], vmin, vmax),
    "zoom: claw + cube",
    "full DR, dotted box at left",
  )

  # LATENCY, and it needs the arm to be MOVING. Under a zero action the scene is
  # static and a lagged frame is bit-identical to the current one -- the panel
  # would look like proof that the delay does nothing.
  lo, hi = ec_lag()
  past = _shot(env)[0]
  move = torch.zeros_like(zero)
  move[:, :7] = 0.6
  for _ in range(hi):
    env.step(move)
  now = _shot(env)[0]
  ax = fig.add_subplot(gs[2, 4])
  show(ax, _rgb(now, vmin, vmax), "the world now", "t, arm driven off the start pose")
  ax = fig.add_subplot(gs[2, 5])
  moved = np.abs(now - past)
  rgb = _rgb(now, vmin, vmax) * 0.45 + 0.55
  rgb[moved > 0.004] = to_rgb(CAUSE_COLORS["occlusion shadow"])
  show(
    ax,
    rgb,
    "what latency costs",
    f"pixels stale at t-{lo}..{hi} ({20 * lo}-{20 * hi} ms): "
    f"{100 * (moved > 0.004).mean():.1f}%",
  )

  fig.suptitle(
    "Depth observation under the full sensor DR  --  160x120, 3.0 m cutoff",
    fontsize=11.5,
    color=INK,
    y=0.965,
  )
  fig.text(
    0.5,
    0.028,
    "Valid depth on a single-hue ramp between the 1st and 99th percentile "
    f"({vmin * 3.0:.2f}-{vmax * 3.0:.2f} m); BLACK is a 0.0 no-return, which is what "
    "a D435 writes and what the policy must not read as a near surface.",
    ha="center",
    fontsize=7.8,
    color=INK_2,
  )
  fig.savefig(out, dpi=170, facecolor=SURFACE)
  print(f"wrote {out}")
  env.close()


def ec_lag() -> tuple[int, int]:
  from mjlab.tasks.manipulation.config.flexiv_two_finger_wide import dr_cfg

  return dr_cfg._CAM_LAG


if __name__ == "__main__":
  main()
