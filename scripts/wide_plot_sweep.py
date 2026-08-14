"""Draw what `wide_eval_sweep.py` measured: success vs object size, and vs place.

  uv run python scripts/wide_plot_sweep.py viz/sweep

Reads one .npz per policy and writes two figures beside them:

  success_vs_size.png      one line per policy over 15-60 mm cubes
  success_vs_position.png  one 5x5 map per policy over the trained spawn box

TWO SUCCESS BARS ON THE SIZE FIGURE, and the second is not decoration. The
task's bar is ABSOLUTE -- the cube's CENTRE 100 mm above the work surface, held
1 s -- so a 15 mm cube has to climb 92.5 mm to clear it and a 60 mm cube only
70. Scored that way alone, a drop at the small end cannot be told apart from the
bar simply asking more there. The right panel re-scores every episode on the
clearance a 50 mm cube would need (cube BOTTOM 75 mm up, same 1 s hold), which
is why the whole height trace is kept in the npz rather than a summary.

THE MAP IS DRAWN AS A BENCH, NOT AS AN ARRAY. The arm's base is at the world
origin, +x runs down the bench away from it and +y is the arm's LEFT, so the
figure puts far-from-the-arm at the top and +y on the left -- i.e. the view a
person standing behind the arm has. An imshow of the raw array would be that
picture rotated and mirrored, which is worse than no map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

# Fixed order, and the colour follows the POLICY rather than its rank, so a
# figure redrawn with one of them missing does not repaint the others.
ORDER = ("R1-SatPen-All", "R1-FaceLevel-All", "R2-Reach", "R2-Small")
COLORS = {
  "R1-SatPen-All": "#2a78d6",
  "R1-FaceLevel-All": "#eb6834",
  "R2-Reach": "#1baf7a",
  "R2-Small": "#eda100",
}
MARKERS = {
  "R1-SatPen-All": "o",
  "R1-FaceLevel-All": "s",
  "R2-Reach": "^",
  "R2-Small": "D",
}
# What each policy's cube size DR actually spanned, in mm of edge length. Drawn
# under the axis so the out-of-distribution part of every curve is visible as
# such rather than having to be remembered.
TRAINED_MM = {
  "R1-SatPen-All": (42.5, 57.5),
  "R1-FaceLevel-All": (42.5, 57.5),
  "R2-Reach": (45.0, 55.0),
  "R2-Small": (15.0, 57.5),
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
# Sequential blue, steps 100 -> 700 of the one-hue ramp.
BLUES = LinearSegmentedColormap.from_list(
  "seq_blue",
  ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
NOMINAL_HALF = 0.025


def wilson(k: np.ndarray, n: int, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
  """95% interval on a proportion. Wilson, not normal: at 100/100 the normal
  interval is zero-width, which would draw the most confident point on the
  figure as having no uncertainty at all."""
  p = k / n
  d = 1.0 + z * z / n
  c = (p + z * z / (2 * n)) / d
  hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return c - hw, c + hw


def held(h: np.ndarray, bar: np.ndarray | float, hold: int) -> np.ndarray:
  """Fraction of episodes whose cube stayed above `bar` for `hold` steps.

  `h` is [..., T, B] and `bar` broadcasts against [..., 1, B], so a per-episode
  bar (the size-fair one, which depends on that episode's cube) is the same code
  path as a constant one.
  """
  above = h > bar
  run = np.zeros(above.shape[:-2] + above.shape[-1:], dtype=np.int32)
  cur = np.zeros_like(run)
  for t in range(above.shape[-2]):
    cur = np.where(above[..., t, :], cur + 1, 0)
    run = np.maximum(run, cur)
  return (run >= hold).astype(np.float64)


def load(d: Path) -> dict[str, dict]:
  out = {}
  for name in ORDER:
    p = d / f"{name}.npz"
    if p.exists():
      out[name] = dict(np.load(p, allow_pickle=True))
  if not out:
    raise SystemExit(f"no {'/'.join(ORDER)} .npz files under {d}")
  return out


# ----------------------------------------------------------------- size figure
# Labelled by a legend rather than at the line ends, which is the exception to
# direct-labelling four series: all four curves converge inside two points of
# each other above 45 mm, so an end label has to be shoved away from its own
# line to be read at all -- and a label 12 points below the curve it names is a
# worse lie than a legend. The marker shapes carry identity redundantly.
def size_figure(runs: dict[str, dict], out: Path) -> None:
  fig, axs = plt.subplots(
    2, 2, figsize=(12.6, 6.0), sharey="row", sharex="col", height_ratios=[1, 0.16]
  )
  axes, rugs = axs[0], axs[1]
  handles: list = []
  for ax, mode, title in (
    (axes[0], "task", "The task's bar\ncube centre 100 mm up, held 1 s"),
    (
      axes[1],
      "fair",
      "Size-fair bar\ncube bottom 75 mm up (what a 50 mm cube needs), held 1 s",
    ),
  ):
    for name, r in runs.items():
      if "size_h" not in r:
        continue
      mm = r["sizes_mm"]
      h = r["size_h"].astype(np.float32)  # [S, T, B]
      hold = int(r["hold_steps"])
      if mode == "task":
        bar = float(r["lift_height"])
      else:
        # Per-episode: the same clearance under the cube that the bar implies
        # for a 50 mm one, so the geometry of the metric stops varying with the
        # thing being measured.
        bar = float(r["lift_height"]) - NOMINAL_HALF + r["size_half"][:, None, :]
      s = held(h, bar, hold)
      p = s.mean(-1)
      lo, hi = wilson(s.sum(-1), s.shape[-1])
      c = COLORS[name]
      line = ax.errorbar(
        mm,
        p * 100,
        yerr=np.stack([(p - lo) * 100, (hi - p) * 100]),
        color=c,
        lw=2,
        marker=MARKERS[name],
        ms=7,
        mew=1.5,
        mfc="white",
        elinewidth=1,
        capsize=2.5,
        ecolor=c,
        zorder=3,
        label=name,
      )
      if mode == "task":
        handles.append(line)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    ax.set_xlim(11, 74)
    ax.set_ylim(-3, 104)
    ax.grid(axis="y", color=MUTED, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
  axes[0].set_ylabel("success rate (%)", fontsize=10, color=INK2)
  # The trained size range per policy, on its own strip under each panel: the
  # part of a curve that sits outside its own strip is EXTRAPOLATION, and that
  # has to be visible without being remembered. Same colour as the line, so it
  # needs no second legend.
  for k, ax in enumerate(rugs):
    for i, name in enumerate(n for n in ORDER if n in runs):
      lo, hi = TRAINED_MM[name]
      ax.plot([lo, hi], [-i, -i], color=COLORS[name], lw=3.5, solid_capstyle="butt")
    ax.set_ylim(-len(runs) + 0.4, 0.6)
    ax.set_xlabel("cube edge length (mm)", fontsize=10, color=INK2)
    ax.set_xticks(list(range(15, 65, 5)))
    ax.set_yticks([])
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right", "left"):
      ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    if k == 0:
      ax.text(
        11.4, 0.55, "trained size range", fontsize=8, color=INK2, va="top", ha="left"
      )

  n = int(next(iter(runs.values()))["num_envs"])
  fig.suptitle(
    "Camera-only students: success against object size",
    fontsize=13,
    color=INK,
    x=0.008,
    ha="left",
    y=0.985,
  )
  fig.text(
    0.008,
    0.915,
    f"{n} episodes per point, spawn drawn over each policy's own trained box."
    "  Bars are 95% Wilson intervals.",
    fontsize=9,
    color=INK2,
    ha="left",
  )
  fig.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.006, 0.905),
    ncol=len(handles),
    frameon=False,
    fontsize=9.5,
    handletextpad=0.5,
    columnspacing=2.0,
    labelcolor=INK2,
  )
  fig.tight_layout(rect=(0, 0.02, 1, 0.86))
  fig.savefig(out, dpi=170, facecolor="white")
  print(f"wrote {out}")


# ----------------------------------------------------------------- grid figure
def _bench_bounds(xe, ye, nx: int, ny: int) -> dict | None:
  """Where the BENCH stops answering, in the heatmap's own index coordinates.

  A map swept past the trained spawn box measures two things at once -- the
  policy, and whether the cube was visible and reachable at all. Beyond the
  D435's cone the student is looking at an empty table; past ~720 mm the arm
  needs more than 1.5 sigma of action to get there. Both are properties of the
  bench that no checkpoint can fix, and a dead cell that is not marked as one of
  them reads as a policy failure.

  Returns None (and draws nothing) when the swept box is inside the trained one,
  where every cell passes and the contours would be empty furniture.
  """
  from wide_spawn_gate import MIN_RANGE, SIGMA_LIMIT, feasibility_map

  fx = np.linspace(xe[0], xe[-1], 41)
  fy = np.linspace(ye[0], ye[-1], 41)
  fm = feasibility_map(fx, fy)
  # MARGIN fields, contoured at zero, rather than a 0/1 mask contoured at 0.5.
  # A boolean field only ever changes value at a sample, so its contour is the
  # staircase of the sampling grid; the margins are smooth and their zero level
  # lands where the constraint actually bites.
  margins = {
    "cam": [fm["range"] - MIN_RANGE, 1.0 - fm["frame"]],
    # NaN is "the IK never converged", i.e. no reach at all rather than a small
    # sigma, so it is pinned to the failing side instead of being interpolated.
    "reach": [SIGMA_LIMIT - np.nan_to_num(fm["sigma"], nan=1e3)],
  }
  if all(m.min() > 0 for ms in margins.values() for m in ms):
    return None
  # Physical -> imshow index: row 0 is the LARGEST x, col 0 the largest y.
  cx = (xe[-1] - xe[0]) / nx
  cy = (ye[-1] - ye[0]) / ny
  return dict(
    rows=(xe[-1] - fx) / cx - 0.5,
    cols=(ye[-1] - fy) / cy - 0.5,
    margins=margins,
    cell=(cx, cy),
    edges=(xe, ye),
  )


def _draw_bounds(ax, b: dict | None, r: dict) -> None:
  if b is None:
    return
  for field, style in (("cam", "solid"), ("reach", "dashed")):
    for margin in b["margins"][field]:
      ax.contour(
        b["cols"],
        b["rows"],
        margin,
        levels=[0.0],
        colors=[INK],
        linewidths=1.6,
        linestyles=style,
        zorder=2,
      )
  # The trained spawn box, so "outside what it was asked to do" is visible on
  # the same picture as "outside what the bench allows".
  xe, ye = b["edges"]
  cx, cy = b["cell"]
  bx, by = r["box_x"], r["box_y"]
  x0, x1 = (xe[-1] - bx[1]) / cx - 0.5, (xe[-1] - bx[0]) / cx - 0.5
  y0, y1 = (ye[-1] - by[1]) / cy - 0.5, (ye[-1] - by[0]) / cy - 0.5
  ax.add_patch(
    plt.Rectangle(
      (y0, x0),
      y1 - y0,
      x1 - x0,
      fill=False,
      edgecolor="#e34948",
      lw=1.8,
      linestyle=(0, (4, 2)),
      zorder=5,
    )
  )


def grid_figure(runs: dict[str, dict], out: Path) -> None:
  have = [n for n in ORDER if n in runs and "grid_h" in runs[n]]
  cells = {
    n: held(
      runs[n]["grid_h"].astype(np.float32),
      float(runs[n]["lift_height"]),
      int(runs[n]["hold_steps"]),
    ).mean(-1)
    * 100
    for n in have
  }
  # The colour scale starts where the DATA does, not at zero. Every cell here is
  # in the nineties, and a 0-100 ramp paints all 100 of them the same navy --
  # the 8-point spread between the best cell and the worst, which is a factor of
  # five in failures, disappears. The floor is stated on the colourbar and every
  # cell carries its own number, so nothing is hidden by the choice; a 0-100 ramp
  # would hide the finding instead.
  vmin = min(5.0 * np.floor(s.min() / 5.0) for s in cells.values())
  xe = runs[have[0]]["grid_x_edges"]
  ye = runs[have[0]]["grid_y_edges"]
  for n_ in have[1:]:
    assert np.allclose(runs[n_]["grid_x_edges"], xe), "grids differ; cannot overlay"
    assert np.allclose(runs[n_]["grid_y_edges"], ye), "grids differ; cannot overlay"
  nx, ny = cells[have[0]].shape
  # Panels drawn to the bench's own proportions, so distances on the picture are
  # distances on the table however wide the swept box is.
  pw = 5.6
  ph = pw * (xe[-1] - xe[0]) / (ye[-1] - ye[0])
  fig, axes = plt.subplots(2, 2, figsize=(2 * pw + 1.9, 2 * ph + 2.1))
  bounds = _bench_bounds(xe, ye, nx, ny)
  fmt = "{:.1f}" if ny <= 6 else "{:.0f}"
  for ax, name in zip(axes.flat, have, strict=False):
    r = runs[name]
    s = cells[name]
    # Bench view: far from the arm at the top, +y (the arm's left) on the left.
    d = s[::-1, ::-1]
    cell = ((xe[-1] - xe[0]) / d.shape[0]) / ((ye[-1] - ye[0]) / d.shape[1])
    im = ax.imshow(d, cmap=BLUES, vmin=vmin, vmax=100, aspect=cell)
    for i in range(d.shape[0]):
      for j in range(d.shape[1]):
        ax.text(
          j,
          i,
          fmt.format(d[i, j]),
          ha="center",
          va="center",
          fontsize=9.5 if ny <= 6 else 8.0,
          fontweight="bold",
          color="white" if d[i, j] > vmin + 0.55 * (100 - vmin) else INK,
          zorder=4,  # above the bench-limit contours, which cross cells
        )
    _draw_bounds(ax, bounds, r)
    xc = ((xe[:-1] + xe[1:]) / 2 * 1000)[::-1]
    yc = ((ye[:-1] + ye[1:]) / 2 * 1000)[::-1]
    ax.set_xticks(range(len(yc)), [f"{v:+.0f}" for v in yc])
    ax.set_yticks(range(len(xc)), [f"{v:.0f}" for v in xc])
    ax.set_xlabel("y (mm)   +y = the arm's left", fontsize=9, color=INK2)
    ax.set_ylabel("x (mm) from the arm base", fontsize=9, color=INK2)
    ax.set_title(
      f"{name}      overall {s.mean():.1f}%   worst cell {s.min():.1f}%",
      fontsize=10.5,
      color=INK,
      loc="left",
      pad=8,
    )
    ax.tick_params(colors=INK2, labelsize=8.5 if ny <= 6 else 7.5, length=0)
    for side in ax.spines.values():
      side.set_visible(False)
  for ax in axes.flat[len(have) :]:
    ax.set_visible(False)

  n = int(next(iter(runs.values()))["num_envs"])
  # Header laid out in INCHES from the top, not in figure fractions: the panel
  # aspect follows the swept box, so a fraction that clears the title on the
  # trained box overlaps it on a box twice as wide.
  h = fig.get_figheight()
  fig.suptitle(
    "Camera-only students: success against where on the bench the cube is",
    fontsize=13,
    color=INK,
    x=0.008,
    ha="left",
    y=1 - 0.30 / h,
  )
  span = (
    f"{(xe[-1] - xe[0]) * 1000:.0f} x {(ye[-1] - ye[0]) * 1000:.0f} mm of bench"
    if bounds is not None
    else "the trained spawn box"
  )
  fig.text(
    0.008,
    1 - 0.58 / h,
    f"50 mm cube, random yaw. {n} episodes per cell, spawned uniformly inside"
    f" it; the {nx * ny} cells tile {span}. Viewed from above, arm base below."
    + (
      "\nThe D435 line is the gate's conservative test -- ALL EIGHT cube corners"
      " in frame -- so a partly-visible cube can still be grasped past it."
      if bounds is not None
      else ""
    ),
    fontsize=9,
    color=INK2,
    ha="left",
    va="top",
  )
  fig.subplots_adjust(top=1 - (1.50 if bounds is not None else 0.95) / h)
  if bounds is not None:
    fig.legend(
      handles=[
        Line2D(
          [],
          [],
          color=INK,
          lw=1.6,
          label="D435: cube not FULLY in frame, or nearer than 200 mm",
        ),
        Line2D(
          [],
          [],
          color=INK,
          lw=1.6,
          ls="dashed",
          label="arm limit: > 1.5σ of action to reach",
        ),
        Line2D(
          [],
          [],
          color="#e34948",
          lw=1.8,
          ls=(0, (4, 2)),
          label="the box these policies were trained on",
        ),
      ],
      loc="upper left",
      bbox_to_anchor=(0.006, 1 - 1.02 / h),
      ncol=3,
      frameon=False,
      fontsize=9,
      labelcolor=INK2,
    )
  cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02)
  cb.set_label(
    f"success rate (%) — colour spans {vmin:.0f}-100", fontsize=9, color=INK2
  )
  cb.ax.tick_params(colors=INK2, labelsize=8.5)
  cb.outline.set_visible(False)
  fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
  print(f"wrote {out}")


# ----------------------------------------------------------------------- table
def tables(runs: dict[str, dict]) -> None:
  names = [n for n in runs]
  first = next(iter(runs.values()))
  if "size_h" in first:
    print(f"\nsuccess vs cube edge length, {int(first['num_envs'])} episodes each")
    print("  task bar (cube centre 100 mm, held 1 s) / size-fair bar in brackets\n")
    print("  size mm  " + "".join(f"{n:>22}" for n in names))
    for k, mm in enumerate(first["sizes_mm"]):
      row = f"  {mm:7.0f}  "
      for n in names:
        r = runs[n]
        h = r["size_h"].astype(np.float32)
        hold = int(r["hold_steps"])
        a = held(h[k], float(r["lift_height"]), hold).mean() * 100
        b = (
          held(
            h[k],
            float(r["lift_height"]) - NOMINAL_HALF + r["size_half"][k][None, :],
            hold,
          ).mean()
          * 100
        )
        row += f"{a:15.1f}% [{b:4.1f}]"
      print(row)
  if "grid_h" in first:
    print("\nsuccess vs spawn cell (rows: far from the arm first; +y left)\n")
    for n in names:
      r = runs[n]
      s = (
        held(
          r["grid_h"].astype(np.float32), float(r["lift_height"]), int(r["hold_steps"])
        ).mean(-1)
        * 100
      )
      print(f"  {n}   overall {s.mean():.1f}%   worst {s.min():.1f}%")
      for row in s[::-1, ::-1]:
        print("    " + "".join(f"{v:7.1f}" for v in row))


def main() -> None:
  d = Path(sys.argv[1] if len(sys.argv) > 1 else "viz/sweep")
  runs = load(d)
  tables(runs)
  if any("size_h" in r for r in runs.values()):
    size_figure(runs, d / "success_vs_size.png")
  if any("grid_h" in r for r in runs.values()):
    grid_figure(runs, d / "success_vs_position.png")


if __name__ == "__main__":
  main()
