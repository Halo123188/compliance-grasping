"""Aggregate eval blocks from SLURM logs into one comparison table.

Each training job ends by evaluating its final checkpoint and then the analytic
oracle on the same fixed batch, so every log holds two eval blocks. This parses
them out and prints student-vs-ceiling per seed, plus the mean across seeds.

  uv run python -m mjlab.tasks.compliance_tracking.scripts.collect_results \
      slurm_log/ct-A-s*.log

The column that decides whether the run worked is K_anisotropy_pull; see the
task README. Read it together with track_err_pull_cm: a policy can post a good
tracking error either by yielding with the teacher or by stiffening against the
hand, and only the anisotropy separates those two.
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

_HEADER = re.compile(r"=== Stage (\w+): (.+) ===")
_ROW = re.compile(r"^\s{2,}([A-Za-z_]\w*)\s+(-?[\d.]+)\s*$")

# Printed in this order; everything else parsed is kept but listed after.
_KEY_ORDER = [
  "track_err_all_cm",
  "track_err_unperturbed_cm",
  "track_err_pull_cm",
  "track_err_release_cm",
  "track_err_post_grasp_cm",
  "K_anisotropy_pull",
  "K_parallel_pull",
  "K_perp_pull",
  "release_overshoot_cm",
  "s_rate_pulled",
  "s_rate_free",
  "s_peak",
  "grasp_force_err_N",
  "grasp_force_err_pushed_N",
  "welded_fraction",
  "pulled_fraction",
  "peak_f_ext_N",
]


def parse_log(path: Path) -> list[tuple[str, str, dict[str, float]]]:
  """Return [(stage, label, metrics), ...] for each eval block in the log."""
  blocks: list[tuple[str, str, dict[str, float]]] = []
  stage = label = None
  metrics: dict[str, float] = {}
  for raw in path.read_text(errors="replace").splitlines():
    header = _HEADER.search(raw)
    if header:
      if stage is not None and metrics:
        blocks.append((stage, label or "?", metrics))
      stage, label, metrics = header.group(1), header.group(2), {}
      continue
    if stage is None:
      continue
    row = _ROW.match(raw)
    if row:
      metrics[row.group(1)] = float(row.group(2))
  if stage is not None and metrics:
    blocks.append((stage, label or "?", metrics))
  return blocks


def _is_oracle(label: str) -> bool:
  return "oracle" in label.lower()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("logs", nargs="+", type=Path)
  args = parser.parse_args()

  students: list[tuple[str, dict[str, float]]] = []
  oracles: list[dict[str, float]] = []
  stage_seen = set()
  for path in args.logs:
    if not path.exists():
      print(f"  (missing: {path})")
      continue
    for stage, label, metrics in parse_log(path):
      stage_seen.add(stage)
      if _is_oracle(label):
        oracles.append(metrics)
      else:
        students.append((path.stem, metrics))

  if not students and not oracles:
    raise SystemExit("no eval blocks found — did the jobs reach their eval leg?")

  keys = [
    k
    for k in _KEY_ORDER
    if any(k in m for _, m in students) or any(k in m for m in oracles)
  ]
  extra = sorted({k for _, m in students for k in m} | {k for m in oracles for k in m})
  keys += [k for k in extra if k not in keys]

  names = [n for n, _ in students]
  width = max([len(k) for k in keys] + [12])
  col = max([len(n) for n in names] + [10]) + 2

  print(f"\n=== Stage {'/'.join(sorted(stage_seen))} ===")
  print(f"{len(students)} student run(s), {len(oracles)} oracle run(s)\n")
  head = f"{'metric':<{width}}" + "".join(f"{n:>{col}}" for n in names)
  if len(students) > 1:
    head += f"{'mean':>{col}}"
  head += f"{'oracle':>{col}}"
  print(head)
  print("-" * len(head))

  for key in keys:
    vals = [m.get(key) for _, m in students]
    line = f"{key:<{width}}"
    for v in vals:
      line += f"{v:>{col}.4f}" if v is not None else f"{'-':>{col}}"
    present = [v for v in vals if v is not None]
    if len(students) > 1:
      line += f"{statistics.fmean(present):>{col}.4f}" if present else f"{'-':>{col}}"
    o = [m[key] for m in oracles if key in m]
    line += f"{statistics.fmean(o):>{col}.4f}" if o else f"{'-':>{col}}"
    print(line)

  # Ratio of the pooled means, not the mean of per-seed ratios: see the warning
  # on mdp.metrics.k_anisotropy_ratio. Recomputed here from K_parallel/K_perp so
  # the verdict is unbiased even for logs written before the eval fix.
  pars = [m["K_parallel_pull"] for _, m in students if "K_parallel_pull" in m]
  perps = [m["K_perp_pull"] for _, m in students if "K_perp_pull" in m]
  if pars and perps:
    mean_aniso = statistics.fmean(pars) / max(statistics.fmean(perps), 1e-3)
    print(
      f"\n  K_parallel/K_perp while pulled = {mean_aniso:.3f} across "
      f"{len(pars)} seed(s)  [ratio of means]"
    )
    if mean_aniso < 0.9:
      print("  -> the policy softens along the pull: genuine arbitration.")
    else:
      print(
        "  -> NOT softening along the pull. The policy is tracking by stiffening,\n"
        "     which an analytic law already does. Treat a good tracking error here\n"
        "     as uninformative and investigate before claiming compliance."
      )


if __name__ == "__main__":
  main()
