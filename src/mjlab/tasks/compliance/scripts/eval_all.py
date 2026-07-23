"""Aggregate the §7 metrics across all runs into one comparison table (plan §13).

Scans the experiment log dir, evaluates each run's latest checkpoint at level 3
(plus the analytic baseline), and prints a markdown table of the five headline
metrics. Run after E2/E3/E4 finish:

  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.eval_all
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import tyro

_LOG_DIR = Path("logs/rsl_rl/compliance_reach_flexiv")

# Run-name prefix -> registered task id (ablations need their own env/policy cfg).
_TASK_BY_PREFIX = {
  "a1_scalark": "Mjlab-Compliance-Reach-Flexiv-ScalarK",
  "a2_mlp": "Mjlab-Compliance-Reach-Flexiv-MLP",
  "a3_nocurric": "Mjlab-Compliance-Reach-Flexiv-NoCurriculum",
}
_MAIN = "Mjlab-Compliance-Reach-Flexiv"

_PATTERNS = {
  "yield": r"Yield ratio\s*:\s*mean=([-\d.]+)",
  "peak_f": r"Peak .*:\s*mean=([-\d.]+)",
  "recovery": r"Recovery time.*:\s*mean=([-\d.]+)",
  "aniso": r"anisotropy.*:\s*mean=([-\d.]+)",
  "success": r"Success rate\s*:\s*([-\d.]+)",
}


def _task_for(run_name: str) -> str:
  for prefix, task in _TASK_BY_PREFIX.items():
    if prefix in run_name:
      return task
  return _MAIN


def _run_eval(args: list[str]) -> dict[str, str]:
  out = subprocess.run(
    ["uv", "run", "python", "-m", "mjlab.tasks.compliance.scripts.eval", *args],
    capture_output=True,
    text=True,
    env={"MUJOCO_GL": "egl", "PATH": __import__("os").environ["PATH"]},
  ).stdout
  return {
    k: (m.group(1) if (m := re.search(p, out)) else "n/a") for k, p in _PATTERNS.items()
  }


def main(n_episodes: int = 200, num_envs: int = 64, device: str = "cpu") -> None:
  common = [
    "--n-episodes",
    str(n_episodes),
    "--num-envs",
    str(num_envs),
    "--device",
    device,
  ]

  rows: list[tuple[str, dict[str, str]]] = []
  print("evaluating analytic baseline ...")
  rows.append(("E1 analytic baseline", _run_eval(["--analytic", *common])))

  suite = ("e2_seed", "e3_t", "e4_a")
  runs = sorted(
    d
    for d in _LOG_DIR.glob("*")
    if d.is_dir() and list(d.glob("model_*.pt")) and any(k in d.name for k in suite)
  )
  for d in runs:
    run_name = d.name.split("_", 3)[-1]  # strip timestamp prefix
    task = _task_for(run_name)
    print(f"evaluating {run_name} ({task}) ...")
    rows.append(
      (run_name, _run_eval(["--task", task, "--checkpoint", str(d), *common]))
    )

  hdr = ["run", "yield", "peak_F(N)", "recovery(s)", "K∥/K⊥", "success"]
  print("\n| " + " | ".join(hdr) + " |")
  print("|" + "|".join(["---"] * len(hdr)) + "|")
  for name, m in rows:
    print(
      f"| {name} | {m['yield']} | {m['peak_f']} | {m['recovery']} | {m['aniso']} | {m['success']} |"
    )


if __name__ == "__main__":
  tyro.cli(main)
