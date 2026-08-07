"""Attribute a camera student's score drop to a SPECIFIC perception knob.

The DR student scores 60.9% against the no-DR student's 100%. "Perception DR
costs 39 points" is not an actionable finding -- three of the perception ranges
have no measurement behind them (the 2.5 deg camera yaw, and both dropout
rates, which upstream leaves at zero and never measured), and any of them could
be carrying the whole cost or none of it.

This re-scores ONE FIXED CHECKPOINT under progressively narrower environments.
The student's weights never change, so every difference is attributable to the
env. Each config runs in its own subprocess because the DR switches are read at
import time, and because building several camera envs in one process trips
mujoco-warp's CUDA graph capture.

Read the result as: how much of the 39 points does this policy get back when
the knob is removed at TEST time. A knob whose removal recovers nothing was not
what broke it -- narrowing that one and retraining would be wasted GPU time. A
knob whose removal recovers most of it is the one to re-examine, and the first
question to ask about it is whether its range was ever measured.

  uv run python scripts/wide_dr_ablate.py <student_checkpoint.pt>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Overridable because the -Fine arm is a different NETWORK, not just a different
# schedule: scoring a fine_cnn checkpoint against the base task builds the wrong
# encoder. `strict=True` in the loader catches it, but only after paying for the
# env build, so pass the task the checkpoint was trained under.
TASK = os.environ.get("CG_WIDE_ABLATE_TASK") or (
  "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr"
)

# (label, env overrides). The baseline repeats the training env so the table is
# self-contained and so a mismatch against the previously reported number shows
# up here rather than being assumed away.
CONFIGS: tuple[tuple[str, dict[str, str]], ...] = (
  ("full DR (as trained)", {}),
  ("cam yaw 2.5 -> 0.5 deg", {"CG_WIDE_CAM_YAW_DEG": "0.5"}),
  ("no camera-pose DR", {"CG_WIDE_CAM_DR": "0"}),
  ("no depth-sensor DR", {"CG_WIDE_DEPTH_DR": "0"}),
  ("physics DR only", {"CG_WIDE_CAM_DR": "0", "CG_WIDE_DEPTH_DR": "0"}),
  # Round 2: the 30 points live in camera POSE, and not in the yaw that this
  # repo widened. Split the pose half so the cost lands on translation or on
  # rotation before anyone edits a range.
  ("cam rotation only (no pos)", {"CG_WIDE_CAM_POS_DR": "0"}),
  ("cam translation only (no rot)", {"CG_WIDE_CAM_ROT_DR": "0"}),
  # Round 3: the 26 points are camera TRANSLATION. Split it per axis. A cost
  # concentrated on the axis nearest the optical axis means the mechanism is a
  # metric-depth bias (fix: calibrate, or give the policy a depth reference);
  # a cost spread evenly means it is apparent image motion (fix: resolution).
  ("cam trans axis X only", {"CG_WIDE_CAM_ROT_DR": "0", "CG_WIDE_CAM_POS_AXES": "0"}),
  ("cam trans axis Y only", {"CG_WIDE_CAM_ROT_DR": "0", "CG_WIDE_CAM_POS_AXES": "1"}),
  ("cam trans axis Z only", {"CG_WIDE_CAM_ROT_DR": "0", "CG_WIDE_CAM_POS_AXES": "2"}),
)


def main() -> None:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  if (
    not Path(ckpt.replace("/work/", "/export/work/")).exists()
    and not Path(ckpt).exists()
  ):
    raise SystemExit(f"checkpoint not found: {ckpt}")
  # Optional substring filter, so a follow-up round can run only the new
  # configs instead of paying for five already-measured re-runs.
  only = sys.argv[2] if len(sys.argv) > 2 else ""
  configs = [c for c in CONFIGS if only in c[0]]
  if not configs:
    raise SystemExit(f"no config matches {only!r}")

  rows: list[tuple[str, str, str, str]] = []
  for label, overrides in configs:
    print(f"\n=== {label}  {overrides or '(defaults)'}", flush=True)
    proc = subprocess.run(
      [
        sys.executable,
        "scripts/eval_deploy.py",
        f"abl={TASK}:{ckpt}",
      ],
      env={**os.environ, "MUJOCO_GL": "egl", **overrides},
      capture_output=True,
      text=True,
      timeout=3600,
    )
    out = proc.stdout + proc.stderr
    # eval_deploy prints one data row under the header; pull the numbers off it.
    succ = peak = jaw = "--"
    for line in out.splitlines():
      if line.strip().startswith("abl"):
        nums = re.findall(r"[-+]?\d+\.\d+(?:mm|d|%)?", line)
        if len(nums) >= 4 and succ == "--":
          succ, peak = nums[0], nums[3]
        elif len(nums) >= 4:
          jaw = nums[-3]
    if succ == "--":
      print(out[-1500:])
    rows.append((label, succ, peak, jaw))
    print(f"  -> success {succ}  median peak {peak}  jaw {jaw}", flush=True)

  print(f"\n{'config':<26} {'success':>9} {'median peak':>13} {'jaw err':>9}")
  for label, succ, peak, jaw in rows:
    print(f"{label:<26} {succ:>9} {peak:>13} {jaw:>9}")
  print("\nSame checkpoint throughout -- every difference is the ENV, not the policy.")


if __name__ == "__main__":
  main()
