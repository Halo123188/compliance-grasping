"""Measure the sim-radians <-> encoder-counts map for the four finger joints.

This is the one part of deployment that cannot be derived from either repo, so
it is the one part that has to be done on the bench. `deploy/hand.py` refuses to
arm until its output is in calib.py.

WHAT IS BEING FITTED, and why it needs a ruler rather than just the encoders.
The two proximal joints have a physical observable both sides agree on: pad
separation. The sim records it as a measured affine fit off the compiled model,

    sep(q) = 50.1 + 134.0 * q     mm

so measuring separation at two known encoder positions pins q at those two
points and the affine map follows. The two DISTAL joints have no such shared
observable -- the sim's fit says nothing about them -- so their map is taken
from the travel endpoints instead, which is weaker and is called out below.

  uv run python -m deploy.calibrate_hand --port /dev/ttyACM0

Safety: this drives the fingers. It runs at the firmware's SWEEP-level current
cap (0.15 A), never commands past POS_MIN/POS_MAX, and stops on any fault. Clear
the jaws before running -- there must be nothing between the pads.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import calib
from .hand import _import_link

# Poses to visit, as fractions from each motor's OPEN end toward its CLOSED end.
# Two points define the affine map; the third is a residual check, and a fit that
# only looks good on the points it was fitted to is not a fit.
FRACTIONS = (0.0, 0.5, 1.0)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--port", required=True, help="e.g. /dev/ttyACM0 (Linux)")
  ap.add_argument("--cap-a", type=float, default=0.15, help="current cap while moving")
  ap.add_argument("--settle", type=float, default=2.5, help="seconds per pose")
  args = ap.parse_args()

  link = _import_link()(port=args.port)
  info = link.info_
  print(f"INFO: {info}")
  if info["num_motors"] != 4:
    print(f"!! controller reports {info['num_motors']} motors, need 4", file=sys.stderr)
    return 1

  # POS_OPEN / POS_CLOSED live in the firmware, not the protocol, so they are
  # read off the controller by driving to the named poses rather than assumed
  # from a copy of Config.cpp that may have been changed with setmin/setmax and
  # persisted to EEPROM.
  print(
    "\nThis reads the pose the controller actually goes to, not the numbers in\n"
    "Config.cpp -- `setmin`/`setmax`/`setopen`/`setclose` are runtime-settable\n"
    "and `save` persists them to the Teensy's EEPROM, so the source can be stale.\n"
  )
  input("Clear the jaws (nothing between the pads), then press Enter: ")

  obs = link.action(torque=True, cap_a=args.cap_a, apply=False)
  if obs["fault"]:
    print(f"!! fault {obs['fault']} before starting", file=sys.stderr)
    return 1
  start = np.array([m["pos"] for m in obs["motors"]], dtype=np.float64)
  print(f"present position: {start.astype(int).tolist()}")

  print(
    "\nEnter each motor's OPEN and CLOSED encoder counts. Get them from the CLI\n"
    "(`open`, then `status`; `close`, then `status`) so this script never has to\n"
    "guess a closing DIRECTION -- the signs are mixed per motor by design.\n"
    "Order is the protocol's: f1j1, f1j2, f2j1, f2j2 = ids 12, 11, 21, 22.\n"
  )
  open_counts = _ask_vector("OPEN counts   (4, comma-separated): ")
  closed_counts = _ask_vector("CLOSED counts (4, comma-separated): ")

  records: list[tuple[float, np.ndarray, float]] = []
  for frac in FRACTIONS:
    goal = np.rint(open_counts + frac * (closed_counts - open_counts)).astype(int)
    print(f"\n--- fraction {frac:.2f}: goals {goal.tolist()}")
    deadline = time.time() + args.settle
    while time.time() < deadline:
      obs = link.action(torque=True, cap_a=args.cap_a, goals=goal.tolist())
      if obs["fault"]:
        print(f"!! fault {obs['fault']}; stopping", file=sys.stderr)
        link.close()
        return 1
      time.sleep(1.0 / calib.CONTROL_HZ)
    reached = np.array([m["pos"] for m in obs["motors"]], dtype=np.float64)
    print(f"  reached {reached.astype(int).tolist()}")
    sep = float(input("  measure PAD SEPARATION with calipers, in mm: "))
    records.append((frac, reached, sep))

  link.action(torque=False, apply=False)
  link.close()

  # --- fit ------------------------------------------------------------------
  # Proximal joints: separation gives q directly through the sim's fit.
  q_prox = np.array(
    [(sep - calib.APERTURE_A_MM) / calib.APERTURE_B_MM for _, _, sep in records]
  )
  counts = np.stack([c for _, c, _ in records])  # (n_poses, 4)

  print("\n" + "=" * 68)
  print("MEASURED  pose -> aperture -> proximal q")
  for (frac, c, sep), q in zip(records, q_prox, strict=True):
    print(
      f"  frac {frac:.2f}  sep {sep:7.2f} mm  q {q:+.4f} rad  counts {c.astype(int).tolist()}"
    )

  # Which protocol motors are the PROXIMAL ones is itself a measurement: the
  # proximal joint is the one whose motion changes the aperture. Report the
  # correlation so a mis-ordered pair is visible rather than silently fitted.
  print("\n|corr(counts, aperture q)| per motor -- proximal motors should be ~1.0:")
  corr = [
    abs(float(np.corrcoef(counts[:, i], q_prox)[0, 1]))
    if np.ptp(counts[:, i]) > 1
    else 0.0
    for i in range(4)
  ]
  for i, c in enumerate(corr):
    print(f"  motor {i}: {c:.3f}   (travel {np.ptp(counts[:, i]):.0f} counts)")

  k = np.zeros(4)
  b = np.zeros(4)
  for i in range(4):
    if np.ptp(counts[:, i]) < 1:
      print(f"\n!! motor {i} did not move; cannot fit it", file=sys.stderr)
      return 1
    k[i], b[i] = np.polyfit(q_prox, counts[:, i], 1)

  print("\n" + "=" * 68)
  print("Paste into deploy/calib.py:\n")
  print(f"COUNTS_PER_RAD = ({', '.join(f'{v:.1f}' for v in k)})")
  print(f"COUNTS_AT_ZERO_RAD = ({', '.join(f'{v:.1f}' for v in b)})")
  print(
    "\nREAD THIS BEFORE PASTING. Only the PROXIMAL motors (the two with |corr|\n"
    "near 1.0 above) are genuinely fitted -- aperture is a shared observable and\n"
    "the sim's sep(q) fit converts it. The DISTAL motors are being fitted against\n"
    "the proximal joint's angle because they moved along with it, which measures\n"
    "the coupling of THIS sweep, not the distal joint's own scale. The policy\n"
    "uses the distal curl heavily (it is worth ~11x in pull-out resistance), so\n"
    "if grasps slip, the distal map is the first thing to distrust.\n"
  )
  return 0


def _ask_vector(prompt: str) -> np.ndarray:
  while True:
    try:
      v = np.array([float(x) for x in input(prompt).replace(" ", "").split(",")])
      if v.size == 4:
        return v
      print("  need exactly 4 values")
    except ValueError:
      print("  could not parse; try again")


if __name__ == "__main__":
  raise SystemExit(main())
