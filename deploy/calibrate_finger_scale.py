"""Measure counts-per-radian for ONE finger joint, against a protractor.

`calib.COUNTS_PER_RAD` is the last guessed number in the deployment. It is
651.9 = 4096 / 2pi, which assumes the joint turns 1:1 with the motor's output
shaft, and nothing in either repo states a linkage ratio. The evidence in
calib.py says the assumption is probably wrong in a specific direction: f1j1's
taught OPEN pose is +221 counts where 1:1 predicts 179, which would put the true
scale nearer 805 counts/rad -- i.e. every commanded finger angle under-travels
by about 19%, and every reported one is over-read by the same factor.

That is not a cosmetic error. The 2026-08-11 hardware runs closed the jaws to
43-64 mm of pad separation on the approach, against a 50 mm cube that presents
70.7 mm at 45 degrees of yaw, and knocked it away; the two runs that arrived at
69 and 95 mm are the two that grasped. A claw that is systematically narrower
than the policy believes is exactly that failure.

WHY THIS AND NOT `calibrate_hand.py`. That script fits the map through PAD
SEPARATION, using the sim's `sep(q) = 50.1 + 134.0 * q` -- and that fit does not
reproduce the compiled model, which says 67.34 mm at q = 0 and 175.5 mm/rad
(the model, via `deploy.kinematics`, puts the pad sites 87.0 mm apart at the
home pose and 40.0 mm at full close). Calibrating against a constant that is
itself wrong buries one error inside another. This measures the joint angle
DIRECTLY, one joint at a time, and needs no aperture model at all.

TWO WAYS TO READ THE ANGLE, and they trade attribution against how easy the
reading is to take:

  --method gauge    A digital inclinometer on the link. Moves ONE motor, so it
                    measures that motor's scale and no other. Needs the joint's
                    axis mounted horizontal, which is the whole difficulty.
  --method caliper  A caliper across the jaw's OUTSIDE width, inverted through
                    OUTER_TABLE. Moves BOTH proximals in MIRROR. Needs no
                    fixturing and the arm can be in any pose -- the width is
                    internal to the hand -- but it reads the AVERAGE of the two
                    proximal scales, so it screens rather than settles.

                    Outside width and not the inner gap, which is the obvious
                    choice and the wrong one: the inner faces are parallel only
                    at q = 0, so past neutral the narrowest point slides up off
                    the pad and stops being findable. The outside also runs at
                    ~190 mm/rad against the gap's 68.

Start at NEUTRAL, before any of this. At 0 counts on all four motors the model
says the inner gap is 50.00 mm, the outside width 90.00 and one finger 20.00
thick, and those three over-determine the two things that can be wrong. Measured
on this hand 2026-08-11: 46.88 / ~87 / 20.00. The finger matches CAD and
`outer - inner` is exactly 2x thickness -- which only holds when the fingers are
PARALLEL -- so the angle is right and the zero is good; both widths are simply
3.12 mm short, i.e. the two fingers sit 1.56 mm inboard of where the model puts
them. That is `--width-offset-mm`, not a calibration constant.

THE PROCEDURE, which is half of the measurement:

  1. Mount so the joint's axis is HORIZONTAL. The proximal joints rotate about
     the hand frame's y axis, so pose the wrist (or unbolt the hand and stand it
     on the bench) until the finger swings in a VERTICAL plane. A gravity
     inclinometer cannot read rotation about a vertical axis at all, and a
     half-tilted axis reads a cosine-shrunk angle that looks like a plausible
     scale error -- the exact mistake this script exists to remove.
  2. Drive the joint to 0 counts, which is the gripper's homing zero, and which
     `calib.COUNTS_AT_ZERO_RAD = 0` claims IS the sim's zero (both fingers
     straight/neutral). Everything below is measured relative to it, so if that
     claim is wrong the slope is still right and only the offset moves.
  3. Zero the angle gauge on a FLAT MACHINED FACE of the link being moved --
     not on a printed fillet, and not on the finger pad.
  4. Run this. It steps the joint through a spread of counts, waits for it to
     settle, and asks for the gauge reading at each.

WHAT THE RESIDUALS ARE FOR. If the linkage is a four-bar rather than a direct
drive, joint angle is NOT affine in motor angle, and no single counts-per-radian
exists -- the fit will still return one, and the residuals are the only thing
that says it is a lie. A residual well under a count is a direct drive; several
counts of systematic curvature means the map has to be a table or a polynomial,
and `deploy/hand.py` needs more than a scale.

  uv run python -m deploy.calibrate_finger_scale --port /dev/ttyACM0 --motor f1j1

Safety: this drives at the sweep-level current cap (0.15 A), leaves every motor
it is not sweeping holding position, and never disables the firmware's
POS_MIN/POS_MAX clamp. Clear the jaws first.

TORQUE IS HELD THROUGH EVERY WAIT, by `_Hold`. It has to be: the controller
drops torque 500 ms after the last ACTION frame, so without a heartbeat the
finger is limp by the time the caliper reaches it.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from . import calib
from .hand import _import_link

# Protocol motor order, which is the order `calib.COUNTS_PER_RAD` is indexed in:
# f1j1, f1j2, f2j1, f2j2 = ids 12, 11, 21, 22. Finger 1 is LEFT (calib.py).
MOTORS = ("f1j1", "f1j2", "f2j1", "f2j2")
SIM_JOINT = ("left_1", "left_2", "right_1", "right_2")

# Counts to visit, relative to the start. Symmetric so a fit cannot hide a sign
# error, and wide because the slope's error falls off with the span: at the two
# candidate scales a 600-count move is 52.7 deg (651.9) or 42.7 deg (805), which
# a +-0.1 deg gauge separates with three orders of magnitude to spare. The
# firmware clamps to POS_MIN/POS_MAX, and a clamped pose is still a valid data
# point because the ACHIEVED counts are what gets fitted.
DEFAULT_OFFSETS = (-600, -300, 0, 300, 600)

# Mirrored counts for the caliper method. All OPENING, and none of them near
# neutral: the outside width only carries the angle on the opening branch (see
# MIN_USABLE_OUTER_MM), so a point at 0 is not a free extra data point, it is a
# bad one. The intercept is recovered from the spread instead.
CALIPER_OFFSETS = (60, 110, 160, 210, 270, 330)

# Jaw gap against proximal angle, MIRRORED (left_1 = +q, right_1 = -q, both
# distals at 0), in mm. Measured on the compiled wide-claw model with
# `mj_geomDistance` between the fingertip collision boxes, which the visual
# meshes agree with exactly: at q = 0 both inner faces sit at |x| = 25.000 mm
# from the centreline, so the gap is 50.00 mm.
#
# Tabulated rather than fitted because it is NOT a straight line -- the slope is
# 167 mm/rad closing, 105 just past neutral and 75 further open, as the pair of
# corners that forms the narrowest gap changes. `deploy/` has no mujoco (that is
# the whole point of kinematics.py), so the numbers are transcribed here the same
# way that chain is. Regenerate by sweeping q on `twofinger_arm_spec().compile()`.
#
# THE FLAT BAND ONLY. The two inner faces are parallel and 50 mm apart over the
# stretch 87..118 mm below the hand base; the last ~8 mm at the fingertip is
# chamfered back and reads 62.5 mm instead. Measure in the middle of the flat.
GAP_TABLE: tuple[tuple[float, float], ...] = (
  (-0.300, 1.36),
  (-0.275, 5.24),
  (-0.250, 9.16),
  (-0.225, 13.11),
  (-0.200, 17.10),
  (-0.175, 21.12),
  (-0.150, 25.18),
  (-0.125, 29.26),
  (-0.100, 33.37),
  (-0.075, 37.50),
  (-0.050, 41.65),
  (-0.025, 45.82),
  (+0.000, 50.00),
  (+0.025, 52.63),
  (+0.050, 55.26),
  (+0.075, 57.91),
  (+0.100, 60.56),
  (+0.125, 63.22),
  (+0.150, 65.54),
  (+0.175, 67.42),
  (+0.200, 69.29),
  (+0.225, 71.17),
  (+0.250, 73.04),
  (+0.275, 74.92),
  (+0.300, 76.79),
  (+0.325, 78.65),
  (+0.350, 80.51),
  (+0.375, 82.37),
  (+0.400, 84.21),
  (+0.425, 86.05),
  (+0.450, 87.88),
  (+0.475, 89.70),
  (+0.500, 91.50),
  (+0.525, 93.29),
  (+0.550, 95.07),
)
NEUTRAL_GAP_MM = 50.00

# Overall OUTSIDE width of the jaw -- the caliper's external jaws across the two
# fingertips -- against the same mirrored angle. Same model, same sweep, taken
# over every mesh vertex of both finger links rather than a collision box.
#
# THIS IS WHAT THE SCALE IS MEASURED ON, not the inner gap, for two reasons:
#
#   * The inner faces are parallel ONLY at q = 0. Mirrored rotation tilts each
#     face by q, so the gap becomes a V and its narrowest point slides along the
#     finger -- at q = +0.20 it has moved 38 mm up toward the knuckle, off the
#     pad entirely. "The narrowest gap" stops being a thing a caliper can find.
#   * The outside width runs at ~190 mm/rad against the gap's 68, so the same
#     caliper resolves the angle three times better.
#
# ONLY IN THE OPENING DIRECTION. Below q = -0.125 the widest part of the jaw is
# the knuckle rather than the fingertip, and the reading sticks at 85.6 mm no
# matter how much further the fingers close -- flat, and therefore useless. The
# table stops at -0.10 for that reason and `outer_to_rad` refuses past it.
OUTER_TABLE: tuple[tuple[float, float], ...] = (
  (-0.100, 86.10),
  (-0.075, 87.09),
  (-0.050, 88.07),
  (-0.025, 89.04),
  (+0.000, 90.00),
  (+0.025, 94.97),
  (+0.050, 99.92),
  (+0.075, 104.86),
  (+0.100, 109.77),
  (+0.125, 114.66),
  (+0.150, 119.51),
  (+0.175, 124.34),
  (+0.200, 129.14),
  (+0.225, 133.90),
  (+0.250, 138.61),
  (+0.275, 143.29),
  (+0.300, 147.92),
  (+0.325, 152.50),
  (+0.350, 157.03),
  (+0.375, 161.50),
  (+0.400, 165.92),
  (+0.425, 170.27),
  (+0.450, 174.57),
  (+0.475, 178.80),
  (+0.500, 182.96),
  (+0.525, 187.05),
  (+0.550, 191.07),
)
NEUTRAL_OUTER_MM = 90.00


def gap_to_rad(gap_mm: float) -> float:
  """Measured jaw gap -> mirrored proximal angle, by inverting GAP_TABLE.

  Refuses to extrapolate. A reading outside the table is not a small error to
  be linearised away -- it means the jaws are somewhere this pose model does not
  describe, most likely because a distal joint is not at zero.
  """
  q = np.array([p[0] for p in GAP_TABLE])
  g = np.array([p[1] for p in GAP_TABLE])
  if not g[0] <= gap_mm <= g[-1]:
    raise ValueError(
      f"gap {gap_mm} mm is outside the table's {g[0]:.1f}..{g[-1]:.1f} mm. "
      "Check that both distal joints are at 0 counts."
    )
  return float(np.interp(gap_mm, g, q))


# Readings below this are thrown away rather than inverted. The table kinks hard
# at q = 0: opening, the width runs at ~198 mm/rad; closing, at ~38. On the
# shallow side a 0.5 mm reading error becomes 13 mrad instead of 2.5, and one
# such point is enough to drag a fit and to fake a curvature term -- measured
# 2026-08-11, where a single reading at 89.12 mm moved the answer from 1028 to
# 976 counts/rad and tripped the NOT AFFINE warning on residuals it created
# itself. The margin over 90.00 is there because a reading near the kink cannot
# be assigned to a branch at all.
MIN_USABLE_OUTER_MM = 92.0


def outer_to_rad(width_mm: float) -> float:
  """Measured outside width -> mirrored proximal angle, by inverting OUTER_TABLE."""
  q = np.array([p[0] for p in OUTER_TABLE])
  w = np.array([p[1] for p in OUTER_TABLE])
  if width_mm > w[-1]:
    raise ValueError(f"width {width_mm} mm is past the table's {w[-1]:.1f} mm end.")
  if width_mm < MIN_USABLE_OUTER_MM:
    raise ValueError(
      f"width {width_mm} mm is at or below the table's kink at "
      f"{NEUTRAL_OUTER_MM:.1f} mm, where the outside width barely responds to the "
      "angle (38 mm/rad against 198 opening). Open further and re-read; a point "
      "here is worse than no point. Below 86 mm the knuckle is the widest part "
      "and the reading carries no angle at all."
    )
  return float(np.interp(width_mm, w, q))


class _Hold:
  """Keep a commanded pose alive while the bench dialogue blocks.

  PROTOCOL.md, "Host watchdog": once torque is armed over the binary link, the
  controller drops it if no ACTION arrives for 500 ms -- deliberately, so a
  crashed policy cannot leave the gripper energised on a stale goal. Every wait
  in this script is longer than that: `--settle` is 2 s, and reading a caliper
  takes as long as it takes.

  Without a heartbeat the finger therefore goes limp precisely while it is being
  measured, and a limp finger sits wherever the jaws push it -- which is a
  measurement of the caliper, not of the joint. Re-sending the same goals at
  20 Hz holds the pose against that.

  The thread owns the link for its lifetime; nothing else may touch the link
  inside the `with`, which is why the OBS the caller needs is served from
  `last_obs` rather than by polling.
  """

  def __init__(self, link, goals: list[int], cap_a: float, hz: float = 20.0):
    self._link, self._goals, self._cap_a = link, goals, cap_a
    self._period = 1.0 / hz
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._lock = threading.Lock()
    self._last_obs: dict | None = None
    self.fault: object | None = None

  @property
  def last_obs(self) -> dict | None:
    with self._lock:
      return self._last_obs

  def __enter__(self) -> _Hold:
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    return self

  def _run(self) -> None:
    while True:
      try:
        obs = self._link.action(torque=True, cap_a=self._cap_a, goals=self._goals)
      except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        self.fault = repr(exc)
        return
      with self._lock:
        self._last_obs = obs
      if obs.get("fault"):
        self.fault = obs["fault"]
        return
      if self._stop.wait(self._period):
        return

  def __exit__(self, *exc_info) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)


def fit_scale(counts: np.ndarray, radians: np.ndarray) -> dict:
  """Least-squares counts = k * rad + b, plus what the residuals say about it.

  Separated from the bench dialogue so it can be tested without a gripper.
  """
  counts = np.asarray(counts, dtype=float)
  radians = np.asarray(radians, dtype=float)
  if counts.size < 3:
    raise ValueError("need at least 3 poses to see a residual at all")
  design = np.stack([radians, np.ones_like(radians)], axis=1)
  (k, b), *_ = np.linalg.lstsq(design, counts, rcond=None)
  resid = counts - (k * radians + b)
  # Curvature, not scatter: a four-bar's residual is a smooth arc through the
  # sweep, so the sign pattern of the residual is what distinguishes it from
  # gauge noise. This is that arc's amplitude, fitted as a quadratic term.
  quad = np.polyfit(radians, counts, 2)[0] if counts.size >= 4 else 0.0
  return {
    "counts_per_rad": float(k),
    "counts_at_zero_rad": float(b),
    "resid_rms": float(np.sqrt(np.mean(resid**2))),
    "resid_max": float(np.max(np.abs(resid))),
    "curvature": float(quad),
    "resid": resid,
  }


def _report(name: str, sim_joint: str, fit: dict, span_rad: float) -> None:
  k, b = fit["counts_per_rad"], fit["counts_at_zero_rad"]
  guess = calib.COUNTS_PER_RAD[MOTORS.index(name)] if calib.COUNTS_PER_RAD else None
  print(f"\n=== {name} ({sim_joint}) ===")
  print(f"  COUNTS_PER_RAD     {k:8.1f}")
  print(f"  COUNTS_AT_ZERO_RAD {b:8.1f}")
  if guess:
    err = 100.0 * (k - guess) / guess
    print(f"  against calib.py's {guess:8.1f}   ({err:+.1f}%)")
    print(
      f"  => every commanded angle is currently {100 * (k - guess) / k:+.1f}% "
      f"{'SHORT' if k > guess else 'LONG'} of what the policy asks for"
    )
  print(f"  residual  rms {fit['resid_rms']:.2f} counts, max {fit['resid_max']:.2f}")
  # A count is the finest thing the encoder can say; a residual under it means
  # the affine model is not what is limiting the answer.
  if fit["resid_max"] > 3.0:
    print(
      "  ^ NOT AFFINE. The joint does not turn linearly with the motor, so no\n"
      "    single counts-per-radian is correct and hand.py needs a table or a\n"
      "    polynomial. Curvature term "
      f"{fit['curvature']:.1f} counts/rad^2 over a {span_rad:.2f} rad span."
    )
  if abs(b) > 15.0:
    print(
      f"  ^ offset {b:.0f} counts is not zero. calib.COUNTS_AT_ZERO_RAD assumes\n"
      "    the homing zero IS the sim's zero pose; either the gauge was not\n"
      "    zeroed at 0 counts, or that assumption is wrong."
    )


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--port", required=True, help="e.g. /dev/ttyACM0 (Linux)")
  ap.add_argument(
    "--method",
    choices=("gauge", "caliper"),
    default="gauge",
    help="how the angle is read; see the module docstring",
  )
  ap.add_argument(
    "--motor",
    choices=MOTORS,
    help="gauge method only: which motor to move. caliper moves both proximals",
  )
  ap.add_argument(
    "--width-offset-mm",
    type=float,
    default=0.0,
    help=(
      "added to every caliper reading before the table lookup, for a jaw that is "
      "built narrower than the model. Measured 2026-08-11: the real jaw reads "
      "3.12 mm narrow at neutral in BOTH the inner gap (46.88 vs 50.00) and the "
      "outer width (86.88 vs 90.00), with one finger exactly the CAD's 20.00 mm "
      "thick -- so the two fingers sit 1.56 mm inboard each and the offset is a "
      "constant in width at every angle, NOT an angle error. It has to be "
      "corrected before the lookup because width -> angle is not linear; leaving "
      "it in the fit's intercept would bias the scale."
    ),
  )
  ap.add_argument("--cap-a", type=float, default=0.15, help="current cap while moving")
  ap.add_argument("--settle", type=float, default=2.0, help="seconds per pose")
  ap.add_argument(
    "--offsets",
    type=int,
    nargs="*",
    default=None,
    metavar="COUNTS",
    help="counts to visit, relative to the pose this starts from",
  )
  args = ap.parse_args()

  caliper = args.method == "caliper"
  if caliper and args.motor:
    print("!! --motor is for --method gauge; caliper moves both proximals.")
    return 1
  if not caliper and not args.motor:
    print("!! --method gauge needs --motor (f1j1, f1j2, f2j1 or f2j2).")
    return 1
  if args.offsets is None:
    args.offsets = list(CALIPER_OFFSETS if caliper else DEFAULT_OFFSETS)

  # Mirrored, because the two proximals close toward each other with the SAME
  # sign of scale (calib.py: f1j1 open +221 counts, f2j1 open -219). Driving
  # both by +N would swing the whole jaw sideways instead of opening it.
  moved = (0, 2) if caliper else (MOTORS.index(args.motor),)
  signs = (+1, -1) if caliper else (+1,)

  idx = moved[0]
  link = _import_link()(port=args.port)
  info = link.info_
  if info["num_motors"] != 4:
    print(f"!! controller reports {info['num_motors']} motors, need 4", file=sys.stderr)
    return 1

  obs = link.action(torque=False, cap_a=args.cap_a, apply=False)
  if obs["fault"]:
    print(f"!! fault {obs['fault']} before starting", file=sys.stderr)
    return 1
  start = [int(m["pos"]) for m in obs["motors"]]
  print(f"present counts: {dict(zip(MOTORS, start, strict=True))}")
  # The caliper's angle comes from GAP_TABLE, which is a pose model of the WHOLE
  # jaw -- so every motor it does not move has to be where the table says it is.
  # Tighter for the caliper, whose offsets are mirrored about the START pose:
  # a start that is not symmetric puts the two fingers at different angles, and
  # the table is written for a mirrored pair.
  tol = 10 if caliper else 30
  stray = [
    MOTORS[i] for i in range(4) if abs(start[i]) > tol and (i in moved or caliper)
  ]
  if stray:
    print(
      f"\n!! {', '.join(stray)} not near 0 counts. Drive them there first with\n"
      "   `pos <finger> <joint> 0` in gripper_ctl.py: 0 counts is the homing zero,\n"
      "   which is the pose GAP_TABLE and calib.COUNTS_AT_ZERO_RAD are written in."
    )
    if input("   continue anyway? [y/N] ").strip().lower() != "y":
      return 1

  if caliper:
    print(
      "\nMoving BOTH proximals in mirror; the distals hold at zero. Clear the jaws.\n"
      "Measure the OUTSIDE WIDTH across the two fingertips with the caliper's\n"
      "external jaws -- the widest point, same spot every time. NOT the inner gap:\n"
      "the inner faces are parallel only at 0 counts, and past that the narrowest\n"
      f"point slides up off the pad. Expect {NEUTRAL_OUTER_MM - args.width_offset_mm:.2f}"
      f" mm at 0 counts.\n"
    )
  else:
    print(
      f"\nMoving {args.motor} ({SIM_JOINT[idx]}) only; the other three hold position.\n"
      "Clear the jaws. The joint's axis must be HORIZONTAL and the gauge zeroed on\n"
      "a flat face of the link that this joint moves -- see the module docstring.\n"
    )
  input("Ready? press Enter: ")

  counts, radians = [], []
  try:
    for off in args.offsets:
      goals = list(start)
      for i, sign in zip(moved, signs, strict=True):
        goals[i] = start[i] + sign * off
      # Everything from the move to the operator's answer happens inside the
      # heartbeat: settling and reading an instrument both take far longer than
      # the controller's 500 ms watchdog.
      with _Hold(link, goals, args.cap_a) as hold:
        time.sleep(args.settle)
        obs = hold.last_obs
        if obs is None:
          print("!! no OBS from the controller", file=sys.stderr)
          return 1
        got = int(obs["motors"][idx]["pos"])
        if abs(got - goals[idx]) > 20:
          print(f"  [clamped or stalled] asked {goals[idx]}, reached {got}")
        unit = "OUTSIDE width in MM" if caliper else "gauge reading in DEGREES"
        raw = input(f"  at {got:+6d} counts -- {unit}: ").strip()
      if hold.fault:
        print(f"!! fault {hold.fault} at offset {off}", file=sys.stderr)
        return 1
      if not raw:
        print("  (skipped)")
        continue
      if caliper:
        try:
          rad = outer_to_rad(float(raw) + args.width_offset_mm)
        except ValueError as exc:
          print(f"  !! {exc}")
          continue
        print(f"     -> proximal {rad:+.4f} rad")
      else:
        rad = np.radians(float(raw))
      counts.append(got)
      radians.append(rad)
  except KeyboardInterrupt:
    print("\ninterrupted")
  finally:
    # Back to where it started, then torque off. Leaving a finger energised at
    # an arbitrary pose is how the next person finds it warm and jammed. Held
    # under the heartbeat too, or the watchdog drops torque part-way home and
    # the finger stops wherever it happens to be.
    try:
      with _Hold(link, list(start), args.cap_a):
        time.sleep(1.5)
      link.action(torque=False, cap_a=args.cap_a, apply=False)
    finally:
      link.close()

  if len(counts) < 3:
    print("\nnot enough points to fit; nothing written.")
    return 1

  span = float(np.max(radians) - np.min(radians))
  fit = fit_scale(np.array(counts), np.array(radians))
  _report(MOTORS[idx], SIM_JOINT[idx], fit, span)
  if caliper:
    print(
      "\nCALIPER: this is the AVERAGE of f1j1 and f2j1, because both moved. It is\n"
      "enough to say whether 651.9 is wrong and by roughly how much; it cannot\n"
      "say the two fingers differ, and calib.py's taught endpoints suggest they\n"
      "might. Settle that with --method gauge, one motor at a time."
    )
  else:
    print(
      "\nThis is ONE joint. calib.COUNTS_PER_RAD is a 4-tuple in the order\n"
      f"{MOTORS}, so repeat for the others -- the two fingers are separate\n"
      "mechanisms and calib.py already records that their taught endpoints differ."
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
