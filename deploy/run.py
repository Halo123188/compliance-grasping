"""The 50 Hz deployment loop: camera + joints -> ONNX -> arm and hand.

  uv run python -m deploy.run --onnx <policy.onnx> --gripper-port /dev/ttyACM0
  uv run python -m deploy.run --onnx <policy.onnx> --dry-run   # no hardware

`--dry-run` exercises the whole policy path with a synthetic depth frame and a
stationary arm, which is what to run first: it catches a metadata mismatch, a
wrong observation width and a missing calibration in a second, on a desk.

Two things this loop deliberately does NOT do, because getting them wrong is
worse than not having them:

  * It does not ramp into the first target. The policy's first action is
    computed from the arm's ACTUAL joint state, so the first command is
    `default + scale * a` regardless of where the arm currently is -- if the arm
    is not near the home pose when this starts, that first command is a large
    step. Move the arm to calib.DEFAULT_JOINT_POS[0:7] BEFORE starting, and
    --require-home enforces it.
  * It does not decide when the task is over. There is no termination in the
    trained env (`cfg.terminations = {}` under play), so the policy will hold a
    lifted cube indefinitely and has no notion of "done". Stop it yourself.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import calib
from .arm import ArmInterface, ReplayArm
from .policy import StudentPolicy


def build_hand(port: str | None, dry_run: bool):
  if dry_run:
    return None
  from .hand import Hand

  return Hand(port=port)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--onnx", required=True, help="exported student policy")
  ap.add_argument("--gripper-port", default=None, help="e.g. /dev/ttyACM0")
  ap.add_argument("--goal-height", type=float, default=calib.GOAL_HEIGHT_M)
  ap.add_argument("--steps", type=int, default=1000, help="50 Hz steps to run")
  ap.add_argument(
    "--dry-run", action="store_true", help="no camera, no gripper, no arm"
  )
  ap.add_argument(
    "--require-home",
    type=float,
    default=0.10,
    help="refuse to start if any arm joint is further than this (rad) from home",
  )
  ap.add_argument(
    "--max-action",
    type=float,
    default=calib.MAX_ABS_ACTION,
    help=(
      "abort if any raw network output exceeds this. Trained actions run to "
      "about |5|; the out-of-distribution blow-up this catches was |600|."
    ),
  )
  args = ap.parse_args()

  lo, hi = calib.GOAL_HEIGHT_RANGE
  if not lo <= args.goal_height <= hi:
    print(
      f"!! goal_height {args.goal_height} is outside the trained range "
      f"[{lo}, {hi}] (metres above the FLOOR, not above the bench)."
    )
    return 1

  policy = StudentPolicy(args.onnx)
  policy.reset()
  print(f"loaded {args.onnx}")

  home = np.asarray(calib.DEFAULT_JOINT_POS)
  arm: ArmInterface = ReplayArm(home)  # replace with the RDK implementation
  hand = build_hand(args.gripper_port, args.dry_run)

  q_arm, _ = arm.read()
  off = np.abs(q_arm - home[calib.ARM_SLICE])
  if off.max() > args.require_home:
    print(
      f"!! arm is {off.max():.3f} rad from the home pose (joint{off.argmax() + 1}). "
      "The policy commands absolute targets offset from home; starting far from "
      "it makes the first command a large step. Move to home first."
    )
    return 1

  if args.dry_run:
    depth_source = _synthetic_depth
  else:
    from .perception import RealSenseDepth

    camera = RealSenseDepth()
    depth_source = camera.read

  period = 1.0 / calib.CONTROL_HZ
  late = 0
  try:
    for i in range(args.steps):
      t0 = time.perf_counter()

      depth = depth_source()
      q_hand, v_hand = (
        (home[calib.HAND_SLICE], np.zeros(4)) if hand is None else hand.read()
      )
      q_arm, v_arm = arm.read()
      joint_pos = np.concatenate([q_arm, q_hand])
      joint_vel = np.concatenate([v_arm, v_hand])

      target = policy.step(joint_pos, joint_vel, depth, args.goal_height)

      # HARD GUARD, and it is not paranoia. The observation normalizer is baked
      # into the ONNX, so an out-of-distribution input does not degrade the
      # output gracefully -- it multiplies it. Measured on this checkpoint:
      # feeding goal_height = 0.0 instead of a value in [0.50, 0.60] takes
      # max|target| from 2.6 rad to 668 rad, a 250x blow-up, because
      # goal_height's training std is ~0.029 so an out-of-range value lands tens
      # of sigma out. The output is an ABSOLUTE joint target that goes straight
      # to the arm, so any input glitch -- a dropped camera frame, a stale
      # encoder read, a units mistake -- is a full-speed command into the bench.
      #
      # Judged on the RAW ACTION, not on target-minus-measured. A position servo
      # legitimately lags its target while reaching, so that distance measures
      # intent and aborting on it kills healthy runs -- which is exactly what the
      # first version of this guard did against a stationary test arm.
      # policy.act() separately clamps the target to calib.JOINT_LIMITS, so an
      # action that is merely large still cannot command past a hard stop.
      worst = float(np.abs(policy.last_action).max())
      if worst > args.max_action:
        j = int(np.abs(policy.last_action).argmax())
        print(
          f"!! step {i}: raw action for {calib.JOINT_NAMES[j]} is {worst:.1f}, over "
          f"--max-action {args.max_action}. Trained actions run to about |5|, so "
          "this is an out-of-distribution observation, not a big reach. Stopping."
        )
        break

      arm.set_targets(target[calib.ARM_SLICE])
      if hand is not None:
        hand.step(target[calib.HAND_SLICE])

      dt = time.perf_counter() - t0
      if dt > period:
        # Not cosmetic: the gripper drops torque after 500 ms without an ACTION,
        # so a loop that slips far enough disarms the hand mid-grasp.
        late += 1
        if late in (1, 10, 100):
          print(f"[warn] step {i} took {dt * 1000:.1f} ms > {period * 1000:.0f} ms")
      else:
        time.sleep(period - dt)
  except KeyboardInterrupt:
    print("\ninterrupted")
  finally:
    arm.stop()
    if hand is not None:
      hand.close()
    if not args.dry_run:
      camera.close()

  print(f"done. {late} late steps of {args.steps}.")
  return 0


def _synthetic_depth() -> np.ndarray:
  """A flat bench at 0.6 m with a box on it -- shape and range only, not a scene."""
  d = np.full(calib.DEPTH_HW, 0.6, dtype=np.float32)
  d[60:80, 70:95] = 0.52
  return (
    np.clip(d, calib.DEPTH_MIN_M, calib.DEPTH_CUTOFF_M) / calib.DEPTH_CUTOFF_M
  ).reshape(1, *calib.DEPTH_HW)


if __name__ == "__main__":
  raise SystemExit(main())
