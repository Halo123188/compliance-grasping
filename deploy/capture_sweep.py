"""Drive the arm through several poses and record a depth frame at each.

One pose cannot calibrate the camera. Fitting the arm's point cloud onto the
sim's render at the home pose says the real arm sits about 20 mm to the right of
where the model puts it, and three different faults predict exactly that
picture:

  the camera is 18 mm to one side of where CAD says
  the camera is yawed 3 deg about the vertical
  joint1's zero is off by 3 deg

At a single pose they are indistinguishable, and the plane fit that produced the
current extrinsic could never see any of them -- a plane has no yaw and no
in-plane position. Several poses do separate them, because each fault predicts a
DIFFERENT pattern of displacement across poses: a camera translation moves every
pose's cloud by the same vector, a camera yaw by an amount proportional to
range, and a joint1 offset by an amount proportional to the distance from the
BASE axis and in a direction that rotates as joint1 does.

  .venv-deploy/bin/python -m deploy.capture_sweep --robot-sn Rizon4s-063501 \\
      --out sweep.npz
  # then, on a host with mjlab:
  uv run python scripts/fit_camera_pose.py sweep.npz

THIS MOVES THE REAL ARM. Every pose is screened before anything moves -- pad
clearance over the table and both pads inside the frame -- and the script
refuses the whole sweep if any single pose fails, rather than stopping halfway
through with the arm somewhere unplanned. It returns to the starting pose on the
way out, including after a fault or Ctrl-C.

Nothing else may hold the camera while this runs; stop `deploy.live_view` first.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import calib, kinematics
from .check_camera import project

# Conservative: this is a measurement, not a cycle-time problem.
MAX_VEL = 0.25  # rad/s per joint
MAX_ACC = 0.5  # rad/s^2
SEND_HZ = 100.0
ARRIVE_TOL = 0.005  # rad on every joint
ARRIVE_TIMEOUT_S = 40.0
SETTLE_S = 1.0  # after arrival, before the first frame
FRAMES = 40  # averaged per pose

# The table with the foam removed, in the base frame. The pad ORIGIN is what
# kinematics.gripper_height returns, and the pad geom hangs below it, so this
# margin has to cover that as well as any tracking error.
TABLE_Z_BASE = -0.015
MIN_CLEARANCE = 0.045
# Both pads have to sit this far inside the 160x120 frame. A pad on the edge is
# half a silhouette, and a half silhouette biases the fit towards the middle.
MARGIN_PX = 12
# No single joint may move more than this between consecutive poses. The sweep
# is a sequence of small deliberate steps; anything larger means the pose table
# is not what the author thought it was.
MAX_STEP = 0.60

# Offsets from the pose the arm is standing in when the sweep starts, so the
# sweep is defined relative to wherever the operator parked it. Ordered to walk
# joint1 across its range once rather than oscillating, and the pure-lift poses
# are visited first while the claw is highest.
#                    j1     j2   j3     j4   j5   j6   j7
SWEEP: tuple[tuple[str, tuple[float, ...]], ...] = (
  ("start", (0, 0, 0, 0, 0, 0, 0)),
  ("up15", (0, +0.26, 0, 0, 0, 0, 0)),
  ("out17", (0, 0, 0, -0.30, 0, 0, 0)),
  ("out17_j1+16", (+0.28, 0, 0, -0.30, 0, 0, 0)),
  ("out17_j1+9", (+0.16, 0, 0, -0.30, 0, 0, 0)),
  ("out17_j1-9", (-0.16, 0, 0, -0.30, 0, 0, 0)),
  ("out17_j1-16", (-0.28, 0, 0, -0.30, 0, 0, 0)),
  ("wrist-20", (0, 0, 0, -0.30, 0, -0.35, 0)),
  ("start2", (0, 0, 0, 0, 0, 0, 0)),
)


def screen(q0: np.ndarray, hand: np.ndarray) -> tuple[list[str], list[str]]:
  """-> (report lines, reasons to refuse). Pure; touches nothing."""
  lines = [
    f"{'pose':>13}{'clear mm':>10}{'left u,v':>13}{'right u,v':>13}{'range m':>9}"
  ]
  bad: list[str] = []
  prev = q0
  for name, delta in SWEEP:
    q7 = q0 + np.asarray(delta, float)
    p = kinematics.forward(np.concatenate([q7, hand]))
    clear = min(p["left_pad"][2], p["right_pad"][2]) - TABLE_Z_BASE
    step = float(np.abs(q7 - prev).max())
    prev = q7
    try:
      lu, lv, lr = project(p["left_pad"])
      ru, rv, rr = project(p["right_pad"])
    except ValueError:
      bad.append(f"{name}: the claw is behind the camera")
      continue
    lines.append(
      f"{name:>13}{clear * 1000:10.0f}{lu:7.1f},{lv:5.1f}{ru:7.1f},{rv:5.1f}"
      f"{0.5 * (lr + rr):9.3f}"
    )
    if clear < MIN_CLEARANCE:
      bad.append(f"{name}: pad {clear * 1000:.0f} mm over the table")
    for label, (u, v) in (("left", (lu, lv)), ("right", (ru, rv))):
      if not (MARGIN_PX <= u <= 160 - MARGIN_PX and MARGIN_PX <= v <= 120 - MARGIN_PX):
        bad.append(f"{name}: {label} pad at ({u:.0f},{v:.0f}) is off the frame")
    if step > MAX_STEP:
      bad.append(f"{name}: a joint moves {step:.2f} rad from the previous pose")
  return lines, bad


def _grab(camera, n: int) -> np.ndarray:
  """Median over `n` frames, in metres, keeping 0 = no return.

  The median rather than the mean: a pixel on the edge of the sensor's
  confidence flickers between a plausible depth and 0, and a mean over that
  invents a depth halfway to nothing.
  """
  frames = []
  for _ in range(n):
    frames.append(camera.read().reshape(calib.DEPTH_HW) * calib.DEPTH_CUTOFF_M)
    time.sleep(1.0 / 60.0)
  a = np.stack(frames)
  out = np.zeros(calib.DEPTH_HW)
  valid = a > 0
  enough = valid.sum(0) >= n // 2
  masked = np.where(valid, a, np.nan)
  with np.errstate(invalid="ignore"):
    med = np.nanmedian(masked, axis=0)
  out[enough] = med[enough]
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--robot-sn", required=True)
  ap.add_argument("--out", default="sweep.npz")
  ap.add_argument("--camera-fps", type=int, default=90)
  ap.add_argument("--yes", action="store_true")
  ap.add_argument(
    "--dry-run",
    action="store_true",
    help="screen the poses and print the table; touch neither arm nor camera",
  )
  args = ap.parse_args()

  hand = np.asarray(calib.DEFAULT_JOINT_POS, float)[calib.HAND_SLICE]

  if args.dry_run:
    q0 = np.asarray(calib.DEFAULT_JOINT_POS, float)[calib.ARM_SLICE]
    print("[dry] screening against the POLICY HOME pose, not the real one\n")
    lines, bad = screen(q0, hand)
    print("\n".join(lines))
    print("\n" + ("\n".join(f"!! {b}" for b in bad) if bad else "all poses pass"))
    return 1 if bad else 0

  import flexivrdk  # noqa: PLC0415  (only on the robot host)

  from .perception import RealSenseDepth  # noqa: PLC0415  (needs pyrealsense2)

  robot = flexivrdk.Robot(args.robot_sn)
  if robot.fault():
    print("fault present; clearing")
    robot.ClearFault()
  robot.Enable()
  print("[arm] enabling; waiting until operational...")
  while not robot.operational():
    time.sleep(0.1)
  info = robot.info()
  q_min = np.asarray(info.q_min, float)[:7]
  q_max = np.asarray(info.q_max, float)[:7]
  q0 = np.asarray(robot.states().q, float)[:7]
  print(f"[arm] start pose {np.round(np.degrees(q0), 2).tolist()} deg")

  lines, bad = screen(q0, hand)
  for name, delta in SWEEP:
    q7 = q0 + np.asarray(delta, float)
    if np.any(q7 < q_min) or np.any(q7 > q_max):
      bad.append(f"{name}: outside the robot's own joint range")
  print("\n" + "\n".join(lines))
  if bad:
    print("\n" + "\n".join(f"!! {b}" for b in bad))
    print("\nrefusing the whole sweep. Move the arm to a better starting pose")
    print("(higher over the table, or more central in frame) and re-run.")
    return 1
  print("\nall poses clear the table and stay in frame")

  if not args.yes:
    print("\nTHIS MOVES THE REAL ARM through the poses above. Hand on the e-stop.")
    try:
      if input("Type 'go' to proceed: ").strip() != "go":
        print("aborted")
        return 1
    except EOFError:
      print("no stdin to confirm on; treating as declined (pass --yes)")
      return 1

  camera = RealSenseDepth(fps=args.camera_fps)
  robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
  names: list[str] = []
  qs: list[np.ndarray] = []
  depths: list[np.ndarray] = []
  try:
    for name, delta in SWEEP:
      target = q0 + np.asarray(delta, float)
      q = _goto(robot, target)
      if q is None:
        print(f"!! did not reach {name}; stopping the sweep here")
        break
      time.sleep(SETTLE_S)
      depth = _grab(camera, FRAMES)
      names.append(name)
      qs.append(q)
      depths.append(depth)
      print(
        f"[{name:>13}] q {np.round(np.degrees(q), 2).tolist()} deg   "
        f"valid {100 * (depth > 0).mean():.1f}%"
      )
  except KeyboardInterrupt:
    print("\ninterrupted")
  finally:
    # Back to where the operator left it, THEN release. Leaving the arm at
    # whichever pose the sweep died on is how the next run starts from a place
    # nobody screened.
    print("\n[arm] returning to the starting pose")
    try:
      _goto(robot, q0)
    except Exception as exc:  # noqa: BLE001 -- best effort on the way out
      print(f"[arm] WARNING: could not return: {exc}")
    try:
      robot.Stop()
      robot.SwitchMode(flexivrdk.Mode.IDLE)
    except Exception as exc:  # noqa: BLE001
      print(f"[arm] WARNING: stop failed: {exc}")
    camera.close()

  if not names:
    print("!! nothing captured")
    return 1
  np.savez_compressed(
    args.out,
    names=np.array(names),
    joint_pos=np.stack(qs),
    hand_pos=hand,
    depth_m=np.stack(depths).astype(np.float32),
    depth_cutoff_m=calib.DEPTH_CUTOFF_M,
  )
  print(f"\nwrote {args.out}: {len(names)} poses")
  print("next: uv run python scripts/fit_camera_pose.py " + args.out)
  return 0


def _goto(robot, target: np.ndarray) -> np.ndarray | None:
  """Interpolated move under the caps above. -> the measured q, or None."""
  t = [float(v) for v in target]
  deadline = time.time() + ARRIVE_TIMEOUT_S
  while time.time() < deadline:
    robot.SendJointPosition(t, [0.0] * 7, [MAX_VEL] * 7, [MAX_ACC] * 7)
    q = np.asarray(robot.states().q, float)[:7]
    if np.abs(q - target).max() <= ARRIVE_TOL:
      return q
    if robot.fault():
      print("!! robot faulted during the move")
      return None
    time.sleep(1.0 / SEND_HZ)
  return None


if __name__ == "__main__":
  raise SystemExit(main())
