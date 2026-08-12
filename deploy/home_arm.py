"""Put the robot at the grasp policy's home pose, before `deploy.run`.

Both halves of it. The policy emits one 11-vector spanning the arm and the four
finger joints, and every observation it sees is `joint_pos - DEFAULT_JOINT_POS`,
so "home" is a claim about all eleven. Starting anywhere else feeds the network
an out-of-distribution `q_rel` AND makes its first command a large step from
wherever that joint happens to be.

Each half is off by default for a different reason, and both bite:

  ARM   the pendant's "home" is Flexiv's default posture, not this one.
  HAND  the gripper's homing zero is the STRAIGHT / NEUTRAL pose, which is about
        0.27 rad more CLOSED than the open pose the policy calls home. Left
        there, the network's first act is to fling the fingers open.

  .venv-deploy/bin/python -m deploy.home_arm --robot-sn Rizon4s-063501 \\
      --gripper-port /dev/ttyACM0

Omit `--gripper-port` to move the arm only. `run.py` re-runs the hand ramp
immediately before its first inference regardless, so a skipped hand here is
recovered; the arm is not, and `run.py --require-home` simply refuses.

ARM FIRST, THEN THE HAND. The hand ramp is open-loop and timed -- no planner, no
collision check, and no way to stop partway -- so it should happen with the hand
in free space rather than wherever the last run left the arm parked. Homing the
fingers means opening the jaw to 86.9 mm of pad separation, which sweeps each
pad outward through whatever is beside it; down at the bench that is the foam,
the cube, and anything else on it.

The cost, which is real: the arm now traverses the workspace with the fingers
wherever they were left, possibly closed on a cube. That is what the previous
order was avoiding. It is the lesser risk -- the arm's move is planned and
interpolated by the robot's own generator under MAX_VEL/MAX_ACC and refuses
absurd deltas, while the finger ramp is none of those things -- but if the last
run ended with the jaw closed on the cube, the arm now carries it up. Open the
gripper with its own host CLI first when that matters.

The arm goes through NRT_JOINT_POSITION, not the impedance mode `deploy/arm.py`
runs the policy in: homing wants a stiff, planned, interpolated move to a known
pose, which is exactly what the policy loop must NOT have. The robot's own
generator interpolates from the current q under the vel/acc caps below, so
sending a far target is safe -- but the REFUSE_DELTA check still rejects absurd
single-joint moves, because a wrong sign convention looks exactly like a far one.

THIS MOVES THE REAL ARM. Keep a hand on the e-stop.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import calib

# Conservative limits for a homing move; the policy loop uses its own.
MAX_VEL = 0.3  # rad/s per joint
MAX_ACC = 0.6  # rad/s^2 per joint
SEND_HZ = 100.0
ARRIVE_TOL = 0.01  # rad, all joints within this of target => done
TIMEOUT_S = 60.0
REFUSE_DELTA = 1.75  # rad (~100 deg): refuse absurd single-joint moves

# The hand has no planner, so its ramp is timed here rather than tolerance-driven.
HAND_RAMP_S = 1.5
# 0.03 rad is the sim's own reset spread (`reset_joints_by_offset`, +-0.03 on
# every joint), so it is the widest start the policy has actually been trained on.
HAND_ARRIVE_TOL = 0.03


def _deg(x: np.ndarray) -> list[float]:
  # float(), not just round(): round(np.float64) stays a np.float64 and the list
  # prints as [np.float64(22.2), ...], which is what the operator has to read.
  return [round(float(v), 1) for v in np.degrees(x)]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--robot-sn", required=True, help="e.g. Rizon4s-063501")
  ap.add_argument(
    "--gripper-port",
    default=None,
    help="e.g. /dev/ttyACM0. Omit to home the arm only.",
  )
  ap.add_argument(
    "--yes", action="store_true", help="skip the confirmation prompt (scripted use)"
  )
  ap.add_argument(
    "--max-delta",
    type=float,
    default=REFUSE_DELTA,
    help=(
      "rad; refuse if any single joint would move further than this. The point "
      "is to make a zero-convention mismatch stop you, because that looks "
      "exactly like a legitimately far starting pose -- so raise it only after "
      "reading the per-joint delta table below and agreeing the path is clear."
    ),
  )
  args = ap.parse_args()

  import flexivrdk  # noqa: PLC0415  (only needed on the robot host)

  home = np.asarray(calib.DEFAULT_JOINT_POS, dtype=float)[calib.ARM_SLICE]
  hand_home = np.asarray(calib.DEFAULT_JOINT_POS, dtype=float)[calib.HAND_SLICE]

  # Built BEFORE the arm is touched, for the same reason `run.py` does it: the
  # hand is the piece that refuses (missing calibration, wrong motor count,
  # wrong control mode), and a refusal must land before anything energises the
  # arm or switches its control mode.
  hand = None
  if args.gripper_port is not None:
    from .hand import Hand  # noqa: PLC0415  (optional, needs the gripper repo)

    hand = Hand(port=args.gripper_port)

  robot = flexivrdk.Robot(args.robot_sn)
  # From here on the robot is about to be ENERGISED, so every path out -- a
  # failed check, a declined confirmation, EOF on the prompt, Ctrl-C -- has to
  # reach Stop(). An early `return` above this line is fine; below it is not.
  try:
    return _home(args, robot, hand, home, hand_home, flexivrdk)
  finally:
    try:
      robot.Stop()
      robot.SwitchMode(flexivrdk.Mode.IDLE)
    except Exception as exc:  # noqa: BLE001 -- best effort on the way out
      print(f"[arm] WARNING: stop failed: {exc}")
    if hand is not None:
      hand.close()


def _home(args, robot, hand, home, hand_home, flexivrdk) -> int:
  if robot.fault():
    print("fault present; clearing")
    robot.ClearFault()
  robot.Enable()
  print("enabling; waiting until operational...")
  while not robot.operational():
    time.sleep(0.1)

  info = robot.info()
  if int(info.DoF) != 7:
    print(f"!! robot reports DoF {info.DoF}, expected 7")
    return 1
  q_min = np.asarray(info.q_min, dtype=float)[:7]
  q_max = np.asarray(info.q_max, dtype=float)[:7]
  if np.any(home < q_min) or np.any(home > q_max):
    print(f"!! home {home.round(4)} is outside the robot's joint range")
    print(f"   q_min {q_min.round(3)}\n   q_max {q_max.round(3)}")
    return 1

  q_now = np.asarray(robot.states().q, dtype=float)[:7]
  delta = home - q_now
  print(f"\nmodel   {info.model_name}")
  print("ARM")
  print(f"  current {_deg(q_now)} deg")
  print(f"  home    {_deg(home)} deg")
  print(f"  delta   {_deg(delta)} deg")
  if hand is not None:
    qh_now, _ = hand.read()
    print("HAND")
    print(f"  current {_deg(qh_now)} deg")
    print(f"  home    {_deg(hand_home)} deg")
    print(f"  delta   {_deg(hand_home - qh_now)} deg")
  else:
    print("HAND    skipped (no --gripper-port); run.py will ramp it before its")
    print("        first inference, so this is recoverable")
  if np.abs(delta).max() > args.max_delta:
    j = int(np.abs(delta).argmax())
    print(
      f"\n!! joint{j + 1} would move {np.degrees(delta[j]):.1f} deg, over the "
      f"{np.degrees(args.max_delta):.0f} deg refusal threshold.\n"
      "   Either bring the arm closer on the pendant, or -- if the delta table "
      "above\n   is what you expect and the path is clear -- re-run with "
      f"--max-delta {np.abs(delta).max() + 0.05:.2f}.\n"
      "   Do NOT raise it just to get past this: a mismatch between RDK's joint "
      "zero\n   and the URDF the sim was built from looks exactly like a far "
      "starting pose,\n   and it would make every action offset wrong from the "
      "first step."
    )
    return 1

  if not args.yes:
    print("\nTHIS MOVES THE REAL ARM. Keep a hand on the e-stop.")
    try:
      answer = input("Type 'go' to proceed: ").strip()
    except EOFError:
      # No stdin (piped, or run from a script without --yes). Declining is the
      # only safe reading of "nobody is there to confirm".
      answer = ""
      print("\nno stdin to confirm on; treating as declined (pass --yes to skip)")
    if answer != "go":
      print("aborted")
      return 1

  # ARM FIRST. See the module docstring: the finger ramp is open-loop and
  # uninterruptible, so it wants the hand up in free space, not down among
  # whatever the last run left on the bench.
  rc = _move_arm(robot, home, flexivrdk)
  if rc != 0:
    if hand is not None:
      # Deliberately NOT homing the hand on a failed arm move. The reason to do
      # the arm first is that the fingers should open in free space, and a move
      # that faulted or timed out is precisely the case where we do not know
      # where the hand ended up. `run.py` re-ramps before its first inference
      # anyway, and `--require-home` will refuse this pose regardless.
      print("[hand] left as it was: the arm did not reach home.")
    return rc

  if hand is not None:
    print(f"\n[hand] ramping to the policy home over {HAND_RAMP_S:.1f}s")
    qh = hand.ramp_to(hand_home, secs=HAND_RAMP_S)
    off = float(np.abs(qh - hand_home).max())
    print(f"[hand] at {_deg(qh)} deg, residual {np.degrees(off):.2f} deg")
    if off > HAND_ARRIVE_TOL:
      print(
        f"[hand] WARNING: that is over {np.degrees(HAND_ARRIVE_TOL):.1f} deg "
        "from home. The sim resets every joint within +-1.7 deg of it, so the "
        "policy has not seen a start this far out."
      )
  return 0


def _move_arm(robot, home, flexivrdk) -> int:
  """Drive the seven arm joints to `home` and wait for arrival.

  NRT_JOINT_POSITION, so the robot's own generator interpolates from wherever
  the arm is under MAX_VEL/MAX_ACC -- the move is planned, unlike the hand's.
  """
  robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
  period = 1.0 / SEND_HZ
  deadline = time.time() + TIMEOUT_S
  while time.time() < deadline:
    robot.SendJointPosition(
      [float(v) for v in home], [0.0] * 7, [MAX_VEL] * 7, [MAX_ACC] * 7
    )
    q = np.asarray(robot.states().q, dtype=float)[:7]
    if np.abs(q - home).max() <= ARRIVE_TOL:
      print(f"\n[arm] arrived: {_deg(q)} deg")
      print(f"[arm] residual {np.abs(q - home).max() * 1000:.2f} mrad")
      return 0
    if robot.fault():
      print("\n!! robot faulted during the move")
      return 1
    time.sleep(period)

  q = np.asarray(robot.states().q, dtype=float)[:7]
  print(f"\n!! did not arrive within {TIMEOUT_S:.0f}s; stopped at {_deg(q)} deg")
  print(f"   worst joint is off by {np.degrees(np.abs(q - home).max()):.2f} deg")
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
