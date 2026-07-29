"""Move a Flexiv Rizon 4S to a compliance policy's HOME pose.

The pendant "home" parks the arm at Flexiv's default posture (joint2 ≈ −40°), but
each trained policy expects to start at the *sim* home its task reset to, because
that pose is what ``joint_pos_rel`` is measured against.  Starting anywhere else
feeds the network an out-of-distribution ``q_rel``.  Pick it with ``--policy``:

  ``reach``     ``[0, 0, 0, 1.57, 0, 0, 0]`` — ``BARE_HOME_KEYFRAME``, used by
                the Stage-1 reach tasks (impedance ``…-Bare-Full`` and torque
                ``…-TorqueSmoothBare``).
  ``tracking``  the retracted top-down pose ``_HOME_ARM_POSE`` in
                ``compliance_tracking/config/flexiv/env_cfg.py``, used by
                ``Mjlab-ComplianceTracking-StageA-TorqueBare-Flexiv``.

This utility drives the arm there via ``NRT_JOINT_POSITION`` with conservative
speed limits, after an explicit confirmation.

RDK v1.9 API (verified against the v1.9 header/example):
    SendJointPosition(positions, velocities, max_vel, max_acc)   # all rad-based
    positions in RADIANS; the robot's generator interpolates from the current q
    to the target respecting max_vel / max_acc, so sending a far target is safe.

Run from the dedicated deploy venv, by file path (imports only flexivrdk):

    .venv-deploy/bin/python src/mjlab/tasks/compliance/deploy/home_rizon.py \\
        --robot-sn Rizon4s-063501 --policy tracking

Keep a hand on the e-stop: this MOVES THE REAL ARM.
"""

from __future__ import annotations

import argparse
import math
import time

# Sim home poses, keyed by policy family. These mirror the ``q_default`` entries
# in deploy_rizon_torque.POLICIES (and deploy_rizon_route_b.Q_DEFAULT for reach).
HOMES: dict[str, list[float]] = {
  "reach": [0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0],
  "tracking": [-0.225705, 0.003626, 0.507476, 2.438249, 0.002759, 0.864276, -1.290931],
}
DEFAULT_POLICY = "reach"

# Conservative joint-motion limits for the homing move.
MAX_VEL = 0.3  # rad/s per joint
MAX_ACC = 0.6  # rad/s^2 per joint
SEND_HZ = 100.0
ARRIVE_TOL = 0.01  # rad, all joints within this of target => done
TIMEOUT_S = 60.0
REFUSE_DELTA = 1.75  # rad (~100 deg): refuse absurd single-joint moves


def _deg(xs: list[float]) -> list[float]:
  return [round(math.degrees(x), 1) for x in xs]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--robot-sn", required=True)
  ap.add_argument(
    "--policy",
    choices=sorted(HOMES),
    default=DEFAULT_POLICY,
    help=f"which policy's sim home to move to (default: {DEFAULT_POLICY})",
  )
  ap.add_argument("--max-vel", type=float, default=MAX_VEL, help="rad/s per joint")
  ap.add_argument("--max-acc", type=float, default=MAX_ACC, help="rad/s^2 per joint")
  ap.add_argument(
    "--max-delta",
    type=float,
    default=math.degrees(REFUSE_DELTA),
    metavar="DEG",
    help="refuse if any single joint would move more than this (default: "
    f"{math.degrees(REFUSE_DELTA):.0f} deg). Raise it deliberately — check the "
    "swept volume and any tool cabling first",
  )
  ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
  args = ap.parse_args()

  q_home = HOMES[args.policy]
  print(f"Homing to the '{args.policy}' policy's sim home pose.")

  import flexivrdk  # noqa: PLC0415  (only needed on the robot)

  mode = flexivrdk.Mode
  robot = flexivrdk.Robot(args.robot_sn)

  if robot.fault():
    print("Fault present; clearing...")
    robot.ClearFault()
  robot.Enable()
  print("Enabling; waiting until operational...")
  while not robot.operational():
    time.sleep(0.1)

  dof = robot.info().DoF
  if dof != len(q_home):
    raise SystemExit(f"robot DoF {dof} != len(q_home) {len(q_home)}; edit HOMES")

  q0 = list(robot.states().q)[:dof]
  target = q_home[:dof]
  deltas = [t - c for t, c in zip(target, q0, strict=False)]
  max_delta = max(abs(d) for d in deltas)

  print("\nCurrent q (deg):", _deg(q0))
  print("Target  q (deg):", _deg(target))
  print("Delta     (deg):", _deg(deltas))
  print(f"Largest joint move: {math.degrees(max_delta):.1f} deg")
  print(f"Limits: max_vel={args.max_vel} rad/s, max_acc={args.max_acc} rad/s^2")

  worst = max(range(dof), key=lambda i: abs(deltas[i]))
  if math.degrees(max_delta) > args.max_delta:
    raise SystemExit(
      f"Refusing: joint{worst + 1} would move {math.degrees(max_delta):.0f} deg "
      f"(> {args.max_delta:.0f}). Check the current pose / --policy, or raise the "
      f"limit with --max-delta {math.ceil(math.degrees(max_delta)):d} once you "
      "have confirmed the path is clear."
    )

  if not args.yes:
    resp = input("\nMove the REAL arm to HOME now? type 'go' to proceed: ").strip()
    if resp != "go":
      print("Aborted.")
      return

  robot.SwitchMode(mode.NRT_JOINT_POSITION)
  zeros = [0.0] * dof
  mv = [args.max_vel] * dof
  ma = [args.max_acc] * dof
  period = 1.0 / SEND_HZ

  print("Homing...")
  t_start = time.perf_counter()
  while True:
    if robot.fault():
      raise SystemExit("Fault occurred during homing; aborted.")
    robot.SendJointPosition(target, zeros, mv, ma)
    q = list(robot.states().q)[:dof]
    err = max(abs(t - c) for t, c in zip(target, q, strict=False))
    if err < ARRIVE_TOL:
      break
    if time.perf_counter() - t_start > TIMEOUT_S:
      raise SystemExit(f"Timeout after {TIMEOUT_S:.0f}s; joint error {err:.4f} rad.")
    time.sleep(period)

  time.sleep(0.5)  # settle
  qf = list(robot.states().q)[:dof]
  final_err = max(abs(t - c) for t, c in zip(target, qf, strict=False))
  print("Arrived. Final q (deg):", _deg(qf))
  print(f"Max joint error: {math.degrees(final_err):.3f} deg")


if __name__ == "__main__":
  main()
