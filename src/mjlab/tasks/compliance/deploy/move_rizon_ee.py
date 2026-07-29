"""Jog a Flexiv Rizon 4S TCP to a Cartesian pose given as XYZ + roll/pitch/yaw.

Companion to ``home_rizon.py`` (which moves in *joint* space to the policy's HOME
posture).  This one works in *Cartesian* space: you type a position in metres and
an orientation in degrees, and the arm's TCP goes there via the built-in
``NRT_CARTESIAN_MOTION_FORCE`` generator with a stiff impedance, conservative
speed caps and a contact-wrench limit.

Frames and conventions
----------------------
* Position is in the robot **base frame** (the same frame the policy's
  ``goal_pos`` lives in), metres.
* Orientation is roll/pitch/yaw in **degrees**, ``R = Rz(yaw)·Ry(pitch)·Rx(roll)``
  (intrinsic Z-Y-X — the convention Flexiv's ``quat2EulerZYX`` reports), applied
  in the base frame.
* The commanded point is whatever the RDK considers the TCP.  If the tool is not
  configured in Flexiv Elements, pass ``--flange-tcp`` and the script treats your
  numbers as the *grasp_site* (flange + 0.05 m along link7's z), converting to
  and from the flange pose itself — matching ``deploy_rizon_route_b.py``.

Before the first move the script runs the ``ZeroFTSensor`` primitive: the robot
refuses to enter any force-control mode until the F/T sensor has been zeroed
(event ``[301004]``).  Keep the TCP free of contact while that happens, or the
contact load is baked in as the sensor's zero.  ``--skip-zero-ft`` opts out.

Absolute vs relative: ``--xyz`` / ``--rpy`` set an absolute target (anything you
omit keeps its current value); ``--dxyz`` / ``--drpy`` add a delta to the current
pose.  ``--rot-frame tool`` applies ``--drpy`` about the TCP's own axes instead of
the base axes.

Examples (run from the deploy venv, by file path — imports only numpy/flexivrdk):

    # Just print where the TCP is now (no motion).
    .venv-deploy/bin/python src/mjlab/tasks/compliance/deploy/move_rizon_ee.py \\
        --robot-sn Rizon4s-063501

    # Absolute: go to (0.50, 0.00, 0.45) m keeping the current orientation.
    ... move_rizon_ee.py --robot-sn Rizon4s-063501 --xyz 0.50 0.0 0.45

    # Absolute pose incl. orientation.
    ... --xyz 0.50 0.0 0.45 --rpy 180 0 0

    # Relative: up 5 cm, and tilt 10 deg about the tool's own x axis.
    ... --dxyz 0 0 0.05 --drpy 10 0 0 --rot-frame tool

    # Interactive jogging: type "0.5 0 0.45" or "d 0 0 0.05" at the prompt.
    ... --interactive

    # Offline check of the CLI/maths with no robot attached.
    .venv-deploy/bin/python .../move_rizon_ee.py --dry-run --dxyz 0 0 0.05

Keep a hand on the e-stop: this MOVES THE REAL ARM.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

# grasp_site (bare Rizon, no UMI gripper) = link7 flange + 0.05 m along link7 z.
BARE_TCP_OFFSET_LINK7 = np.array([0.0, 0.0, 0.05], dtype=np.float64)

# Motion caps for the NRT Cartesian generator (RDK defaults are 0.5 / 2.0).
MAX_LINEAR_VEL = 0.10  # m/s
MAX_LINEAR_ACC = 0.5  # m/s^2

# Positioning impedance: stiff enough to track accurately, soft enough to be safe
# on contact. Nominal K_x for this robot is [10000]*3 + [1500]*3.
K_TRANS = 2000.0  # N/m
K_ROT = 150.0  # Nm/rad
DAMPING_RATIO = 0.7  # Z_x, must stay inside Flexiv's [0.3, 0.8]
Z_X_RANGE = (0.3, 0.8)

# Safety clamp on the wrench the impedance controller may exert while moving.
MAX_CONTACT_WRENCH = [60.0, 60.0, 60.0, 15.0, 15.0, 15.0]

# Reachability guard (base frame, m). Generous around the trained workspace box
# [0.30,0.70]x[-0.20,0.20]x[0.35,0.65]; --force overrides.
WORKSPACE_BOX = ((0.20, 0.80), (-0.45, 0.45), (0.15, 0.85))
# Refuse a single command that would move this far; --force overrides.
MAX_STEP_M = 0.35
MAX_STEP_DEG = 90.0

ZERO_FT_TIMEOUT_S = 15.0  # ZeroFTSensor takes ~1-2 s in practice

SEND_HZ = 100.0
POS_TOL = 0.002  # m
ROT_TOL_DEG = 1.0
TIMEOUT_S = 60.0
SETTLE_S = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / Euler helpers (wxyz, intrinsic Z-Y-X == extrinsic X-Y-Z).
# ─────────────────────────────────────────────────────────────────────────────


def rpy_to_quat(rpy_rad: np.ndarray) -> np.ndarray:
  """Roll/pitch/yaw (rad) -> wxyz quaternion, R = Rz(yaw)·Ry(pitch)·Rx(roll)."""
  r, p, y = (float(a) * 0.5 for a in rpy_rad)
  cr, sr = math.cos(r), math.sin(r)
  cp, sp = math.cos(p), math.sin(p)
  cy, sy = math.cos(y), math.sin(y)
  return np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dtype=np.float64,
  )


def quat_to_rpy(q_wxyz: np.ndarray) -> np.ndarray:
  """wxyz quaternion -> roll/pitch/yaw (rad), inverse of :func:`rpy_to_quat`."""
  w, x, y, z = (float(c) for c in q_wxyz)
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product ``a ⊗ b`` (both wxyz)."""
  aw, ax, ay, az = (float(c) for c in a)
  bw, bx, by, bz = (float(c) for c in b)
  return np.array(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dtype=np.float64,
  )


def quat_rotate_vec(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate 3-vector ``v`` by wxyz quaternion ``q`` (``v' = R(q)·v``)."""
  w, x, y, z = (float(c) for c in q_wxyz)
  vx, vy, vz = (float(c) for c in v)
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  return np.array(
    [
      vx + w * tx + (y * tz - z * ty),
      vy + w * ty + (z * tx - x * tz),
      vz + w * tz + (x * ty - y * tx),
    ],
    dtype=np.float64,
  )


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
  """Shortest rotation angle between two wxyz quaternions, in degrees."""
  dot = abs(float(np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64))))
  return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _fmt_pose(pos: np.ndarray, quat: np.ndarray) -> str:
  rpy = np.degrees(quat_to_rpy(quat))
  return (
    f"xyz=[{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}] m  "
    f"rpy=[{rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}] deg"
  )


# ─────────────────────────────────────────────────────────────────────────────
# Robot I/O. RDK v1.9.x flat single-arm API (see deploy_rizon_route_b.py).
# ─────────────────────────────────────────────────────────────────────────────


def wait_primitive_done(robot, name: str, timeout: float = ZERO_FT_TIMEOUT_S) -> None:
  """Block until primitive ``name`` reports ``terminated``.

  ``robot.busy()`` is NOT a completion signal in ``NRT_PRIMITIVE_EXECUTION``:
  the mode holds the primitive resident, so ``busy()`` stays True indefinitely
  (measured: still True 14 s after ZeroFTSensor finished).  What actually flips
  is ``primitive_states()["terminated"]``, ~0.5 s in.
  """
  t0 = time.perf_counter()
  while True:
    st = robot.primitive_states()
    if st.get("primitiveName") == name and st.get("terminated") in (1, 1.0, "1", True):
      return
    if time.perf_counter() - t0 > timeout:
      raise SystemExit(f"{name} did not terminate in {timeout:.0f}s; states={st}")
    time.sleep(0.05)


class RizonJog:
  """Read the TCP pose and drive it to Cartesian targets."""

  def __init__(self, robot, mode, tcp_offset: np.ndarray, args) -> None:
    self.robot = robot
    self.Mode = mode
    self.offset = np.asarray(tcp_offset, np.float64)
    self.args = args
    self._mode_ready = False

  # --- state ---------------------------------------------------------------
  def read_pose(self) -> tuple[np.ndarray, np.ndarray]:
    """Current (position, wxyz quat) of the controlled point, base frame."""
    tcp = np.asarray(self.robot.states().tcp_pose, dtype=np.float64)
    pos, quat = tcp[0:3], tcp[3:7]
    # Shift the RDK TCP point out to grasp_site when the tool is not configured
    # in Elements (--flange-tcp); a no-op when offset is zero.
    return pos + quat_rotate_vec(quat, self.offset), quat

  def _to_rdk_point(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Inverse of the read_pose shift: grasp_site target -> RDK TCP target."""
    return pos - quat_rotate_vec(quat, self.offset)

  # --- motion --------------------------------------------------------------
  def zero_ft_sensor(self) -> None:
    """Run the ``ZeroFTSensor`` primitive.

    The robot REFUSES to enter any force-control mode (including
    ``NRT_CARTESIAN_MOTION_FORCE``) until the 6-axis F/T sensor has been zeroed
    at least once this session — otherwise ``SwitchMode`` raises with event
    ``[301004]``.  Zeroing measures the current wrench and treats it as the
    bias, so the TCP must be **free of contact** (and holding no unregistered
    payload) while it runs, or that load gets baked in as "zero".
    """
    print("Zeroing the F/T sensor — the TCP must be free of contact...")
    self.robot.SwitchMode(self.Mode.NRT_PRIMITIVE_EXECUTION)
    self.robot.ExecutePrimitive("ZeroFTSensor", {})
    wait_primitive_done(self.robot, "ZeroFTSensor")
    self.robot.Stop()  # release the primitive before switching modes
    print("F/T sensor zeroed.")

  def _enter_mode(self) -> None:
    if self._mode_ready:
      return
    if not self.args.skip_zero_ft:
      self.zero_ft_sensor()
    self.robot.SwitchMode(self.Mode.NRT_CARTESIAN_MOTION_FORCE)
    self.robot.SetMaxContactWrench(MAX_CONTACT_WRENCH)
    lo, hi = Z_X_RANGE
    z = min(max(self.args.damping, lo), hi)
    self.robot.SetCartesianImpedance(
      [self.args.k_trans] * 3 + [self.args.k_rot] * 3, [z] * 6
    )
    self._mode_ready = True

  def move_to(self, pos: np.ndarray, quat: np.ndarray) -> bool:
    """Drive the TCP to ``pos``/``quat``. Returns True once inside tolerance."""
    self._enter_mode()
    pose = np.concatenate([self._to_rdk_point(pos, quat), quat]).tolist()
    period = 1.0 / SEND_HZ
    t_start = time.perf_counter()
    while True:
      if self.robot.fault():
        raise SystemExit("Fault occurred during the move; aborted.")
      self.robot.SendCartesianMotionForce(
        pose,
        max_linear_vel=self.args.max_linear_vel,
        max_linear_acc=self.args.max_linear_acc,
      )
      cur_pos, cur_quat = self.read_pose()
      pos_err = float(np.linalg.norm(pos - cur_pos))
      rot_err = quat_angle_deg(quat, cur_quat)
      if pos_err < POS_TOL and rot_err < ROT_TOL_DEG:
        break
      if time.perf_counter() - t_start > self.args.timeout:
        print(
          f"Timeout after {self.args.timeout:.0f}s: "
          f"pos err {pos_err * 1000:.1f} mm, rot err {rot_err:.2f} deg. "
          "Target may be unreachable or blocked."
        )
        return False
      time.sleep(period)

    time.sleep(SETTLE_S)
    cur_pos, cur_quat = self.read_pose()
    print("Arrived.  ", _fmt_pose(cur_pos, cur_quat))
    print(
      f"  residual: {float(np.linalg.norm(pos - cur_pos)) * 1000:.1f} mm, "
      f"{quat_angle_deg(quat, cur_quat):.2f} deg"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Target resolution + safety.
# ─────────────────────────────────────────────────────────────────────────────


def resolve_target(
  cur_pos: np.ndarray,
  cur_quat: np.ndarray,
  xyz: list[float] | None,
  rpy_deg: list[float] | None,
  dxyz: list[float] | None,
  drpy_deg: list[float] | None,
  rot_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Fold the absolute/relative CLI options into one (pos, quat) target."""
  pos = np.asarray(xyz, np.float64) if xyz is not None else cur_pos.copy()
  quat = rpy_to_quat(np.radians(rpy_deg)) if rpy_deg is not None else cur_quat.copy()
  if dxyz is not None:
    pos = pos + np.asarray(dxyz, np.float64)
  if drpy_deg is not None:
    dq = rpy_to_quat(np.radians(drpy_deg))
    # base: rotate about the base axes; tool: about the TCP's own axes.
    quat = quat_mul(dq, quat) if rot_frame == "base" else quat_mul(quat, dq)
  return pos, quat / np.linalg.norm(quat)


def check_target(
  cur_pos: np.ndarray,
  cur_quat: np.ndarray,
  pos: np.ndarray,
  quat: np.ndarray,
  force: bool,
) -> bool:
  """Print the move summary; return False if a guard rejects it."""
  step_m = float(np.linalg.norm(pos - cur_pos))
  step_deg = quat_angle_deg(quat, cur_quat)
  print("\n  current:", _fmt_pose(cur_pos, cur_quat))
  print("  target :", _fmt_pose(pos, quat))
  print(f"  move   : {step_m * 1000:.1f} mm, {step_deg:.2f} deg")

  problems = []
  for axis, value, (lo, hi) in zip("xyz", pos, WORKSPACE_BOX, strict=True):
    if not lo <= value <= hi:
      problems.append(f"{axis}={value:.3f} outside [{lo}, {hi}] m")
  if step_m > MAX_STEP_M:
    problems.append(f"linear step {step_m:.3f} m > {MAX_STEP_M} m")
  if step_deg > MAX_STEP_DEG:
    problems.append(f"angular step {step_deg:.1f} deg > {MAX_STEP_DEG} deg")
  if not problems:
    return True
  for p in problems:
    print(f"  REJECTED: {p}")
  if force:
    print("  --force given; proceeding anyway.")
    return True
  print("  Not moving. Re-check the target, or pass --force if it is intended.")
  return False


def do_move(jog: RizonJog | None, cur, target, args) -> None:
  """Confirm and execute one move; ``jog=None`` is a dry run (no robot)."""
  cur_pos, cur_quat = cur
  pos, quat = target
  if not check_target(cur_pos, cur_quat, pos, quat, args.force):
    return
  if jog is None:
    print("  (dry run — no robot; nothing sent)")
    return
  if not args.yes and input("  Move the REAL arm? type 'go': ").strip() != "go":
    print("  Aborted.")
    return
  jog.move_to(pos, quat)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive jog loop.
# ─────────────────────────────────────────────────────────────────────────────

_INTERACTIVE_HELP = """
Commands (empty line or 'q' quits, 'p' prints the current pose):
  X Y Z                  absolute position (m), keep orientation
  X Y Z R P YAW          absolute position + orientation (deg)
  d DX DY DZ             relative move (m)
  d DX DY DZ DR DP DYAW  relative move + relative rotation (deg)
  frame base|tool        which axes relative rotations turn about
A pure rotation is "d 0 0 0 DR DP DYAW".
"""


_Vec3 = list[float] | None
# (xyz, rpy, dxyz, drpy) as the CLI options would have supplied them.
_Parsed = tuple[_Vec3, _Vec3, _Vec3, _Vec3]


def parse_interactive(line: str) -> _Parsed | None:
  """Parse one REPL line into (xyz, rpy, dxyz, drpy); None on a bad line."""
  parts = line.split()
  relative = parts[0].lower() in ("d", "r", "rel")
  if relative:
    parts = parts[1:]
  try:
    nums = [float(p) for p in parts]
  except ValueError:
    return None
  if len(nums) not in (3, 6):
    return None
  pos, rot = nums[:3], (nums[3:6] if len(nums) == 6 else None)
  return (None, None, pos, rot) if relative else (pos, rot, None, None)


def interactive_loop(jog: RizonJog | None, args) -> None:
  print(_INTERACTIVE_HELP)
  fake = (np.array([0.5, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0]))
  while True:
    try:
      line = input("ee> ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      return
    if not line or line.lower() in ("q", "quit", "exit"):
      return
    cur = jog.read_pose() if jog is not None else fake
    if line.lower() in ("p", "print"):
      print("  ", _fmt_pose(*cur))
      continue
    if line.lower() in ("h", "help", "?"):
      print(_INTERACTIVE_HELP)
      continue
    if line.lower().startswith("frame"):
      choice = line.split()[1].lower() if len(line.split()) > 1 else ""
      if choice in ("base", "tool"):
        args.rot_frame = choice
      print(f"  relative rotations are about the {args.rot_frame} axes")
      continue
    parsed = parse_interactive(line)
    if parsed is None:
      print("  Unrecognised. Type 'h' for help.")
      continue
    xyz, rpy, dxyz, drpy = parsed
    target = resolve_target(*cur, xyz, rpy, dxyz, drpy, args.rot_frame)
    do_move(jog, cur, target, args)


# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--robot-sn", default="Rizon4s-063501")
  ap.add_argument("--dry-run", action="store_true", help="no robot; check the maths")
  ap.add_argument(
    "--xyz", type=float, nargs=3, metavar=("X", "Y", "Z"), help="absolute position, m"
  )
  ap.add_argument(
    "--rpy",
    type=float,
    nargs=3,
    metavar=("R", "P", "Y"),
    help="absolute orientation, degrees (Z-Y-X, base frame)",
  )
  ap.add_argument(
    "--dxyz", type=float, nargs=3, metavar=("DX", "DY", "DZ"), help="relative move, m"
  )
  ap.add_argument(
    "--drpy",
    type=float,
    nargs=3,
    metavar=("DR", "DP", "DY"),
    help="relative rotation, degrees",
  )
  ap.add_argument(
    "--rot-frame",
    choices=("base", "tool"),
    default="base",
    help="frame the --drpy rotation is applied in (default: base)",
  )
  ap.add_argument("--interactive", action="store_true", help="REPL jogging mode")
  ap.add_argument(
    "--flange-tcp",
    action="store_true",
    help="RDK TCP is at the flange; treat poses as grasp_site (+0.05 m link7 z)",
  )
  ap.add_argument("--max-linear-vel", type=float, default=MAX_LINEAR_VEL, metavar="MPS")
  ap.add_argument(
    "--max-linear-acc", type=float, default=MAX_LINEAR_ACC, metavar="MPS2"
  )
  ap.add_argument("--k-trans", type=float, default=K_TRANS, metavar="N_PER_M")
  ap.add_argument("--k-rot", type=float, default=K_ROT, metavar="NM_PER_RAD")
  ap.add_argument("--damping", type=float, default=DAMPING_RATIO, metavar="ZETA")
  ap.add_argument("--timeout", type=float, default=TIMEOUT_S, metavar="S")
  ap.add_argument(
    "--skip-zero-ft",
    action="store_true",
    help="skip the ZeroFTSensor primitive (only if it was already zeroed this "
    "session; the robot rejects force-control modes otherwise)",
  )
  ap.add_argument("--force", action="store_true", help="override the safety guards")
  ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
  args = ap.parse_args()

  offset = BARE_TCP_OFFSET_LINK7 if args.flange_tcp else np.zeros(3)

  jog: RizonJog | None = None
  if not args.dry_run:
    import flexivrdk  # noqa: PLC0415  (only needed on the robot)

    robot = flexivrdk.Robot(args.robot_sn)
    if robot.fault():
      print("Fault present; clearing...")
      robot.ClearFault()
    robot.Enable()
    print("Enabling; waiting until operational...")
    while not robot.operational():
      time.sleep(0.1)
    jog = RizonJog(robot, flexivrdk.Mode, offset, args)

  if args.interactive:
    interactive_loop(jog, args)
    return

  cur = (
    jog.read_pose()
    if jog is not None
    else (np.array([0.5, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0]))
  )
  if all(v is None for v in (args.xyz, args.rpy, args.dxyz, args.drpy)):
    print("Current TCP pose:", _fmt_pose(*cur))
    print("(no target given — pass --xyz/--rpy/--dxyz/--drpy, or --interactive)")
    return

  target = resolve_target(
    *cur, args.xyz, args.rpy, args.dxyz, args.drpy, args.rot_frame
  )
  do_move(jog, cur, target, args)


if __name__ == "__main__":
  main()
