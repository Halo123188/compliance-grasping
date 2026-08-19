"""The 50 Hz deployment loop: camera + joints -> ONNX -> arm and hand.

  uv run python -m deploy.run --onnx <policy.onnx> --gripper-port /dev/ttyACM0
  uv run python -m deploy.run --onnx <policy.onnx> --dry-run   # no hardware
  uv run python -m deploy.run --onnx <policy.onnx> --no-hand --speed 0.5
      # arm only, half the (already slow) bring-up limits: does the arm reach?

`--dry-run` exercises the whole policy path with a synthetic depth frame and a
stationary arm, which is what to run first: it catches a metadata mismatch, a
wrong observation width and a missing calibration in a second, on a desk.

Two things this loop deliberately does NOT do, because getting them wrong is
worse than not having them:

  * It does not ramp into the first target, beyond whatever the profile's slew
    limit gives. The policy's first action is computed from the arm's ACTUAL joint
    state, and without a slew limit the first command is `default + scale * a`
    regardless of where the arm currently is -- if the arm is not near the home
    pose when this starts, that first command is a large step. With a limit it
    is a bounded one, which is softer but is NOT a substitute: the command still
    heads somewhere the policy chose from an out-of-distribution pose. Move the
    arm to calib.DEFAULT_JOINT_POS[0:7] BEFORE starting; --require-home enforces
    it.
  * It does not decide when the task is over. There is no termination in the
    trained env (`cfg.terminations = {}` under play), so the policy will hold a
    lifted cube indefinitely and has no notion of "done". Stop it yourself.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import calib, kinematics, profiles
from .arm import ArmInterface, ReplayArm
from .policy import StudentPolicy


def build_hand(port: str | None, dry_run: bool, no_hand: bool, cap_a: float):
  """The real gripper, or None for a run that deliberately has no hand.

  `--no-hand` is not the same thing as a broken gripper. The loop still needs
  eleven joint positions and eleven velocities every step, so with no hand it
  reports the fingers as sitting exactly at their home pose with zero velocity
  -- i.e. a perfect finger servo that never leaves home. That is a lie the
  policy cannot detect, and it is the RIGHT lie for a reach test: the finger
  block of `joint_pos - default` stays at zero, which is what the network sees
  at the start of every sim episode, so the arm behaves as it would before the
  fingers have moved. It is NOT a grasp test -- the policy will command the
  fingers closed, nothing will close, and it will keep reaching for a cube it
  can never pick up.
  """
  if dry_run or no_hand:
    return None
  from .hand import Hand

  return Hand(port=port, cap_a=cap_a)


def build_arm(
  robot_sn: str | None,
  dry_run: bool,
  max_vel: float,
  max_acc: float,
  max_offset: list[float] | None,
) -> ArmInterface:
  """The real arm when a serial number is given, the stationary stub otherwise.

  `--dry-run` deliberately ignores `--robot-sn`: the whole point of the dry run
  is that nothing moves, and a flag combination that quietly moved a 7-DoF arm
  would be the worst possible surprise.
  """
  home = np.asarray(calib.DEFAULT_JOINT_POS)
  if dry_run or robot_sn is None:
    return ReplayArm(home)
  from .arm import FlexivArm

  return FlexivArm(
    robot_sn, max_vel=max_vel, max_acc=max_acc, max_offset_rad=max_offset
  )


class _Record:
  """Every step's observation, action and command, for reading back afterwards.

  Preallocated for the same reason the stage timers are: the instrumentation
  must not become the cost it is measuring.

  DEPTH IS float16, AND uint8 WAS A MISTAKE. The first version stored the
  observation as uint8, reasoning that 256 levels over the 3 m cutoff is 12 mm
  and that "what was in frame" only needs a picture. But the quantity this
  policy is most sensitive to is depth bias along the optical axis, where the
  measured knee sits between 1.4 mm (free) and 2.6 mm (-26 points of success)
  -- an order of magnitude BELOW that quantum. So the recording could show
  that nothing was grossly wrong and could not show the one thing worth
  looking for. float16 holds the normalised observation to about 0.06 mm at
  bench range, at 38 KB a frame; 1000 steps is 38 MB before compression.
  """

  def __init__(self, steps: int, depth_hw: tuple[int, int] = calib.DEPTH_HW):
    self.depth_hw = depth_hw
    self.pos = np.zeros((steps, 11), np.float32)
    self.vel = np.zeros((steps, 11), np.float32)
    self.act = np.zeros((steps, 11), np.float32)
    self.tgt = np.zeros((steps, 11), np.float32)
    self.img = np.zeros((steps, *depth_hw), np.float16)
    # The four finger motors' current, from the same OBS packet as the pose and
    # the velocity beside it, so the three are one consistent reading of one
    # instant rather than three polls of a moving hand.
    self.cur = np.zeros((steps, 4), np.float32)

  def put(self, i, joint_pos, joint_vel, action, target, depth, current_a) -> None:
    self.pos[i], self.vel[i] = joint_pos, joint_vel
    self.act[i], self.tgt[i] = action, target
    self.img[i] = depth.reshape(self.depth_hw)  # already [0, 1]
    self.cur[i] = current_a

  def save(
    self,
    path: str,
    n: int,
    goal_height: float,
    policy: StudentPolicy,
    cap_a: float,
  ) -> None:
    np.savez_compressed(
      path,
      joint_pos=self.pos[:n],
      joint_vel=self.vel[:n],
      action=self.act[:n],
      target=self.tgt[:n],
      # Normalised exactly as the policy saw it: clamp(m, 0.01, 3.0) / 3.0,
      # with 0.0 meaning no return. Multiply by DEPTH_CUTOFF_M for metres.
      depth=self.img[:n],
      depth_cutoff_m=np.float32(calib.DEPTH_CUTOFF_M),
      goal_height=np.float32(goal_height),
      default_joint_pos=np.asarray(calib.DEFAULT_JOINT_POS, np.float32),
      # From the POLICY, not from a module constant: a recording is read back
      # long after the run, and `default + scale * action` reconstructs nothing
      # if the scale was another checkpoint's.
      action_scale=policy.scale,
      profile=str(policy.profile.name),
      cube_half_extent=np.float32(policy.cube_half_extent),
      # Signed, amperes, in JOINT_NAMES[7:11] order. Zero for the whole run on
      # `--no-hand`, where there is no gripper to report one.
      finger_current_a=self.cur[:n],
      # The cap those currents are to be read against. It is an operator flag,
      # so a recording that did not carry it could not be compared with the
      # next one.
      grip_cap_a=np.float32(cap_a),
    )


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--onnx", required=True, help="exported student policy")
  ap.add_argument("--gripper-port", default=None, help="e.g. /dev/ttyACM0")
  ap.add_argument(
    "--no-hand",
    action="store_true",
    help=(
      "run the ARM ONLY: no gripper is opened, the finger half of every action "
      "is discarded, and the fingers are reported to the policy as parked at "
      "their home pose. For checking that the arm reaches the grasp pose "
      "before the hand is trusted. Cannot grasp anything -- see build_hand()."
    ),
  )
  ap.add_argument(
    "--robot-sn",
    default=None,
    help=(
      "Flexiv serial, e.g. Rizon4s-063501. Without it the arm is the stationary "
      "stub even outside --dry-run, so the arm cannot move by accident."
    ),
  )
  ap.add_argument(
    "--grip-cap",
    type=float,
    default=calib.GRIP_CURRENT_CAP_A,
    help=(
      "amps; the current cap IS the grip force. The controller's hard ceiling "
      "is 1.5 A. Raising this is the one knob for a weak grasp, but four motors "
      "pulling hard is exactly what browns out the 5 V rail (gripper "
      "NEXT_STEPS.md calls that the project blocker), and a motor that resets "
      "under load faults as MOTOR_REBOOTED having already dropped its load."
    ),
  )
  # Bring-up defaults, deliberately WELL below what the policy trained under. In
  # sim this checkpoint's arm runs |dq| p99 2.3 rad/s and peaks at 5.3, so
  # 0.25 rad/s is about a tenth speed. That is a real change to the closed loop,
  # not just a comfort setting, and it cuts BOTH ways:
  #
  #   * a slower arm reaches a different state by the next inference than the
  #     policy expects, so behaviour drifts from sim;
  #   * and because the arm now lags its target, |target - measured| grows, so
  #     --max-jump fires more readily. That is the guard working, not a bug: it
  #     means the policy is asking for motion this speed cap cannot deliver.
  #
  # Use these to watch and debug; restore --arm-max-vel 2.5 --arm-max-acc 6.0
  # before judging a success rate.
  ap.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help=(
      "ONE knob for 'too fast'. Multiplies --arm-max-vel, --arm-max-acc and "
      "--arm-max-offset together, so 0.5 halves the top speed, the "
      "aggressiveness of getting there, and the torque ceiling in one go. This "
      "is the flag to turn down on the bench; the three below are the base "
      "limits it scales."
    ),
  )
  ap.add_argument(
    "--arm-max-vel",
    type=float,
    default=0.25,
    help=(
      "rad/s handed to SendJointPosition; the robot's generator enforces it, so "
      "this is the honest top-speed limit. Sim reference: p99 2.3, peak 5.3."
    ),
  )
  ap.add_argument(
    "--arm-max-acc",
    type=float,
    default=0.6,
    help=(
      "rad/s^2. This, not the speed cap, is usually what makes the motion feel "
      "violent -- it sets how fast the arm reaches its top speed."
    ),
  )
  ap.add_argument(
    "--arm-ramp-s",
    type=float,
    default=2.0,
    help=(
      "seconds to scale the arm's motion limits up from a standstill. The "
      "policy does NOT ramp into its first target -- action 0 is computed from "
      "the measured pose and can sit a radian or more away -- so without this "
      "the arm's first move is also its hardest. 0 disables."
    ),
  )
  # NOT a safety knob, and it was wrong to treat it as one. It caps
  # |commanded - measured|, so it caps the impedance controller's torque at
  # K_q * offset -- and the TRAINING environment was torque-limited too, at
  # `actfrcrange` = [123, 123, 64, 64, 39, 39, 39] Nm, which is exactly the
  # robot's own tau_max (verified on the compiled model; the old claim that the
  # sim was unlimited read `actuator_forcerange`, the wrong field). So the
  # offset that reproduces training is tau_max / K_q per joint, and the default
  # is now to read it off the robot. A scalar 0.08 was between 19% and 48% of
  # that depending on the joint -- a rate limit inside the closed loop, which
  # produces a limit cycle rather than a slower version of the same motion.
  ap.add_argument(
    "--arm-max-offset",
    type=float,
    nargs="*",
    default=None,
    metavar="RAD",
    help=(
      "rad, one value or seven. Omit for tau_max/K_q read from the robot, "
      "which is the training environment's own saturation "
      "(~0.43 0.18 0.29 0.17 0.16 0.17 0.21). Pass a value only to "
      "deliberately throttle, and expect different behaviour when you do."
    ),
  )
  ap.add_argument("--goal-height", type=float, default=calib.GOAL_HEIGHT_M)
  ap.add_argument(
    "--profile",
    default=None,
    choices=sorted(profiles.PROFILES),
    help=(
      "which checkpoint family this is. Omit to select it from the run name and "
      "the graph (deploy/profiles.py); pass it for a renamed export, where the "
      "shape alone cannot say whether the slew limit is on."
    ),
  )
  ap.add_argument(
    "--cube-mm",
    type=float,
    default=calib.CUBE_EDGE_MM,
    help=(
      "the cube's EDGE in mm, measured with a caliper. Round 2's checkpoints "
      "OBSERVE this (as a half-extent, in metres) and refuse a value outside "
      "the size range they trained on; the older ones are size-blind and only "
      "warn, since a cube they never saw is still a cube they cannot grasp."
    ),
  )
  ap.add_argument(
    "--ema-tau",
    type=float,
    default=None,
    help=(
      "override the profile's command low-pass constant, seconds; 0 turns it "
      "off. Unlike the slew limit this is a DEPLOYMENT choice -- the action "
      "term is outside the network, so the lag can be changed on a finished "
      "checkpoint and the -SlowEma sweep priced doing so at roughly nothing "
      "(see profiles.Profile.ema_tau). Raise it if the arm still snaps."
    ),
  )
  ap.add_argument("--steps", type=int, default=1000, help="50 Hz steps to run")
  ap.add_argument(
    "--dry-run", action="store_true", help="no camera, no gripper, no arm"
  )
  # The camera is a control-loop parameter, not a sensor setting. At 30 fps
  # `wait_for_frames` blocks for 33.3 ms of a 20 ms budget; measured on the
  # bench that produced 44 ms per step (23 Hz) with every step late, which puts
  # the policy well outside the 50 Hz it was trained at.
  ap.add_argument(
    "--camera-fps",
    type=int,
    default=90,
    help=(
      "D435 depth rate. The 848x480 profile does up to 90. If the stream will "
      "not start, lower THIS -- never the resolution, which is what the "
      "student's field of view comes from."
    ),
  )
  ap.add_argument(
    "--camera-preset",
    default=None,
    help=(
      "D400 visual preset (high_density / high_accuracy / default / hand / "
      "medium_density). This sets how much of the frame comes back as 'no "
      "return', which is the single biggest measured sim-to-real gap in the "
      "observation -- see deploy/perception.py."
    ),
  )
  ap.add_argument(
    "--camera-blocking",
    action="store_true",
    help=(
      "read the camera on the control thread instead of a reader thread. "
      "Slower by construction (the loop aliases to the sensor's phase); exists "
      "to attribute a timing regression to the thread rather than assume it."
    ),
  )
  ap.add_argument(
    "--record",
    default=None,
    metavar="PATH.npz",
    help=(
      "log every step -- measured joints, raw action, commanded target, the "
      "depth frame and the four finger motors' current -- and write it on the "
      "way out, including after an abort. "
      "Depth is float16, about 38 MB per 1000 steps, which resolves the "
      "millimetre-scale depth bias this policy is actually sensitive to; the "
      "earlier uint8 quantised to 12 mm and hid exactly that."
    ),
  )
  ap.add_argument(
    "--print-every",
    type=int,
    default=25,
    help=(
      "steps between progress lines (25 = twice a second). Prints the measured "
      "arm pose, the TCP position when RDK reports one, and the worst "
      "|target - measured| -- which is the number that tells you whether the "
      "arm is following the policy or being dragged behind it. 0 disables."
    ),
  )
  ap.add_argument(
    "--require-home",
    type=float,
    default=0.10,
    help="refuse to start if any arm joint is further than this (rad) from home",
  )
  # The guard that was missing when the arm was driven into the bench. Joint
  # space cannot express "do not go below the table"; this can. Default is the
  # foam top exactly. The policy legitimately touches the foam -- the trained
  # env has a `fingertip_table_contact` penalty, not a prohibition -- so a floor
  # ABOVE the surface would abort every normal grasp. It is compared against the
  # pad frame ORIGIN, which sits above the pad's own lowest point, so the real
  # clearance is smaller than the number suggests: erring low costs a run and
  # erring high costs the bench.
  ap.add_argument(
    "--floor-z",
    type=float,
    default=calib.WORK_SURFACE_Z - calib.ARM_BASE_Z,
    help=(
      "metres above the arm's base plate; stop if a pad reaches it. Default "
      "is the foam top itself. Measured on the v11 family, which grasps 42.5-"
      "57.5 mm cubes, the pads never go below 0.057 -- 22 mm of clearance -- so "
      "a floor at the surface cannot fire on trained behaviour. THAT MARGIN IS "
      "NOT THE ROUND-3 MARGIN: those arms grasp objects down to 15 mm tall, "
      "whose mid-height is ~17 mm lower, which spends most of the 22 mm. See "
      "--floor-lookahead. Negative disables it."
    ),
  )
  ap.add_argument(
    "--floor-lookahead",
    type=float,
    default=0.15,
    help=(
      "seconds of descent to subtract from the measured height before "
      "comparing against --floor-z. 0 makes the guard purely reactive, which "
      "is too late: at 0.4 m/s the pads cross the last 20 mm in one step. THIS "
      "IS THE KNOB FOR SMALL OBJECTS, not --floor-z: on a 15 mm object the pads "
      "must reach a few mm above the foam, and 0.15 s of a 0.2 m/s descent is "
      "30 mm of margin the geometry no longer has. Lower this if the guard "
      "aborts a grasp that was going fine; do not lower the floor, which is "
      "where the bench is."
    ),
  )
  ap.add_argument(
    "--max-jump",
    type=float,
    default=1.2,
    help=(
      "abort if a commanded ARM target is further than this (rad) from the "
      "measured joint. Set from the checkpoint's own behaviour in sim, where "
      "arm |target - measured| runs p99 0.39 / p99.9 0.67 / max 1.40: 1.2 fires "
      "on 0.05%% of sim steps, 0.6 on 1.3%%. The fingers are excluded -- see the "
      "comment at the check."
    ),
  )
  ap.add_argument(
    "--max-action",
    type=float,
    default=calib.MAX_ABS_ACTION,
    help=(
      "abort if any raw network output exceeds this. Trained actions run to "
      "about |5|; the out-of-distribution blow-up this catches was |600|. This "
      "is the guard that still works under the profile's slew limit, which by "
      "construction hides a blown-up action from --max-jump."
    ),
  )
  args = ap.parse_args()

  # Forcing the choice rather than defaulting either way. `--gripper-port` alone
  # is ambiguous in the direction that matters: on Linux the port cannot be
  # auto-detected (GripperLink's probe is macOS-only), so a forgotten flag would
  # otherwise mean "silently run with no hand" -- which is a legitimate mode
  # here, and therefore exactly the mistake that must not be silent.
  if not args.dry_run and (args.gripper_port is None) != args.no_hand:
    print(
      "!! pass either --gripper-port <port> to use the gripper, or --no-hand to "
      "run the arm alone. Passing both, or neither, is ambiguous."
    )
    return 1

  if args.speed <= 0.0 or args.speed > 1.0:
    print(f"!! --speed {args.speed} must be in (0, 1]; it only ever slows down.")
    return 1

  if args.floor_z is not None and args.floor_z < 0:
    print("!! --floor-z disabled; the arm has no lower bound in Cartesian space.")
    args.floor_z = None

  if args.arm_max_offset is not None and len(args.arm_max_offset) not in (1, 7):
    print(
      f"!! --arm-max-offset takes one value or seven, got "
      f"{len(args.arm_max_offset)}. Omit it entirely for tau_max/K_q."
    )
    return 1

  lo, hi = calib.GOAL_HEIGHT_RANGE
  if not lo <= args.goal_height <= hi:
    print(
      f"!! goal_height {args.goal_height} is outside the trained range "
      f"[{lo}, {hi}] (metres above the FLOOR, not above the bench)."
    )
    return 1

  # onnxruntime's own missing-file error names the path and nothing else, and
  # the path is nearly always the mistake: the exports live beside the
  # checkpoints rather than in the repo, so a bare filename run from here finds
  # nothing. Cheap to say so, and it costs one stat.
  if not Path(args.onnx).expanduser().is_file():
    print(f"!! no such policy: {args.onnx}")
    siblings = sorted(Path("~/yiboc").expanduser().glob("*.onnx"))
    if siblings:
      print("   exports found in ~/yiboc:")
      for path in siblings:
        print(f"     {path}")
    return 1

  # The profile is selected from the run name and the graph, and it decides the
  # observation layout, the action scale, the finger travel and the slew limit.
  # Constructed before anything touches hardware so a mismatch -- an unknown
  # export, a cube outside the trained range -- raises on a desk.
  try:
    policy = StudentPolicy(
      str(Path(args.onnx).expanduser()),
      profile=None if args.profile is None else profiles.by_name(args.profile),
      cube_half_extent=args.cube_mm / 2000.0,
    )
  except ValueError as e:
    print(f"!! {e}")
    return 1
  # Before reset(), so the smoother is built with the constant it will run with.
  if args.ema_tau is not None:
    try:
      policy.set_ema_tau(args.ema_tau)
    except ValueError as e:
      print(f"!! {e}")
      return 1
  policy.reset()
  print(f"loaded {args.onnx}")
  print(policy.describe())
  # The slew limit is printed rather than checked, because it CANNOT be checked:
  # the ONNX metadata does not carry the training env's action term, so this
  # pairing rests on the profile having named the right run.
  if policy.rate_limit is None:
    print(
      "[policy] command slew limit: OFF. Correct only for a checkpoint trained "
      "without one -- publishing a limited policy's raw request is a full-speed "
      "move into the bench on the first step."
    )
  # A size-blind checkpoint cannot refuse an unfamiliar cube, so say so. The
  # grasp is the part that fails: the trained closing depth is the one that
  # suits the sizes below, and nothing in the policy adapts it.
  lo, hi = policy.profile.cube_edge_mm
  if not policy.profile.observes_cube_size and not lo <= args.cube_mm <= hi:
    print(
      f"[warn] --cube-mm {args.cube_mm:.1f} is outside {policy.profile.name}'s "
      f"trained {lo:.1f}-{hi:.1f} mm, and this checkpoint cannot see the size. "
      "It will close to the depth it always closes to."
    )

  home = np.asarray(calib.DEFAULT_JOINT_POS)

  # ORDER AND SCOPE BOTH MATTER, and getting either wrong strands an ENERGISED
  # arm. Build the hand first: it is the piece that refuses (missing calibration),
  # and a refusal must happen before anything switches the robot into a control
  # mode. Build the arm last, already inside the try, so that every exit path
  # from here on -- a raise, the home check failing, the camera failing to open,
  # Ctrl-C -- reaches `arm.stop()`. `arm` starts as None because the `finally`
  # can now run before it is assigned.
  arm: ArmInterface | None = None
  hand = None
  camera = None
  period = 1.0 / calib.CONTROL_HZ
  late = 0
  # Four stages, timed separately, because "the loop is late" is not actionable
  # and "the camera costs 33 ms" is. Preallocated rather than appended so the
  # instrumentation cannot itself become the cost being measured, and declared
  # out here so the summary can be printed after an abort or a Ctrl-C too.
  stage_names = ("camera", "sensors", "policy", "command")
  stages = np.zeros((4, args.steps))
  loop_dt = np.zeros(args.steps)
  # Worst joint's limiter overdrive per step. The one number that says whether
  # the profile's slew limit is the one this checkpoint trained under: a policy
  # that trained against this cap requests a little more than it, a policy that
  # trained against a looser one (or none) requests far more, every step.
  overdrive = np.zeros(args.steps)
  # The four finger currents per step, kept whether or not `--record` is on,
  # because the summary they feed is the one that says which KIND of failed
  # grasp this was and it should not need a flag set in advance.
  finger_cur = np.zeros((args.steps, 4), np.float32)
  h_prev: float | None = None  # last gripper height, for the descent rate
  done = 0  # steps that completed, i.e. reached the command
  seen = 0  # steps whose OBSERVATION was recorded, which includes the abort
  rec = _Record(args.steps, policy.depth_hw) if args.record else None
  try:
    hand = build_hand(args.gripper_port, args.dry_run, args.no_hand, args.grip_cap)
    max_vel = args.arm_max_vel * args.speed
    max_acc = args.arm_max_acc * args.speed
    # `--speed` scales the offset too, so it still throttles torque authority as
    # well as speed -- but at --speed 1 with no --arm-max-offset the arm now
    # gets exactly the training limit instead of an invented fraction of it.
    # None stays None so FlexivArm can derive it from the robot's tau_max.
    max_offset = (
      None
      if args.arm_max_offset is None
      else [v * args.speed for v in args.arm_max_offset]
    )
    print(
      f"[arm] limits: {max_vel:.2f} rad/s, {max_acc:.2f} rad/s^2  "
      f"(--speed {args.speed:g}). Sim |dq| p99 2.3 rad/s, peak 5.3."
    )
    if args.no_hand and not args.dry_run:
      print(
        "[hand] --no-hand: ARM ONLY. The fingers are reported to the policy as "
        "parked at home, and the finger half of every action is discarded. The "
        "policy will still reach for the cube and 'close' on it; nothing will. "
        "Judge the REACH here, not the grasp."
      )
    arm = build_arm(
      args.robot_sn,
      args.dry_run,
      max_vel,
      max_acc,
      max_offset,
    )

    q_arm, _ = arm.read()
    off = np.abs(q_arm - home[calib.ARM_SLICE])
    if off.max() > args.require_home:
      print(
        f"!! arm is {off.max():.3f} rad from the home pose (joint{off.argmax() + 1}). "
        "The policy commands absolute targets offset from home; starting far from "
        "it makes the first command a large step. Run `python -m deploy.home_arm` "
        "first."
      )
      return 1

    if hand is not None:
      # Kept even though `home_arm` does it too: this is the last thing before
      # the first inference, and the policy's first action is computed from the
      # measured state. If the hand is already there this is a no-op ramp.
      want = home[calib.HAND_SLICE]
      got = hand.ramp_to(want)
      print(f"[hand] at {got.round(3)} rad, wanted {want.round(3)}")

    if args.dry_run:

      def depth_source() -> np.ndarray:
        return _synthetic_depth(policy.depth_hw)
    else:
      from .perception import RealSenseDepth

      # The frame SIZE is the checkpoint's, not the camera's: the stream and the
      # centre crop -- i.e. the field of view -- are the same for every
      # checkpoint, and only the downsample factor moves.
      camera = RealSenseDepth(
        fps=args.camera_fps,
        threaded=not args.camera_blocking,
        preset=args.camera_preset,
        out_hw=policy.depth_hw,
      )
      mode = "blocking" if args.camera_blocking else "threaded"
      h, w = policy.depth_hw
      print(f"[cam] D435 depth at {args.camera_fps} fps, {mode}, -> {h}x{w}")
      depth_source = camera.read

    # Seed the command smoother from where the joints ACTUALLY are, AFTER the
    # home check and the hand's ramp -- those are the deployment's equivalent of
    # the sim's reset events, and the sim seeds from the post-reset pose.
    # Harmless when the profile has neither a limiter nor an EMA; when it has
    # one, seeding from the nominal home instead would make the limiter spend
    # the first steps of every trial ramping across the home check's tolerance,
    # a scripted move no action asked for.
    q_arm, _ = arm.read()
    q_hand_0 = home[calib.HAND_SLICE] if hand is None else hand.read()[0]
    policy.reset(np.concatenate([q_arm, q_hand_0]))

    ramp_steps = int(args.arm_ramp_s * calib.CONTROL_HZ)
    if ramp_steps:
      print(f"[arm] ramping motion limits in over {args.arm_ramp_s:.1f}s")

    for i in range(args.steps):
      t0 = time.perf_counter()

      # Ramp the LIMITS, not the target. The policy still gets to ask for
      # whatever it wants; the arm just refuses to get there quickly at first.
      arm.set_speed_scale(1.0 if i >= ramp_steps else (i + 1) / max(ramp_steps, 1))

      depth = depth_source()
      t_cam = time.perf_counter()

      q_hand, v_hand = (
        (home[calib.HAND_SLICE], np.zeros(4)) if hand is None else hand.read()
      )
      q_arm, v_arm = arm.read()
      joint_pos = np.concatenate([q_arm, q_hand])
      joint_vel = np.concatenate([v_arm, v_hand])
      # From the very packet `q_hand` and `v_hand` were decoded out of, so all
      # three describe one instant rather than three polls of a moving hand.
      if hand is not None:
        finger_cur[i] = hand.last_current_a
      t_sensors = time.perf_counter()

      target = policy.step(joint_pos, joint_vel, depth, args.goal_height)
      overdrive[i] = policy.last_overdrive.max()
      t_policy = time.perf_counter()

      if rec is not None:
        # Recorded BEFORE the --max-jump guard, so the aborting step -- the one
        # worth looking at -- is in the file rather than the one before it.
        rec.put(
          i, joint_pos, joint_vel, policy.last_action, target, depth, finger_cur[i]
        )
        seen = i + 1

      # CARTESIAN FLOOR WITH STOPPING DISTANCE. This guard exists because of a
      # specific incident on 2026-08-07: with the speed cap removed the pads
      # dropped 213 mm in 0.52 s (0.41 m/s) into the foam, and `--max-jump` --
      # which measures JOINT TRACKING ERROR -- did not reach its threshold
      # until two steps AFTER the pads were already 1 mm under the surface.
      # Tracking error is a lagging indicator of a Cartesian problem by
      # construction: it only grows once the arm is failing to follow, and an
      # arm that follows a bad target perfectly never trips it at all.
      #
      # Height alone is not enough either -- at 0.4 m/s the pads cross the last
      # 20 mm in one step. What matters is height MINUS the distance the arm
      # will cover before anything can react, so the threshold scales with how
      # fast it is descending. Replayed against the three recorded runs at a
      # 0.15 s lookahead: run3 (the crash) trips 88 mm above the foam while
      # descending 0.67 m/s, seven steps before the breach.
      #
      # Deliberately on the MEASURED pose, not the commanded target. The
      # commanded target is an absolute joint goal the arm never reaches -- the
      # offset clamp holds it back -- so its forward kinematics sit far below
      # anything the gripper actually visits, and checking it aborted run1 at
      # step 5 of 648 in a run whose pads never went below the foam at all.
      if args.floor_z is not None:
        h = kinematics.gripper_height(joint_pos)
        descent = 0.0 if h_prev is None else max(0.0, (h_prev - h) / period)
        h_prev = h
        if h - descent * args.floor_lookahead < args.floor_z:
          print(
            f"!! step {i}: pads at {h * 1000:.0f} mm above the base plate "
            f"(foam top {(calib.WORK_SURFACE_Z - calib.ARM_BASE_Z) * 1000:.0f} mm)"
            f", descending {descent:.2f} m/s -- inside the {args.floor_lookahead:.2f}s "
            f"stopping distance of --floor-z {args.floor_z * 1000:.0f} mm. Stopping."
          )
          break

      # HARD GUARD, and it is not paranoia. The observation normalizer is baked
      # into the ONNX, so an out-of-distribution input does not degrade the
      # output gracefully -- it multiplies it. Measured on this checkpoint:
      # feeding goal_height = 0.0 instead of a value in [0.50, 0.60] takes
      # max|target| from 2.6 rad to 668 rad, a 250x blow-up, because
      # goal_height's training std is ~0.029 so an out-of-range value lands tens
      # of sigma out. The output is an ABSOLUTE joint target that goes straight
      # to the arm, so any input glitch -- a dropped camera frame, a stale
      # encoder read, a units mistake -- is a full-speed command into the bench.
      # ARM ONLY, and that is not a shortcut. Measured over 7680 sim steps of
      # this checkpoint, |target - measured| behaves completely differently on
      # the two halves:
      #
      #   arm      p50 0.02   p95 0.21   p99 0.39   max 1.40
      #   left_1   p50 1.47   p95 2.46   p99 2.60   max 2.71
      #
      # The fingers sit a radian and a half from their target as a matter of
      # course, because that IS how a position servo makes grip force: you
      # command through the object and the force limit holds you short. Judging
      # them by the arm's standard aborted 62% of sim steps. The hand needs no
      # guard here anyway -- the firmware clamps every goal to POS_MIN/POS_MAX.
      stages[0, i] = t_cam - t0
      stages[1, i] = t_sensors - t_cam
      stages[2, i] = t_policy - t_sensors

      # Judged on the RAW ACTION, and under a slew limit it is the only one of
      # the two that still sees a blow-up in time. The limiter publishes at most
      # `rate / CONTROL_HZ` of new motion per step -- 20 mrad on the arm -- so a
      # |600| action and a |3| action produce the SAME first command and
      # --max-jump only diverges after several steps of ramping the wrong way.
      # policy.act() separately clamps the target to the profile's own joint
      # limits, so an action that is merely large cannot command past a stop.
      worst = float(np.abs(policy.last_action).max())
      if worst > args.max_action:
        j = int(np.abs(policy.last_action).argmax())
        print(
          f"!! step {i}: raw action for {calib.JOINT_NAMES[j]} is {worst:.1f}, over "
          f"--max-action {args.max_action}. Trained actions run to about |5|, so "
          "this is an out-of-distribution observation, not a big reach. Stopping."
        )
        break

      jump = np.abs(target - joint_pos)[calib.ARM_SLICE]
      if jump.max() > args.max_jump:
        j = int(jump.argmax())
        print(
          f"!! step {i}: target for {calib.JOINT_NAMES[j]} is {jump[j]:.2f} rad from "
          f"measured ({target[j]:+.2f} vs {joint_pos[j]:+.2f}), over --max-jump "
          f"{args.max_jump}. Refusing to command it; check the observation."
        )
        break

      arm.set_targets(target[calib.ARM_SLICE])
      if hand is not None:
        hand.step(target[calib.HAND_SLICE])
      t_command = time.perf_counter()
      stages[3, i] = t_command - t_policy
      loop_dt[i] = t_command - t0
      done = i + 1

      if args.print_every and i % args.print_every == 0:
        # Read AFTER the command so the pose shown is the freshest one; the TCP
        # is the whole point of a reach test and is the one thing seven joint
        # angles will not tell you by eye. The rate is over the window just
        # finished, not since the start, so a stall shows up while it is
        # happening rather than being averaged away.
        tcp = arm.tcp_pose()
        where = "" if tcp is None else f" tcp {tcp.round(3)} m"
        lo = max(0, i - args.print_every + 1)
        work = loop_dt[lo : i + 1].mean()
        hz = 1.0 / max(work, period)
        print(
          f"[{i:5d}] q {np.degrees(q_arm).round(1)} deg{where} "
          f"jump {jump.max():.3f} rad  {hz:.1f} Hz"
        )

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
    if arm is not None:
      arm.stop()
    if hand is not None:
      hand.close()
    if camera is not None:
      camera.close()

  print(f"done. {late} late steps of {args.steps}.")
  _report_timing(stage_names, stages, loop_dt, done, period, late, camera)
  _report_overdrive(
    overdrive, done, policy, live=args.robot_sn is not None and not args.dry_run
  )
  _report_grip(finger_cur, seen, args.grip_cap, live=hand is not None)
  if rec is not None and seen:
    rec.save(args.record, seen, args.goal_height, policy, args.grip_cap)
    print(f"[record] {seen} steps -> {args.record}")
  return 0


def _report_timing(names, stages, loop_dt, done, period, late, camera) -> None:
  """Where the 20 ms went, per stage.

  Printed unconditionally, including after an abort, because the run that
  aborts is exactly the one whose timing you want. `late` alone says the loop
  slipped; this says which call did it, which is the difference between a
  camera profile to change and a control loop to redesign.
  """
  if done == 0:
    return
  s, dt = stages[:, :done] * 1e3, loop_dt[:done] * 1e3
  print(f"\n[timing] {done} steps, ms per step")
  print(f"  {'stage':9} {'mean':>7} {'p50':>7} {'p99':>7} {'max':>7}")
  for name, row in zip((*names, "TOTAL"), (*s, dt), strict=True):
    print(
      f"  {name:9} {row.mean():7.2f} {np.percentile(row, 50):7.2f} "
      f"{np.percentile(row, 99):7.2f} {row.max():7.2f}"
    )
  hz = 1.0 / max(dt.mean() * 1e-3, period)
  print(f"  budget {period * 1e3:.0f} ms => {hz:.1f} Hz sustained, {late} steps late")
  if camera is not None and getattr(camera, "_thread", None) is not None:
    # A repeat is the reader thread not having produced a new frame since the
    # last read, i.e. the loop outrunning the sensor. Some is expected and
    # harmless; a large fraction means --camera-fps is still the binding
    # constraint, just no longer a blocking one.
    pct = 100.0 * camera.repeats / done
    print(f"  camera: {camera.repeats} repeated frames ({pct:.1f}% of steps)")


def _report_overdrive(overdrive, done: int, policy: StudentPolicy, live: bool) -> None:
  """How hard the policy leaned on the slew limit, and whether that is normal.

  There is no way to CHECK the profile's rate limit against the weights -- the ONNX
  metadata does not carry the training env's action term -- so this is the
  closest thing to a verification the deployment has. A policy trained against
  this cap asks for a little more than it and gets held back rarely; one trained
  against a looser cap, or none, asks for many times it on nearly every step,
  and every one of those steps is motion the sim never had to survive.

  Only meaningful against a MOVING arm. A stationary stub never reaches the
  state the policy asked for, so the request grows without the command ever
  arriving -- a large overdrive there says the arm is not moving, which you
  already know, and says nothing about the limit being right.
  """
  if done == 0 or policy.rate_limit is None:
    return
  o = overdrive[:done]
  held = 100.0 * float((o > 0).mean())
  print(
    f"\n[limiter] overdrive p50 {np.percentile(o, 50):.2f} "
    f"p95 {np.percentile(o, 95):.2f} max {o.max():.2f}, held back on "
    f"{held:.0f}% of steps"
  )
  if not live:
    print("  (stationary arm: out of distribution by construction, so ignore this)")
  elif np.percentile(o, 95) > 2.0:
    print(
      "  ^ the policy is asking for several times the cap as a matter of course."
      " Either the profile's rate limit is tighter than the checkpoint trained"
      " under, or the observation is wrong. Check the run's env.yaml first."
    )


def _report_grip(current, seen: int, cap_a: float, live: bool) -> None:
  """What the four finger motors were doing with the torque they were allowed.

  The cap IS the grip force -- mode 5 is a position goal plus a current limit --
  so this is the only readout of how hard the hand actually squeezed, and it is
  what separates the two failures that look identical in the joint trace. A
  finger stopped short of its goal AT the cap was held by the object, which is
  a grasp working as designed and, if the object still did not come up, points
  at `--grip-cap`. A finger stopped short while drawing well UNDER the cap was
  not pushing at all, which points at the linkage rather than at the setting.

  Reported per motor and not aggregated, because the interesting case is the
  ASYMMETRIC one: one pad leaning on the object while the other never arrives
  is how an object gets pushed out of the jaw instead of picked up.
  """
  if not live or seen == 0:
    return
  c = np.abs(current[:seen])
  print(f"\n[grip] finger current against the {cap_a:.2f} A cap, amperes")
  print(
    f"  {'motor':<9}{'p50':>7}{'p95':>7}{'max':>7}{'% of steps at >=90% of cap':>28}"
  )
  for j, name in enumerate(calib.JOINT_NAMES[calib.HAND_SLICE]):
    at_cap = 100.0 * float((c[:, j] >= 0.90 * cap_a).mean())
    print(
      f"  {name:<9}{np.percentile(c[:, j], 50):>7.3f}"
      f"{np.percentile(c[:, j], 95):>7.3f}{c[:, j].max():>7.3f}{at_cap:>27.0f}%"
    )
  if c.max() < 0.5 * cap_a:
    print(
      "  ^ nothing ever came within half the cap, so no finger was working "
      "against anything. A failed grasp here is a MISS -- the pads closed on "
      "empty space -- not a weak squeeze, and raising --grip-cap will not "
      "change it."
    )


def _synthetic_depth(hw: tuple[int, int] = calib.DEPTH_HW) -> np.ndarray:
  """A flat bench at 0.6 m with a box on it -- shape and range only, not a scene.

  Sized to the checkpoint, so `--dry-run` exercises the same tensor shape the
  bench would hand it. The box is placed in fractions of the frame rather than
  in pixels, so it lands in the same place at either resolution.
  """
  h, w = hw
  d = np.full(hw, 0.6, dtype=np.float32)
  d[h // 2 : (h * 2) // 3, (w * 7) // 16 : (w * 19) // 32] = 0.52
  return (
    np.clip(d, calib.DEPTH_MIN_M, calib.DEPTH_CUTOFF_M) / calib.DEPTH_CUTOFF_M
  ).reshape(1, *hw)


if __name__ == "__main__":
  raise SystemExit(main())
