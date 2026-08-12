"""The seven Flexiv joints. Interface only -- the driver is not in this repo.

The policy emits ONE 11-vector spanning arm and hand, but they are two different
pieces of hardware with two different vendors' interfaces. The gripper repo
(`/home/yiboc/gripper`) covers the four finger motors and nothing else; the arm
goes through Flexiv RDK, which is not vendored here.

So this file pins the CONTRACT the arm side has to satisfy, and `run.py` talks
to it through that. Implement `FlexivArm` against RDK, or drive a different arm,
without touching the policy path.

The contract, and each clause is a way the deployment goes silently wrong:

  * Joint order is calib.JOINT_NAMES[0:7] -- joint1..joint7, base to wrist.
  * Positions in RADIANS, matching the URDF the sim was built from. RDK reports
    radians too, but confirm the ZERO: the sim's home pose is
    (0, -0.368, 0.217, 2.338, -0.180, 1.122, 1.153) and if that pose does not
    put the real arm in the same configuration, every action offset is measured
    from the wrong place.
  * `set_targets` is a POSITION command at 50 Hz, not a trajectory. The sim ran
    a joint-position servo with the stiffness/damping in the ONNX metadata
    (kp 289/673/224/373/237/232/186, kd 61/143/36/59/13/12/9.9). RDK's impedance
    mode with comparable gains is the closest match; a MoveJ-style planner is
    NOT, because it will smooth away exactly the 50 Hz corrections the policy is
    making.
  * The base frame the policy assumes has the arm's mounting plate at
    calib.ARM_BASE_Z above the floor, with the work surface at
    calib.WORK_SURFACE_Z. Those two numbers are what make `goal_height`
    meaningful.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class ArmInterface(Protocol):
  """What run.py needs from the arm."""

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """-> (joint_pos[7] rad, joint_vel[7] rad/s), in calib.JOINT_NAMES order."""
    ...

  def set_targets(self, q_target: np.ndarray) -> None:
    """Command joint position targets[7] in rad. Called at calib.CONTROL_HZ."""
    ...

  def set_speed_scale(self, scale: float) -> None:
    """Scale the motion limits, in [0, 1]. Used to ramp in from a standstill."""
    ...

  def tcp_pose(self) -> np.ndarray | None:
    """-> TCP position [x, y, z] in metres in the ROBOT BASE frame, or None.

    Reporting only. It is what answers "did the arm get to the cube", which no
    amount of joint angles answers by inspection, and it is the one quantity a
    hand-less bring-up run is actually watching.
    """
    ...

  def stop(self) -> None:
    """Bring the arm to a safe hold. Called on every exit path."""
    ...


def resolve_max_offset(
  want: float | Sequence[float] | None,
  tau_max: np.ndarray,
  k_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """-> (offset to use, tau_max/K_q) both per joint, in rad.

  Split out of `FlexivArm.__init__` so the one piece of arithmetic that decides
  how much torque the policy is allowed can be tested without a robot.
  """
  full = np.asarray(tau_max, dtype=float) / np.asarray(k_q, dtype=float)
  if want is None:
    return full.copy(), full
  return np.broadcast_to(np.asarray(want, dtype=float), full.shape).copy(), full


class FlexivArm:
  """The seven Rizon joints, through RDK's non-realtime joint-impedance mode.

  WHICH RDK. v1.9.x is the line that still supports the Rizon 4S -- 2.x dropped
  the model -- so this is written against the flat single-arm API of that line:
  `Robot(sn)`, `states()` returning a struct of attributes, `SwitchMode(Mode.X)`,
  no joint-group arguments.

  WHY IMPEDANCE AND NOT NRT_JOINT_POSITION. The policy emits a new absolute
  target every 20 ms and expects a compliant servo to chase it, which is what it
  had in sim (`actuator_gainprm` kp, `-actuator_biasprm` kd). Impedance mode with
  K_q set to those same kp values is the closest available match. The position
  controller would track each target stiffly and is a worse fit, and a MoveJ-style
  planner is not a fit at all -- it smooths away the 50 Hz corrections that ARE
  the policy.

  WHAT IS NOT MATCHED, and it is not fixable from here. The sim's kd is an
  ABSOLUTE damping in N.m.s/rad; RDK's optional `Z_q` is a damping RATIO, and
  converting between them needs the joint-space inertia at the current pose,
  which RDK does not expose. So this sets STIFFNESS only and leaves RDK's own
  damping in place. Expect the real arm to be damped differently from sim; if it
  rings at 50 Hz, that is the first thing to look at, not the policy.

  THE TORQUE LIMIT IS NOT A DEPLOYMENT CHOICE -- it is part of the training
  environment, and this used to say the opposite. The old note here claimed the
  sim arm was unlimited because `actuator_forcerange` is 0,0 on all seven
  joints. That is the wrong field. The wide-claw model sets the JOINT's
  `actfrcrange` from the sys-id fit (`_SYSID_JOINTS` in arm_cfg.py), and those
  values are [123, 123, 64, 64, 39, 39, 39] Nm -- EXACTLY the robot's own
  `info().tau_max`. Verified on the compiled model.

  So sim and hardware saturate identically, and the largest offset the training
  environment could ever sustain is tau_max / K_q per joint:

      j1 0.426  j2 0.183  j3 0.286  j4 0.172  j5 0.165  j6 0.168  j7 0.210 rad

  `max_offset_rad` therefore defaults to exactly that vector, read from the
  robot's own tau_max at startup rather than transcribed. It is a faithful
  reproduction of the training limit, not a safety margin bolted on top: a
  SCALAR below it -- 0.08 rad, say -- silently cuts torque authority to between
  19% (joint1) and 48% (joint5) of what the policy trained with, which is a
  rate limit inside the closed loop and produces a limit cycle, not a slower
  version of the same behaviour. Pass a scalar only to deliberately throttle,
  and expect the policy to behave differently when you do.
  """

  def __init__(
    self,
    robot_sn: str,
    stiffness: np.ndarray | None = None,
    max_vel: float = 1.5,
    max_acc: float = 6.0,
    max_offset_rad: float | Sequence[float] | None = None,
    limit_margin_rad: float = 0.05,
  ):
    import flexivrdk  # noqa: PLC0415  (only needed on the robot host)

    from . import calib  # noqa: PLC0415  (avoids a cycle at module import)

    self._n = 7
    self.robot = flexivrdk.Robot(robot_sn)
    self.Mode = flexivrdk.Mode
    self.max_vel = max_vel
    self.max_acc = max_acc
    self.speed_scale = 1.0
    self._want_offset = max_offset_rad  # resolved once K_q and tau_max are known

    if self.robot.fault():
      print("[arm] fault present; clearing")
      self.robot.ClearFault()
    self.robot.Enable()
    print("[arm] enabling; waiting until operational...")
    while not self.robot.operational():
      time.sleep(0.1)

    info = self.robot.info()
    if int(info.DoF) != self._n:
      raise RuntimeError(
        f"robot reports DoF {info.DoF}, this policy commands {self._n}. "
        "Wrong robot model?"
      )
    self.q_min = np.asarray(info.q_min, dtype=float)[: self._n] + limit_margin_rad
    self.q_max = np.asarray(info.q_max, dtype=float)[: self._n] - limit_margin_rad
    self.tau_max = np.asarray(info.tau_max, dtype=float)[: self._n]
    kq_nom = np.asarray(info.K_q_nom, dtype=float)[: self._n]

    want = (
      np.asarray(calib.JOINT_STIFFNESS, dtype=float)[calib.ARM_SLICE]
      if stiffness is None
      else np.asarray(stiffness, dtype=float)
    )
    # SetJointImpedance rejects anything above K_q_nom, so a sim gain the robot
    # cannot represent has to be clamped -- but silently clamping changes how
    # every action is tracked, so say so.
    self.k_q = np.clip(want, 0.0, kq_nom)
    if not np.allclose(self.k_q, want):
      print(
        f"[arm] WARNING: sim stiffness {want.round(0)} exceeds the robot's "
        f"K_q_nom {kq_nom.round(0)} and was clamped to {self.k_q.round(0)}. "
        "Those joints will track the policy's targets more softly than sim did."
      )

    # The offset that reproduces the training environment's own saturation.
    # Derived from the robot rather than transcribed, so it cannot drift from
    # the machine it is running on -- and the sim's actfrcrange IS this tau_max.
    self.max_offset, self.full_offset = resolve_max_offset(
      self._want_offset, self.tau_max, self.k_q
    )
    ceiling = self.k_q * self.max_offset
    print(f"[arm] model {info.model_name}, K_q {self.k_q.round(0)}")
    print(
      f"[arm] |q_d - q| <= {self.max_offset.round(3)} rad => torque "
      f"{ceiling.round(1)} Nm of tau_max {self.tau_max.round(0)}"
    )
    frac = self.max_offset / self.full_offset
    if np.any(frac < 0.95):
      # Not cosmetic. Anything below tau_max/K_q is a rate limit sitting inside
      # the policy's closed loop: the arm cannot reach the commanded target, the
      # policy sees a stale pose and pushes harder, then overshoots when it
      # finally arrives. That is a limit cycle, not a slower version of the
      # trained behaviour, so it has to be visible in the log.
      print(
        f"[arm] WARNING: this is {(100 * frac).round(0)}% of the torque "
        f"authority the policy trained with (tau_max/K_q = "
        f"{self.full_offset.round(3)}). The training environment saturates at "
        "the SAME tau_max, so anything below 100% is a limit the policy has "
        "never seen and is a common cause of oscillation."
      )
    if np.any(ceiling > self.tau_max * 1.001):
      print("[arm] WARNING: commanded ceiling exceeds tau_max; the robot will clip.")

    self.robot.SwitchMode(self.Mode.NRT_JOINT_IMPEDANCE)
    self.robot.SetJointImpedance([float(k) for k in self.k_q])

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    s = self.robot.states()
    q = np.asarray(s.q, dtype=float)[: self._n]
    dq = np.asarray(s.dq, dtype=float)[: self._n]
    return q, dq

  def set_targets(self, q_target: np.ndarray) -> None:
    q_now, _ = self.read()
    target = np.asarray(q_target, dtype=float)[: self._n]
    # Two clamps, in this order. The offset clamp bounds the torque the
    # impedance controller will ask for; the limit clamp keeps the commanded
    # pose inside the robot's own joint range. The trained pipeline has NO
    # action clipping, so without these the network's raw output is the command.
    s = max(self.speed_scale, 1e-3)  # never send a zero limit; the robot stalls
    target = q_now + np.clip(target - q_now, -self.max_offset * s, self.max_offset * s)
    target = np.clip(target, self.q_min, self.q_max)
    self.robot.SendJointPosition(
      [float(v) for v in target],
      [0.0] * self._n,
      [self.max_vel * s] * self._n,
      [self.max_acc * s] * self._n,
    )

  def set_speed_scale(self, scale: float) -> None:
    self.speed_scale = float(np.clip(scale, 0.0, 1.0))

  def tcp_pose(self) -> np.ndarray | None:
    """First three of RDK's `tcp_pose` (x, y, z, qw, qx, qy, qz), in metres.

    Wrapped rather than read inline because the attribute name is the one thing
    here that has moved between RDK lines, and a reporting field is never worth
    taking down a running policy loop for.
    """
    try:
      return np.asarray(self.robot.states().tcp_pose, dtype=float)[:3]
    except Exception:  # noqa: BLE001 -- reporting only
      return None

  def stop(self) -> None:
    try:
      self.robot.Stop()
      self.robot.SwitchMode(self.Mode.IDLE)
    except Exception as exc:  # noqa: BLE001 -- best effort on the way out
      print(f"[arm] WARNING: stop failed: {exc}")


class ReplayArm:
  """A stand-in that never moves, for testing the policy path without the arm.

  Reports the sim's home pose as a static measurement, so `run.py` can be driven
  end to end -- camera, ONNX, gripper -- with the arm out of the loop. The
  targets it is handed are recorded rather than executed.
  """

  def __init__(self, home: np.ndarray):
    self.q = np.asarray(home, dtype=np.float64)[:7].copy()
    self.commanded: list[np.ndarray] = []

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    return self.q.copy(), np.zeros(7)

  def set_targets(self, q_target: np.ndarray) -> None:
    self.commanded.append(np.asarray(q_target, dtype=np.float64).copy())

  def set_speed_scale(self, scale: float) -> None:
    pass

  def tcp_pose(self) -> np.ndarray | None:
    return None

  def stop(self) -> None:
    pass
