"""The four finger joints: sim radians <-> gripper encoder counts.

Wraps `firmware/host/policy_link.GripperLink` from the gripper repo. That link is
already the right shape for this -- synchronous ACTION -> OBS, one exchange per
control step -- so this module is only the unit conversion and the safety
interlocks around it.

WHAT THIS MODULE CANNOT KNOW. The sim and the hardware agree on the topology
(two fingers, a proximal and a distal joint each) and on nothing else:

  sim   left_1 = +0.2746 rad is OPEN (86.9 mm of pad separation), -0.0754 rad is
        closed (40 mm); the distal joints sit at 0 and the policy curls them.
  real  homing offset is zeroed at the STRAIGHT / NEUTRAL pose, mid-travel and
        deliberately not a mechanical limit, so 0 counts is roughly halfway
        between open and closed on all four; the closing direction is per-finger
        mixed sign because a pinch is a mirror-image motion.

There is no shared zero, no shared sign and no shared scale, and neither repo
records the linkage ratio. `deploy/calibrate_hand.py` measures the affine map;
until its output is pasted into calib.py this module refuses to arm rather than
driving a finger into the frame on a guessed sign.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from . import calib

# The gripper repo is a sibling checkout, not a package. Point at it explicitly
# so the import failure names the actual problem.
#
# Derived from THIS file's location rather than hardcoded: the default used to
# be one machine's absolute path (`/home/yiboc/...`), which is a
# FileNotFoundError on every host whose checkout lives anywhere else -- and it
# fails at `home_arm`, i.e. after you have already walked to the robot. Any
# layout that keeps the two repos side by side now works unconfigured, and
# $GRIPPER_HOST_DIR still overrides for one that does not.
GRIPPER_HOST_DIR = Path(
  os.environ.get(
    "GRIPPER_HOST_DIR",
    Path(__file__).resolve().parents[2] / "gripper" / "firmware" / "host",
  )
)


def _import_link():
  if not (GRIPPER_HOST_DIR / "policy_link.py").exists():
    raise FileNotFoundError(
      f"policy_link.py not found under {GRIPPER_HOST_DIR}. That default assumes "
      "the gripper repo sits beside this one; point $GRIPPER_HOST_DIR at your "
      "checkout's firmware/host if it does not."
    )
  sys.path.insert(0, str(GRIPPER_HOST_DIR))
  from policy_link import GripperLink  # type: ignore[import-not-found]

  return GripperLink


class Hand:
  """Finger joints in sim radians, over the binary policy protocol.

  Motor order is the protocol's global order (f1j1, f1j2, f2j1, f2j2 = ids
  12, 11, 21, 22), and calib.COUNTS_* are indexed to match. Which physical
  finger is "left" is part of the calibration, not a convention -- get it
  backwards and every asymmetric grasp mirrors, silently.
  """

  def __init__(self, port: str | None = None, cap_a: float = calib.GRIP_CURRENT_CAP_A):
    """`port` is not optional in practice on Linux.

    GripperLink's auto-detect globs ``/dev/cu.usbmodem*``, which is the macOS
    naming; a Linux host enumerates the Teensy as ``/dev/ttyACM0``. Auto-detect
    there fails with "no Teensy serial port found" even though the board is
    plugged in, so pass the port explicitly.
    """
    if calib.COUNTS_PER_RAD is None or calib.COUNTS_AT_ZERO_RAD is None:
      raise RuntimeError(
        "hand calibration missing: calib.COUNTS_PER_RAD / COUNTS_AT_ZERO_RAD are "
        "None. Run deploy/calibrate_hand.py and paste its output into calib.py. "
        "Refusing to command counts from a guessed sign."
      )
    if getattr(calib, "HAND_CALIB_IS_PROVISIONAL", False):
      # Loud at every arm, because the failure it causes is quiet: a wrong scale
      # produces grasps that are simply the wrong width, which reads as a bad
      # policy rather than a bad constant.
      print(
        "[hand] WARNING: running on a PROVISIONAL calibration. The offset and "
        "signs are derived, but COUNTS_PER_RAD is the 1:1 guess 4096/2pi. Finger "
        "angles -- and so grasp width -- are off by however wrong that is. Do "
        "not diagnose a failed grasp before finishing calib.py's hand block."
      )
    self.k = np.asarray(calib.COUNTS_PER_RAD, dtype=np.float64)
    self.b = np.asarray(calib.COUNTS_AT_ZERO_RAD, dtype=np.float64)
    self.cap_a = cap_a
    self.link = _import_link()(port=port)
    self.counts_per_rev = self.link.info_["counts_per_rev"]
    if self.link.n != 4:
      raise RuntimeError(
        f"controller reports {self.link.n} motors; this policy commands 4. A "
        "partial set means the gripper failed bring-up -- check the 4th motor "
        "(gripper NEXT_STEPS.md lists slot id 22 as not fitted)."
      )
    if self.link.info_["mode"] != 5:
      raise RuntimeError(
        f"controller is in mode {self.link.info_['mode']}, not 5 "
        "(current-based position). The policy commands positions with a force "
        "cap; any other mode reinterprets those numbers."
      )

  # --- units ---------------------------------------------------------------
  def rad_to_counts(self, q: np.ndarray) -> np.ndarray:
    """(4,) sim radians -> (4,) encoder counts, in protocol motor order."""
    return np.rint(self.k * np.asarray(q, dtype=np.float64) + self.b).astype(np.int64)

  def counts_to_rad(self, counts: np.ndarray) -> np.ndarray:
    return (np.asarray(counts, dtype=np.float64) - self.b) / self.k

  @staticmethod
  def aperture_mm(q_proximal: float) -> float:
    """Pad separation for a proximal joint angle, from the sim's measured fit."""
    return calib.APERTURE_A_MM + calib.APERTURE_B_MM * q_proximal

  # --- control -------------------------------------------------------------
  def step(self, q_target: np.ndarray) -> dict:
    """Command 4 finger targets in sim radians; return the decoded OBS.

    One exchange per call. The controller drops torque after 500 ms without an
    ACTION, so this must keep being called -- a stalled policy disarms the hand
    rather than leaving it clamped on a stale goal.
    """
    obs = self.link.action(
      torque=True,
      cap_a=self.cap_a,
      goals=self.rad_to_counts(q_target).tolist(),
    )
    if obs.get("fault"):
      raise RuntimeError(f"gripper fault {obs['fault']} (see PROTOCOL.md Fault enum)")
    return obs

  def ramp_to(self, target: np.ndarray, secs: float = 1.5) -> np.ndarray:
    """Walk the fingers to `target` (sim radians) over `secs`, and report where
    they landed.

    Ramped rather than commanded in one step. The servo would otherwise chase
    the whole error at the full current cap, and the pose this is usually called
    with -- the policy's default -- is about 0.27 rad from the gripper's homing
    zero, which is a long way to lunge.

    Used by `home_arm.py` before a run and by `run.py` immediately before the
    first inference: the policy's first action is computed from the MEASURED
    state, so a hand sitting anywhere but its default makes that first command a
    large step.
    """
    q0, _ = self.read()
    target = np.asarray(target, dtype=np.float64)
    n = max(1, int(secs * calib.CONTROL_HZ))
    for i in range(1, n + 1):
      self.step(q0 + (target - q0) * (i / n))
      time.sleep(1.0 / calib.CONTROL_HZ)
    q1, _ = self.read()
    return q1

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """Read-only poll -> (joint_pos rad, joint_vel rad/s), protocol motor order.

    Velocity arrives in the DYNAMIXEL register unit (0.229 rev/min at the motor),
    so it takes two steps: to counts/s via counts_per_rev, then to joint rad/s
    through the SAME k the positions use. Signed division, not |k| -- k carries
    the per-motor sign flip, and dropping it reports a closing finger as opening.
    """
    obs = self.link.poll()
    return self.decode(obs)

  def decode(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Split an OBS into (joint_pos rad, joint_vel rad/s)."""
    counts = np.array([m["pos"] for m in obs["motors"]], dtype=np.float64)
    vel_raw = np.array([m["vel"] for m in obs["motors"]], dtype=np.float64)
    counts_per_s = vel_raw * 0.229 / 60.0 * self.counts_per_rev
    return self.counts_to_rad(counts), counts_per_s / self.k

  def close(self) -> None:
    self.link.close()
