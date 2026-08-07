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

  def stop(self) -> None:
    """Bring the arm to a safe hold. Called on every exit path."""
    ...


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

  def stop(self) -> None:
    pass
