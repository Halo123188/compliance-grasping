"""Python half of the RT torque bridge.

``ShmAdapter`` and ``ShmTorqueBackend`` are drop-in replacements for
``RizonAdapter`` / ``RtTorqueBackend`` in ``deploy_rizon_torque.py``: same
``read_state()`` / ``start()`` / ``send()`` surface, but every robot interaction
goes through shared memory to the C++ bridge, which owns the RDK connection and
the 1 kHz ``RT_JOINT_TORQUE`` stream.

Start the bridge first, then the policy:

    ./rt_torque_bridge Rizon4s-063501 --enable
    uv run python .../deploy_rizon_torque.py --policy tracking --backend shm
"""

from __future__ import annotations

import mmap
import os
import time

import numpy as np

from . import shm_layout as L


class BridgeNotRunning(RuntimeError):
  pass


class _Shm:
  """mmap of the C++ bridge's segment, with the seqlock read/write helpers."""

  def __init__(self, timeout_s: float = 10.0):
    deadline = time.perf_counter() + timeout_s
    while True:
      if os.path.exists(L.SHM_PATH):
        try:
          self._fd = os.open(L.SHM_PATH, os.O_RDWR)
          break
        except OSError:
          pass
      if time.perf_counter() > deadline:
        raise BridgeNotRunning(
          f"{L.SHM_PATH} not present after {timeout_s:.0f}s. Start the C++ bridge "
          f"first:\n    ./rt_torque_bridge <robot_sn> --enable\n"
          f"(or --sim to test without a robot)"
        )
      time.sleep(0.05)

    self.buf = mmap.mmap(self._fd, L.SHM_SIZE)

    while True:
      if (
        L.read_u64(self.buf, L.OFF_INFO_READY) == 1
        and L.read_u64(self.buf, L.OFF_MAGIC) == L.MAGIC
      ):
        break
      if time.perf_counter() > deadline:
        raise BridgeNotRunning("bridge mapped but never published its header")
      time.sleep(0.05)

    ver = L.read_u64(self.buf, L.OFF_VERSION)
    if ver != L.LAYOUT_VERSION:
      raise BridgeNotRunning(
        f"layout mismatch: bridge speaks v{ver}, this client v{L.LAYOUT_VERSION}. "
        f"Rebuild rt_torque_bridge."
      )
    dof = L.read_u64(self.buf, L.OFF_DOF)
    if dof != L.NUM_JOINTS:
      raise BridgeNotRunning(f"bridge reports DoF {dof} != {L.NUM_JOINTS}")

    self._cmd_seq = 0
    self._cmd_epoch = 0

  def read_state(self) -> dict:
    """Seqlock read of the state block. Retries until a clean snapshot."""
    for _ in range(64):
      s0 = L.read_u64(self.buf, L.OFF_STATE_SEQ)
      if s0 & 1:
        continue
      q = L.read_vec(self.buf, L.OFF_Q)
      dq = L.read_vec(self.buf, L.OFF_DQ)
      tau = L.read_vec(self.buf, L.OFF_TAU)
      tau_ext = L.read_vec(self.buf, L.OFF_TAU_EXT)
      tick = L.read_u64(self.buf, L.OFF_STATE_TICK)
      fault = L.read_u64(self.buf, L.OFF_FAULT)
      if L.read_u64(self.buf, L.OFF_STATE_SEQ) == s0:
        return {
          "q": np.asarray(q, dtype=float),
          "dq": np.asarray(dq, dtype=float),
          "tau": np.asarray(tau, dtype=float),
          "tau_ext": np.asarray(tau_ext, dtype=float),
          "tick": int(tick),
          "fault": bool(fault),
        }
    raise RuntimeError("state seqlock never settled (is the bridge wedged?)")

  def write_command(
    self,
    tau: np.ndarray,
    lock_mask: int | None = None,
    q_hold: np.ndarray | None = None,
  ) -> None:
    """Seqlock write of the command block.

    ``lock_mask`` and ``q_hold`` are the layout-v2 runtime lock override: which
    joints the bridge holds (bit i => joint i+1) and the angles it holds them
    about.  Passing neither leaves the override off, which is what the policy
    loop wants -- the bridge then keeps whatever ``--lock-joints`` it was started
    with.  They are written inside the same seqlock as the torque so the bridge
    can never pair a torque with a stale lock set.

    The bridge treats ``q_hold`` as a request: it slews toward it and clamps it
    inside the robot's own joint limits, so this call cannot step a joint or
    drive one into a stop.
    """
    if lock_mask is not None and q_hold is None:
      raise ValueError("lock_mask needs q_hold: a lock with no target is undefined")
    self._cmd_seq += 1
    L.write_u64(self.buf, L.OFF_CMD_SEQ, self._cmd_seq)  # odd: in progress
    L.write_vec(self.buf, L.OFF_TAU_CMD, [float(t) for t in tau])
    if lock_mask is not None and q_hold is not None:
      L.write_u64(self.buf, L.OFF_LOCK_MASK, lock_mask)
      L.write_vec(self.buf, L.OFF_Q_HOLD, [float(x) for x in q_hold])
    L.write_u64(self.buf, L.OFF_LOCK_OVERRIDE, 0 if lock_mask is None else 1)
    self._cmd_epoch += 1
    L.write_u64(self.buf, L.OFF_CMD_EPOCH, self._cmd_epoch)
    self._cmd_seq += 1
    L.write_u64(self.buf, L.OFF_CMD_SEQ, self._cmd_seq)  # even: consistent

  def request_stop(self) -> None:
    L.write_u64(self.buf, L.OFF_STOP_FLAG, 1)

  def limits(self) -> dict:
    return {
      "q_min": np.asarray(L.read_vec(self.buf, L.OFF_Q_MIN), dtype=float),
      "q_max": np.asarray(L.read_vec(self.buf, L.OFF_Q_MAX), dtype=float),
      "dq_max": np.asarray(L.read_vec(self.buf, L.OFF_DQ_MAX), dtype=float),
      "tau_max": np.asarray(L.read_vec(self.buf, L.OFF_TAU_MAX), dtype=float),
      "kq_nom": np.asarray(L.read_vec(self.buf, L.OFF_KQ_NOM), dtype=float),
    }

  def close(self) -> None:
    try:
      self.buf.close()
      os.close(self._fd)
    except Exception:  # noqa: BLE001 — best effort on the way out
      pass


class _RobotShim:
  """Just enough of the RDK Robot surface for ``check_guards`` to work as-is."""

  def __init__(self, shm: _Shm):
    self._shm = shm

  def fault(self) -> bool:
    return self._shm.read_state()["fault"]


class ShmAdapter:
  """``RizonAdapter`` over the bridge. The C++ side owns the RDK connection.

  ``state_factory`` is ``deploy_rizon_torque.state_from_joints``, passed in
  rather than imported to keep this module free of a circular import (and
  runnable on its own for IPC tests).
  """

  def __init__(self, spec, state_factory, timeout_s: float = 10.0):
    self.spec = spec
    self._state_factory = state_factory
    self.shm = _Shm(timeout_s=timeout_s)
    self.robot = _RobotShim(self.shm)

    lim = self.shm.limits()
    self.dof = L.NUM_JOINTS
    self.q_min = lim["q_min"]
    self.q_max = lim["q_max"]
    self.dq_max = lim["dq_max"]
    self.tau_max = lim["tau_max"]
    self.kq_nom = lim["kq_nom"]
    self._last_tick = -1
    self.stale_reads = 0

  def read_state(self):
    s = self.shm.read_state()
    # The bridge ticks at 1 kHz and we poll at 100 Hz, so the tick must always
    # have advanced. If it has not, the bridge is wedged or dead.
    if s["tick"] == self._last_tick:
      self.stale_reads += 1
    else:
      self.stale_reads = 0
    self._last_tick = s["tick"]
    return self._state_factory(
      s["q"][: L.NUM_JOINTS],
      s["dq"][: L.NUM_JOINTS],
      s["tau"][: L.NUM_JOINTS],
      self.spec,
    )

  def probe(self) -> None:
    s = self.shm.read_state()
    print("\n--- via RT bridge (shared memory) ---")
    print(f"  tick       : {s['tick']}  (1 kHz since bridge start)")
    print(f"  fault      : {s['fault']}")
    print(f"  q          : {s['q'].round(4)}")
    print(f"  dq         : {s['dq'].round(4)}")
    print(f"  tau        : {s['tau'].round(3)}")
    print(f"  tau_ext    : {s['tau_ext'].round(3)}")
    print(f"  q_min (deg): {np.degrees(self.q_min).round(1)}")
    print(f"  q_max (deg): {np.degrees(self.q_max).round(1)}")
    print(f"  tau_max    : {self.tau_max.round(1)}")

  def stop(self) -> None:
    try:
      self.shm.request_stop()
    finally:
      self.shm.close()


class ShmTorqueBackend:
  """Write the residual to the bridge; it streams at 1 kHz with gravity comp.

  The 10:1 ratio between the bridge's 1 kHz stream and this 100 Hz loop *is*
  the sim's ``decimation=10``: one policy torque is held for ten ticks.
  """

  send_hz = 100.0

  def __init__(self, adapter: ShmAdapter):
    self.adapter = adapter

  @staticmethod
  def supported(adapter) -> bool:
    return isinstance(adapter, ShmAdapter)

  def describe(self) -> str:
    return (
      "RT_JOINT_TORQUE via the C++ bridge @ 1 kHz (gravity comp on-robot); "
      "policy writes the residual at 100 Hz over shared memory"
    )

  def start(self) -> None:
    # Deliberately does NOT send anything. The bridge holds its startup pose
    # under joint impedance until the *first* command arrives, and sending a
    # zero here would end that hold early — leaving the arm at zero stiffness
    # for the rest of this call and the first loop iteration. The bridge is
    # already in RT_JOINT_TORQUE; the first real send() takes over, crossfaded
    # over the bridge's --release-ms.
    pass

  def send(self, tau_residual: np.ndarray) -> None:
    if self.adapter.stale_reads > 5:
      raise RuntimeError(
        "RT bridge stopped ticking — it has died or faulted. Check its console."
      )
    self.adapter.shm.write_command(tau_residual)
