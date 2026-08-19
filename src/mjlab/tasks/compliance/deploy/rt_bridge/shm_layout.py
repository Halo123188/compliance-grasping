"""Shared-memory layout for the RT torque bridge.

This is the single source of truth for the wire format between the C++ RT shim
(``rt_torque_bridge.cpp``, 1 kHz) and the Python policy loop (100 Hz).  The C++
side mirrors these offsets and ``static_assert``s each one, so a mismatch is a
compile error there rather than silent garbage at runtime.

Concurrency
-----------
Both blocks use a **seqlock**: the writer bumps the sequence counter to an odd
value, writes the payload, then bumps it to the next even value.  A reader
snapshots the counter, reads the payload, and re-reads the counter -- if it is
odd, or changed, the read was torn and is retried.  No locks, no syscalls, and
a slow reader can never block the 1 kHz writer.

All fields are 8-byte aligned and the payloads are naturally aligned doubles,
so each individual load/store is atomic on x86-64.  The seqlock protects the
*group*, which is what actually matters here.
"""

from __future__ import annotations

import struct

SHM_NAME = "flexiv_rt_bridge"
SHM_PATH = f"/dev/shm/{SHM_NAME}"
SHM_SIZE = 4096

MAGIC = 0x464C58565254_3031  # "FLXVRT01"
# v2 added the runtime lock override (lock mask + hold target) to the command
# block.  The C++ side publishes this and ``_Shm`` refuses to attach on a
# mismatch, so an un-rebuilt bridge fails loudly instead of ignoring the new
# fields and leaving six joints unheld.
LAYOUT_VERSION = 2

NUM_JOINTS = 7
_VEC = 8 * NUM_JOINTS  # 56 bytes

# ── header: written once by C++ before it signals info_ready ────────────────
OFF_MAGIC = 0x000
OFF_VERSION = 0x008
OFF_INFO_READY = 0x010
OFF_DOF = 0x018
OFF_Q_MIN = 0x020
OFF_Q_MAX = 0x058
OFF_DQ_MAX = 0x090
OFF_TAU_MAX = 0x0C8
OFF_KQ_NOM = 0x100

# ── state block: written by C++ at 1 kHz, read by Python at 100 Hz ──────────
OFF_STATE_SEQ = 0x400
OFF_Q = 0x408
OFF_DQ = 0x440
OFF_TAU = 0x478
OFF_TAU_EXT = 0x4B0
OFF_STATE_TICK = 0x4E8
OFF_LATE_COUNT = 0x4F0
OFF_FAULT = 0x4F8

# ── command block: written by Python at 100 Hz, read by C++ at 1 kHz ────────
OFF_CMD_SEQ = 0x800
OFF_TAU_CMD = 0x808
OFF_CMD_EPOCH = 0x840
OFF_STOP_FLAG = 0x848
# Runtime lock override, all three inside the command seqlock so the torque and
# the lock set can never be read from different generations.
#
# ``OFF_LOCK_OVERRIDE`` 0 means "ignore the two fields below and behave exactly
# as ``--lock-joints`` plus the startup pose did".  That default is what keeps
# ``deploy_rizon_torque.py`` working untouched: it writes zero here and the
# bridge's CLI locks stay in force.  Set it to 1 and the bridge takes its lock
# set from ``OFF_LOCK_MASK`` (bit i => joint i+1 held) and its hold target from
# ``OFF_Q_HOLD`` instead of the angles it captured at startup.
#
# The bridge does not trust either field: it slew-rate-limits the target it
# actually tracks and clamps it inside the robot's own joint limits, so a bug on
# this side cannot step a joint or drive one into a stop.
OFF_LOCK_OVERRIDE = 0x850
OFF_LOCK_MASK = 0x858
OFF_Q_HOLD = 0x860

_U64 = struct.Struct("<Q")
_VEC7 = struct.Struct("<7d")

assert OFF_Q_MIN + 5 * _VEC <= OFF_STATE_SEQ
assert OFF_TAU_EXT + _VEC <= OFF_STATE_TICK
assert OFF_TAU_CMD + _VEC <= OFF_CMD_EPOCH
assert OFF_STOP_FLAG + 8 <= OFF_LOCK_OVERRIDE
assert OFF_LOCK_OVERRIDE + 8 <= OFF_LOCK_MASK
assert OFF_LOCK_MASK + 8 <= OFF_Q_HOLD
assert OFF_Q_HOLD + _VEC <= SHM_SIZE


def read_u64(buf, off: int) -> int:
  return _U64.unpack_from(buf, off)[0]


def write_u64(buf, off: int, val: int) -> None:
  _U64.pack_into(buf, off, val)


def read_vec(buf, off: int) -> tuple[float, ...]:
  return _VEC7.unpack_from(buf, off)


def write_vec(buf, off: int, vals) -> None:
  _VEC7.pack_into(buf, off, *vals)
