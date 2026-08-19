"""Per-joint system identification on the real Rizon 4S, for transfer into sim.

Measures the four things the MuJoCo model gets wrong, one joint at a time:

  ``armature``      reflected rotor inertia.  The XML declares 3.17/1.38/0.13 in
                    its tier default classes but nothing applies them to the
                    ``<joint>`` elements, so the model compiled with
                    ``dof_armature`` all zero -- the shoulder came out 6x too
                    light.  This script measures the *total* effective inertia and
                    subtracts the model's rigid-body ``M[j,j]``; the remainder is
                    the armature to set.
  ``frictionloss``  Coulomb friction.  Zero in the model.  A harmonic drive has
                    several Nm of it, and leaving it out is why the real arm needs
                    more torque than sim to start moving at all.
  ``damping``       viscous friction.  Also zero in the model.
  gravity-comp bias how much torque the robot's on-board gravity compensation
                    leaves on the joint at the start pose -- a read on whether the
                    real arm carries a payload (hand, F/T sensor) the model does
                    not.

Method.  Gravity compensation runs on the robot, so with a perfect model the
joint obeys

    tau_cmd = I*qddot + d*qdot + tau_c*sign(qdot) + b

and every term is identified by least squares from one recording.  Three
excitation phases feed it, and each exists because the others cannot do its job:

  1. **Breakaway ramps.**  Torque ramped slowly from zero until the joint just
     starts to move, in both directions.  The half-difference of the two
     breakaway torques is Coulomb friction; the half-sum is the gravity-comp
     error at that pose.  This is the one measurement that does not depend on the
     regression converging, so it is the cross-check on ``tau_c``, and its
     breakaway value is what sizes everything after it.
  2. **Sines**, for inertia.  A sinusoid keeps the joint oscillating *about its
     start angle* -- position excursion is bounded by ``A / (I * omega^2)``,
     which a torque staircase is not -- while exciting the qddot that inertia
     lives in.
  3. **Velocity sweeps**, for damping.  Sines alone cannot separate ``d*qdot``
     from ``tau_c*sign(qdot)``, because within one half-cycle the velocity has
     essentially a single magnitude and the two regressors then differ only by a
     scale factor.  Holding three deliberately different speeds under a
     proportional velocity loop is what breaks the tie.  See ``design_vsweeps``.

Safety.  Exactly one joint is ever driven; the other six are pinned by the
bridge's own 1 kHz impedance lock.  Four independent things bound what the driven
joint can do:

  - a position window ``Q_WINDOW`` around its start angle, and a ``Q_LIMIT_MARGIN``
    standoff from the robot's own joint limits -- either aborts the block;
  - a speed cap at ``DQ_CAP_FRAC`` of the joint's *rated* dq_max;
  - a torque cap of ``--tau-frac`` of its limit, refused above ``TAU_FRAC_MAX``;
  - the bridge's clamp on any hold target, 0.10 rad inside the real limits,
    which this side cannot override.

Every abort fades the torque out rather than dropping it, and hands the joint back
to the 1 kHz hold.  A robot fault or a wedged bridge stops the sweep.

Running it.  One bridge, one command, all seven joints:

    # terminal 1 -- no --lock-joints; the sweep sets the lock set per joint
    sudo LD_LIBRARY_PATH=~/rdk_install/lib \\
      ./rt_torque_bridge Rizon4s-063501 --enable --hold-slew 0.3

    # terminal 2
    uv run python -m mjlab.tasks.compliance.deploy.identify_rizon --joints 1-7

``--print-bridge-cmd`` prints that first line.  For each joint the sweep parks it
at an angle with room to be excited, hands it over from the bridge's hold to its
own torque, runs the blocks, then puts it back and re-locks it before moving on --
so the arm ends where it started and no joint is ever left at zero stiffness.
Only joints that actually lack room move at all (``--park mid`` centres them
instead); a park needing more than ``PARK_REFUSE`` is refused rather than made
unattended.

Each joint writes an .npz and the sweep merges them at the end.  ``--fit-only
run.npz`` re-fits one offline and ``--report a.npz b.npz ...`` re-merges, both
without a robot.

Read the per-block lines as they print.  A block that says anything other than
"completed" (or "breakaway captured" for a ramp) aborted on a guard and
contributed almost nothing, and a run whose blocks all aborted still produces a
confident-looking fit with a small residual -- it fits the little data it has
perfectly.  ``--report`` refuses such joints by name; trust that over the residual.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from mjlab.tasks.compliance.deploy.rt_bridge import shm_layout as L
from mjlab.tasks.compliance.deploy.rt_bridge.shm_client import _Shm

NUM_JOINTS = L.NUM_JOINTS
HZ = 100.0
DT = 1.0 / HZ

# Datasheet per-joint torque limit; the probe torques are fractions of this.
EFFORT_LIMIT = np.array([123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0])

# ─────────────────────────────────────────────────────────────────────────────
# Safety envelope.  Deliberately far tighter than the deploy guards: this needs
# excitation, not travel, and an identification run is not the place to explore
# the edge of the envelope.
# ─────────────────────────────────────────────────────────────────────────────
Q_WINDOW = 0.30  # rad from the start angle before a trial aborts
Q_WINDOW_MIN = 0.025  # rad; below this a joint has too little room to identify
# Speed cap, as a fraction of the joint's *rated* dq_max rather than one figure
# for the whole arm.  A single 0.80 rad/s cap is generous for the shoulder (rated
# 2.09) and crippling for the wrist (rated 4.89), which needs real speed before a
# torque above its breakaway friction produces a bounded excursion.
DQ_CAP_FRAC = 0.30
DQ_CAP_FLOOR = 0.50  # rad/s, so no joint ends up with an unusably tight cap
Q_LIMIT_MARGIN = 0.25  # rad, stay this far from the robot's own joint limits
SETTLE_DQ = 0.010  # rad/s; a ramp may not start until the joint is this still
BREAK_DQ = 0.030  # rad/s; above this a ramp counts the joint as having broken away
SETTLE_TIMEOUT_S = 3.0  # give up waiting for stillness after this long
TAU_FRAC_MAX = 0.40  # never command more than this fraction of the joint's limit
FADE_S = 0.15  # torque fade-out at the end of every trial
SETTLE_S = 0.60  # zero-torque settle between trials
# Between blocks the joint goes back to its start angle under the bridge's own
# 1 kHz hold, so there are no return-move gains here any more -- see
# ``Runner.return_to``.  ``--hold-slew`` on the bridge bounds how fast that is.
HOLD_SLEW_DEFAULT = 0.30  # rad/s; must match the bridge's --hold-slew
# Refuse to park a joint further than this without being asked again. A sweep runs
# unattended once it starts, so the one thing it must not do is decide by itself
# that a joint should swing most of its range.
PARK_REFUSE = math.radians(75.0)
# Rotor inertia the XML declares but never applies, used only as the prior that
# sizes the sine amplitudes before anything has been measured.
PRIOR_ROTOR = np.array([3.17, 3.17, 1.38, 1.38, 0.13, 0.13, 0.13])


def parse_joints(spec: str) -> list[int]:
  """``'1-7'`` / ``'2'`` / ``'1,3,5'`` -> zero-based joint indices."""
  out: list[int] = []
  for part in spec.replace(" ", "").split(","):
    if not part:
      continue
    if "-" in part:
      a, _, b = part.partition("-")
      lo, hi = int(a), int(b)
      if lo > hi:
        raise SystemExit(f"--joints range '{part}' runs backwards")
      out.extend(range(lo, hi + 1))
    else:
      out.append(int(part))
  bad = [j for j in out if not 1 <= j <= NUM_JOINTS]
  if bad:
    raise SystemExit(f"--joints must be 1..{NUM_JOINTS}, got {bad}")
  if not out:
    raise SystemExit("--joints selected nothing")
  return [j - 1 for j in dict.fromkeys(out)]  # dedupe, keep order


@dataclass
class Excitation:
  """One excitation block. ``kind`` is ``"ramp"``, ``"sine"`` or ``"vsweep"``."""

  kind: str
  tau_amp: float  # Nm: ramp target, sine amplitude, or vsweep torque cap
  duration: float  # s
  freq: float = 0.0  # Hz, sine only
  sign: float = 1.0  # ramp direction
  label: str = ""
  offset: float = 0.0  # Nm added to a sine, to cancel the gravity-comp residual
  # vsweep only: hold |dq| at ``vel`` with a proportional velocity loop, feeding
  # forward ``ff`` against Coulomb friction, reversing at ``x_rev`` from q0.
  vel: float = 0.0  # rad/s
  gain: float = 0.0  # Nm per rad/s
  ff: float = 0.0  # Nm
  x_rev: float = 0.0  # rad


@dataclass
class Recording:
  q: list[np.ndarray] = field(default_factory=list)
  dq: list[np.ndarray] = field(default_factory=list)
  tau_meas: list[np.ndarray] = field(default_factory=list)
  tau_ext: list[np.ndarray] = field(default_factory=list)
  tau_cmd: list[np.ndarray] = field(default_factory=list)
  block: list[int] = field(default_factory=list)

  def add(self, s: dict, tau: np.ndarray, block: int) -> None:
    self.q.append(s["q"][:NUM_JOINTS].copy())
    self.dq.append(s["dq"][:NUM_JOINTS].copy())
    self.tau_meas.append(s["tau"][:NUM_JOINTS].copy())
    self.tau_ext.append(s["tau_ext"][:NUM_JOINTS].copy())
    self.tau_cmd.append(tau.copy())
    self.block.append(block)

  def arrays(self) -> dict[str, np.ndarray]:
    return {
      "q": np.asarray(self.q),
      "dq": np.asarray(self.dq),
      "tau_meas": np.asarray(self.tau_meas),
      "tau_ext": np.asarray(self.tau_ext),
      "tau_cmd": np.asarray(self.tau_cmd),
      "block": np.asarray(self.block),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The model, for the rigid-body inertia and gravity regressors.
# ─────────────────────────────────────────────────────────────────────────────


class ModelRef:
  """Rigid-body ``M[j,j]`` and ``qfrc_bias[j]`` from the bare arm model.

  Built with armature forced to zero whatever the asset currently sets, because
  the quantity being measured here is the *total* inertia and the number this
  script reports is total minus rigid body.  Reading a model that already has
  armature applied would subtract it twice.

  The arm XML is loaded directly rather than through ``get_spec()``, which mounts
  the UMI jaw on link7: the payload the robot actually carries during a sweep is
  what the gravity-comp bias is there to measure, so putting a modelled one on
  the reference would fold it into the number and hide it.  Actuators are left
  as the XML has them -- ``mj_fullM`` and ``qfrc_bias`` do not read them.
  """

  def __init__(self) -> None:
    from mjlab.asset_zoo.robots.flexiv_three_hand.constants import (  # noqa: PLC0415
      FLEXIV_XML,
    )

    spec = mujoco.MjSpec.from_file(str(FLEXIV_XML))
    for key in list(spec.keys):
      spec.delete(key)
    for j in spec.joints:
      j.armature = 0.0
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    self._M = np.zeros((self.model.nv, self.model.nv))

  def at(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rigid-body M diagonal, qfrc_bias)`` at pose ``q``, both (7,)."""
    self.data.qpos[:NUM_JOINTS] = q[:NUM_JOINTS]
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)
    mujoco.mj_fullM(self.model, self._M, self.data.qM)
    return self._M.diagonal()[:NUM_JOINTS].copy(), self.data.qfrc_bias[
      :NUM_JOINTS
    ].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Excitation design.
# ─────────────────────────────────────────────────────────────────────────────


def design_ramps(tau_cap: float, room_pos: float, room_neg: float) -> list[Excitation]:
  """Breakaway ramps, one per direction that has room to move.

  A joint parked near a limit -- j4 sits 5 degrees under its upper stop at the
  tracking home -- has room on one side only, and driving it the other way just
  trips the limit-margin abort before it ever breaks friction.  Running the one
  direction that fits still identifies everything the regression needs; only the
  breakaway *cross-check* on Coulomb friction (which needs both directions to
  separate friction from the gravity-comp error) is lost.
  """
  out = []
  if room_pos >= Q_WINDOW_MIN:
    out.append(Excitation("ramp", tau_cap, 4.0, sign=+1.0, label="breakaway +"))
  if room_neg >= Q_WINDOW_MIN:
    out.append(Excitation("ramp", tau_cap, 4.0, sign=-1.0, label="breakaway -"))
  return out


def design_sines(
  inertia_guess: float,
  tau_break: float,
  tau_cap: float,
  x_max: float,
  v_max: float,
  offset: float = 0.0,
) -> list[Excitation]:
  """Sines that satisfy the position window AND the speed cap at the same time.

  The obvious sizing -- pick a frequency, then the amplitude that fills the
  position window -- cannot work, and the first hardware run proved it.  A sine
  of amplitude ``A`` at ``omega`` on inertia ``I`` gives

      position amplitude  X = A / (I omega^2)
      velocity amplitude  V = A / (I omega) = X * omega

  so fixing ``X`` at half the window makes ``V = X * omega`` regardless of
  inertia: 0.94 rad/s at 1 Hz, 2.36 at 2.5 Hz.  Against a 0.80 rad/s cap every
  block aborted on every joint, and the recordings came back with 7-16 samples
  each and a peak |dq| pinned exactly at the cap.

  So the amplitude is chosen first -- it has to clear the *measured* breakaway
  torque or the joint never moves at all -- and the frequency is then derived as
  the lowest one at which both bounds hold:

      X <= X_max  =>  omega >= sqrt(A / (I X_max))
      V <= V_max  =>  omega >= A / (I V_max)

  A margin above the binding bound keeps the live aborts as a backstop rather
  than a routine outcome.  Lower frequencies would be nicer for identification
  (more samples per cycle) but they are exactly what the two bounds forbid.

  ``offset`` cancels the gravity-compensation residual, and without it the whole
  bound calculation above is beside the point.  A zero-mean torque on a joint with
  a constant residual does not oscillate about its start angle -- it ratchets.  Each
  cycle the residual biases which half of the swing escapes the friction dead zone,
  so the joint walks one way, and the excursion is set by the drift rather than by
  the amplitude.  Measured against the fake bridge: joint 1's first sine was sized
  for 0.080 rad and aborted at 0.299 rad, of which 0.299 was monotone drift and
  none was oscillation.  The breakaway pair already measures the residual -- it is
  the half-*sum*, where Coulomb friction is the half-difference -- so feeding it
  forward here costs nothing and is what makes the sine excite rather than walk.
  """
  out: list[Excitation] = []
  # Amplitude must clear breakaway to move at all; two levels give the
  # regression a range of velocities to separate viscous from Coulomb.
  base = max(tau_break, 0.05 * tau_cap)
  for scale in (1.6, 2.6):
    amp = float(min(base * scale, tau_cap - abs(offset)))
    if amp <= 0.0:
      continue
    w_pos = math.sqrt(amp / max(inertia_guess * x_max, 1e-9))
    w_vel = amp / max(inertia_guess * v_max, 1e-9)
    omega = 1.25 * max(w_pos, w_vel)
    freq = float(np.clip(omega / (2.0 * math.pi), 0.5, 5.0))
    out.append(
      Excitation(
        "sine",
        amp,
        duration=max(2.0, 5.0 / freq),
        freq=freq,
        offset=offset,
        label=f"sine {freq:.2f} Hz @ {amp:.1f} Nm "
        f"{offset:+.2f} offset "
        f"(x~{amp / (inertia_guess * (2 * math.pi * freq) ** 2) * 1e3:.0f} mm-rad, "
        f"v~{amp / (inertia_guess * 2 * math.pi * freq):.2f} rad/s)",
      )
    )
  return out


VSWEEP_LEVELS = (0.25, 0.50, 0.85)  # fractions of the usable speed
VSWEEP_TAU_S = 0.10  # s, closed-loop velocity time constant the gain targets
VSWEEP_REV_FRAC = 0.40  # of the available room, before the sweep reverses


def design_vsweeps(
  inertia_guess: float,
  tau_break: float,
  tau_cap: float,
  x_rev: float,
  v_max: float,
) -> list[Excitation]:
  """Constant-speed holds at several speeds, to separate viscous from Coulomb.

  Sines alone cannot do it.  Inside one half-cycle the velocity has essentially
  one magnitude, so ``d*qdot`` and ``tau_c*sgn(qdot)`` differ only by a scale
  factor and the regression cannot tell them apart -- it splits the friction
  torque between them however the noise happens to fall.  The synthetic check is
  unambiguous: on ramps plus two sines the inertia comes back within 2-7% but the
  damping lands at 0.54 against a true 1.50, and on joint 3 it goes *negative*,
  with Coulomb picking up the difference.  Two sine amplitudes do not fix it
  either; the position window keeps their velocities within ~1.4x of each other.

  Holding a *deliberately different* speed for a while is what breaks the tie,
  because ``tau_c`` is the same at every speed and ``d*qdot`` is not.  Under a
  proportional velocity loop the joint settles near its target and the fit sees
  three well-separated velocity plateaus; ``qddot`` is ~0 there, so these blocks
  say almost nothing about inertia, which is why the sines stay.

  The loop reverses at ``x_rev`` from the start angle instead of running out of
  travel, so one block yields several plateaus in each direction.
  """
  gain = min(inertia_guess / VSWEEP_TAU_S, 0.5 * tau_cap / max(v_max, 1e-6))
  out: list[Excitation] = []
  for frac in VSWEEP_LEVELS:
    vel = frac * v_max
    out.append(
      Excitation(
        "vsweep",
        tau_amp=tau_cap,
        duration=max(2.0, 6.0 * x_rev / max(vel, 1e-6)),
        vel=vel,
        gain=gain,
        ff=tau_break,
        x_rev=x_rev,
        label=f"vsweep {vel:.2f} rad/s (gain {gain:.1f} Nm/(rad/s), ff {tau_break:.1f} Nm)",
      )
    )
  return out


def excitation_torque(
  exc: Excitation, t: float, dq: float = 0.0, direction: float = 1.0
) -> float:
  if exc.kind == "ramp":
    return exc.sign * exc.tau_amp * min(1.0, t / exc.duration)
  if exc.kind == "vsweep":
    target = direction * exc.vel
    return exc.gain * (target - dq) + exc.ff * math.copysign(1.0, target)
  # Fade the sine in over its first cycle.  Switching a sinusoid on at full
  # amplitude from rest excites the homogeneous solution as well as the particular
  # one, and the two add: the first swing reaches up to twice the steady-state
  # velocity amplitude.  That is not a detail -- against the fake bridge, joint 1's
  # second sine was sized for 0.47 rad/s and tripped the 0.63 rad/s cap on its
  # first swing, so the block contributed nothing.  A one-cycle envelope removes
  # the transient, and the regression does not care that the amplitude varies.
  envelope = min(1.0, t * exc.freq)
  return envelope * (exc.offset + exc.tau_amp * math.sin(2.0 * math.pi * exc.freq * t))


# ─────────────────────────────────────────────────────────────────────────────
# The run loop.
# ─────────────────────────────────────────────────────────────────────────────


ALL_LOCKED = (1 << NUM_JOINTS) - 1
PARK_TOL = 0.01  # rad, close enough to the park target
PARK_SETTLE_DQ = 0.02  # rad/s, and slow enough


def plan_park(
  q_now: np.ndarray, q_min: np.ndarray, q_max: np.ndarray, mid: bool = False
) -> np.ndarray:
  """Where each joint has to sit before it can be excited.

  A trial needs ``Q_WINDOW`` of travel on each side of the start angle, inside a
  ``Q_LIMIT_MARGIN`` standoff from the stops, so the legal start angles are

      [q_min + Q_LIMIT_MARGIN + Q_WINDOW,  q_max - Q_LIMIT_MARGIN - Q_WINDOW]

  and a joint outside that band cannot be identified where it stands -- j4 at the
  tracking home (139.7 deg, upper stop 159) has 0.087 rad above it and aborted
  every positive-going block before it moved.

  The default is the *nearest* legal angle, so a joint already with room does not
  move at all and one without it moves the minimum.  ``mid`` picks the centre of
  the legal band instead.  Geometric mid-range is deliberately not the default:
  on this arm it means j2 at 0 and j4 at 50 deg, which unfolds the arm to nearly
  horizontal -- a large swept volume and the worst gravity load on the shoulder,
  for no benefit over an angle that merely has room.
  """
  lo = q_min + Q_LIMIT_MARGIN + Q_WINDOW
  hi = q_max - Q_LIMIT_MARGIN - Q_WINDOW
  centre = 0.5 * (q_min + q_max)
  # A joint whose whole travel is under 2*(margin + window) has no legal band at
  # all; park it dead centre and let the live guards bound the trial.
  target = np.where(lo <= hi, np.clip(centre if mid else q_now, lo, hi), centre)
  # Final backstop against the limits themselves. The bounds have to be checked
  # for being inverted before clipping to them: a joint with under 2*margin of
  # travel gives lo > hi, and np.clip applies the lower bound last, so it would
  # return a target *outside* the joint's range -- below q_min for a short joint.
  # The bridge would refuse it, but this side should not ask.
  safe_lo = q_min + Q_LIMIT_MARGIN
  safe_hi = q_max - Q_LIMIT_MARGIN
  return np.where(safe_lo <= safe_hi, np.clip(target, safe_lo, safe_hi), centre)


class Runner:
  """Drives one joint while the bridge holds the other six at 1 kHz.

  Which joints the bridge holds, and about what angles, travel with every torque
  command (layout v2), so a whole seven-joint sweep runs in one process against
  one bridge instance.  The hold itself stays where it was -- in the 1 kHz RT
  loop, at its stiff gains -- because a 100 Hz Python PD would be a worse job of
  the one thing that keeps six joints still while the seventh is driven.
  """

  def __init__(
    self,
    shm: _Shm,
    joint: int,
    tau_cap: float,
    q_hold: np.ndarray,
    slew: float = 0.30,
    inertia_guess: float = 1.0,
  ):
    self.shm = shm
    self.j = joint  # zero-based
    self.tau_cap = tau_cap
    self.slew = slew
    self.inertia_guess = inertia_guess
    lim = shm.limits()
    self.q_min = lim["q_min"]
    self.q_max = lim["q_max"]
    self.dq_max = lim["dq_max"]
    self.dq_cap = max(DQ_CAP_FLOOR, DQ_CAP_FRAC * float(self.dq_max[joint]))
    # Hold target for all seven. Clamped here as well as in the bridge: this side
    # should never *ask* for something illegal, and the bridge should never honour
    # it if it does.
    self.q_hold = np.clip(
      np.asarray(q_hold, dtype=float).copy(),
      self.q_min + Q_LIMIT_MARGIN,
      self.q_max - Q_LIMIT_MARGIN,
    )
    # Everything held while parking; the joint under test is released later.
    self.lock_mask = ALL_LOCKED
    self.rec = Recording()
    self.block = 0
    self.breakaway: dict[str, float] = {}
    self.drifting = False
    self._last_tick = -1
    self._stale = 0

  def room(self, q0: float) -> tuple[float, float]:
    """Travel available above and below ``q0`` before an abort fires.

    The position window is not the only bound: a joint parked near a stop runs
    out of room against ``Q_LIMIT_MARGIN`` first.  j4 at the tracking home has
    0.087 rad above it and 4.1 rad below, so a symmetric window is simply wrong
    there -- it aborted every positive-going block before the joint had moved.
    """
    up = min(Q_WINDOW, (self.q_max[self.j] - Q_LIMIT_MARGIN) - q0)
    down = min(Q_WINDOW, q0 - (self.q_min[self.j] + Q_LIMIT_MARGIN))
    return float(max(up, 0.0)), float(max(down, 0.0))

  def await_still(self, seconds: float = SETTLE_TIMEOUT_S) -> bool:
    """Hold zero torque until the joint stops, or report that it will not.

    Zero residual under gravity compensation is zero *stiffness*, so a joint with
    any gravity-comp error drifts instead of stopping.  The first hardware run
    started its ramps while j3 and j6 were still moving, and the breakaway
    detector -- "the first torque at which |dq| exceeds 0.03" -- fired on the very
    first sample and recorded a breakaway of 1e-6 Nm.  A ramp that starts from
    motion measures nothing, so wait for stillness and say so when it never comes.
    """
    deadline = time.perf_counter() + seconds
    still = 0
    while time.perf_counter() < deadline:
      self._send(self._vec(0.0))
      s = self._read()
      still = still + 1 if abs(s["dq"][self.j]) < SETTLE_DQ else 0
      if still >= 10:  # 0.1 s of continuous stillness
        return True
      time.sleep(DT)
    return False

  def _vec(self, value: float) -> np.ndarray:
    """A full 7-vector with only the joint under test non-zero."""
    tau = np.zeros(NUM_JOINTS)
    tau[self.j] = float(np.clip(value, -self.tau_cap, self.tau_cap))
    return tau

  def _read(self) -> dict:
    s = self.shm.read_state()
    if s["tick"] == self._last_tick:
      self._stale += 1
      if self._stale > 5:
        raise RuntimeError("bridge stopped ticking — check its console")
    else:
      self._stale = 0
    self._last_tick = s["tick"]
    if s["fault"]:
      raise RuntimeError("robot reported a fault")
    return s

  def _check(self, s: dict, q0: float) -> str | None:
    q, dq = s["q"][self.j], s["dq"][self.j]
    if abs(q - q0) > Q_WINDOW:
      return f"position window: {abs(q - q0):.3f} > {Q_WINDOW} rad from start"
    if abs(dq) > self.dq_cap:
      return f"speed cap: |dq| {abs(dq):.2f} > {self.dq_cap:.2f} rad/s"
    if (
      q < self.q_min[self.j] + Q_LIMIT_MARGIN or q > self.q_max[self.j] - Q_LIMIT_MARGIN
    ):
      return f"within {Q_LIMIT_MARGIN} rad of the joint limit (q={q:.3f})"
    return None

  def _send(self, tau: np.ndarray) -> None:
    self.shm.write_command(tau, self.lock_mask, self.q_hold)

  def park_to(self, target: float, what: str = "park") -> bool:
    """Hold every joint and walk the joint under test to ``target``.

    This is the bridge's own stiff hold used as a position controller: the target
    it tracks slews toward what we ask at ``--hold-slew`` rad/s, so a 30 deg move
    is a position move rather than a torque step into the cap.  Returns whether
    the joint arrived and settled.
    """
    start = float(self._read()["q"][self.j])
    limited = float(
      np.clip(
        target, self.q_min[self.j] + Q_LIMIT_MARGIN, self.q_max[self.j] - Q_LIMIT_MARGIN
      )
    )
    if abs(limited - target) > 1e-9:
      print(
        f"    {what}: target {math.degrees(target):.1f} deg clipped to "
        f"{math.degrees(limited):.1f} deg by the {Q_LIMIT_MARGIN} rad limit margin"
      )
    self.lock_mask = ALL_LOCKED
    self.q_hold[self.j] = limited
    dist = abs(limited - start)
    if dist < PARK_TOL:
      # Still send once, so the bridge has the lock set and target for this joint.
      self._send(np.zeros(NUM_JOINTS))
      return True
    budget = dist / max(self.slew, 1e-6) + 4.0
    print(
      f"    {what}: {math.degrees(start):.1f} -> {math.degrees(limited):.1f} deg "
      f"({math.degrees(dist):.1f} deg, <= {budget:.1f} s) ... ",
      end="",
      flush=True,
    )
    deadline = time.perf_counter() + budget
    still = 0
    while time.perf_counter() < deadline:
      self._send(np.zeros(NUM_JOINTS))
      s = self._read()
      close = abs(float(s["q"][self.j]) - limited) < PARK_TOL
      slow = abs(float(s["dq"][self.j])) < PARK_SETTLE_DQ
      still = still + 1 if (close and slow) else 0
      if still >= 20:  # 0.2 s of continuous arrival
        print("arrived")
        return True
      time.sleep(DT)
    s = self._read()
    print(
      f"TIMEOUT at {math.degrees(float(s['q'][self.j])):.1f} deg "
      f"(dq {float(s['dq'][self.j]):+.3f})"
    )
    return False

  def release(self) -> None:
    """Hand the joint under test over from the bridge's hold to our torque.

    Only legal once the joint is parked and still: releasing it drops its hold
    torque to zero in one tick, and at the target that torque is already ~0, so
    the handover is a step of nothing.  Releasing a joint that is still moving
    would drop a real torque.
    """
    self.lock_mask = ALL_LOCKED & ~(1 << self.j)
    self._send(np.zeros(NUM_JOINTS))

  def _ramp_overshoot(self, exc: Excitation) -> float:
    """How far past breakaway the ramp has already climbed when motion is seen.

    A ramp cannot detect breakaway at breakaway.  Motion is only unambiguous once
    ``|qdot|`` clears ``BREAK_DQ``, and while the joint accelerates from rest the
    ramp keeps climbing.  With net torque ``rate*t`` on inertia ``I`` the joint
    reaches the threshold at ``t = sqrt(2*I*dq/rate)``, by which point the command
    has overshot the true breakaway by ``rate*t = sqrt(2*I*dq*rate)``.

    That is not a rounding error: against the fake bridge, joint 1's raw detection
    read 2.72 Nm on a joint whose true Coulomb friction was 1.60, and since the
    sine amplitudes are sized from this number every sine came out too big and
    aborted on the position window.  Subtracting the predicted overshoot brings the
    cross-check to within ~7% and the sines back inside their bounds.
    """
    rate = exc.tau_amp / max(exc.duration, 1e-6)
    return math.sqrt(2.0 * max(self.inertia_guess, 1e-6) * BREAK_DQ * rate)

  def _fade(self, from_value: float) -> None:
    n = max(1, int(FADE_S * HZ))
    for i in range(n, 0, -1):
      self._send(self._vec(from_value * i / n))
      time.sleep(DT)
    self._send(self._vec(0.0))

  def settle(self, seconds: float = SETTLE_S) -> None:
    for _ in range(int(seconds * HZ)):
      self._send(self._vec(0.0))
      self._read()
      time.sleep(DT)

  def run_block(self, exc: Excitation, q0: float) -> None:
    """One excitation block. Records throughout; aborts safely on any guard."""
    self.block += 1
    print(f"    [{self.block}] {exc.label or exc.kind} ... ", end="", flush=True)
    t0 = time.perf_counter()
    tau_now = 0.0
    moved_at: float | None = None
    reason = "completed"
    direction = 1.0
    reversals = 0
    while True:
      t = time.perf_counter() - t0
      if t >= exc.duration:
        break
      s = self._read()
      abort = self._check(s, q0)
      if abort is not None:
        reason = abort
        break
      if exc.kind == "vsweep":
        # Turn around well inside the window, so the block ends on its own clock
        # rather than on a guard. Only reverse once past the line and heading the
        # wrong way, or a plateau that overshoots slightly would chatter.
        offset = float(s["q"][self.j]) - q0
        if direction * offset > exc.x_rev:
          direction = -direction
          reversals += 1
      tau_now = excitation_torque(exc, t, float(s["dq"][self.j]), direction)
      tau = self._vec(tau_now)
      self._send(tau)
      self.rec.add(s, tau, self.block)
      # Breakaway: the first torque at which the joint is unambiguously moving.
      if exc.kind == "ramp" and moved_at is None and abs(s["dq"][self.j]) > BREAK_DQ:
        moved_at = tau[self.j]
      if exc.kind == "ramp" and moved_at is not None and abs(s["dq"][self.j]) > 0.20:
        reason = "moving (breakaway captured)"
        break
      time.sleep(DT)
    self._fade(tau_now)
    if exc.kind == "ramp":
      key = "pos" if exc.sign > 0 else "neg"
      if moved_at is None:
        print(f"no motion up to {self.tau_cap:.1f} Nm — raise --tau-frac")
      elif abs(moved_at) < 0.05 * self.tau_cap:
        # The joint was already moving when the ramp began, so this is not a
        # breakaway measurement -- it is the drift detector firing. Discard it
        # rather than recording a ~0 Nm friction that the report would believe.
        print(f"discarded: joint was already moving ({moved_at:+.3f} Nm)")
        self.drifting = True
      else:
        raw = float(moved_at)
        corrected = raw - math.copysign(self._ramp_overshoot(exc), raw)
        self.breakaway[key] = corrected
        print(
          f"breakaway {corrected:+.2f} Nm ({reason}; "
          f"{raw:+.2f} at detection, less {self._ramp_overshoot(exc):.2f} overshoot)"
        )
    elif exc.kind == "vsweep":
      print(f"{reason}, {reversals} reversals")
    else:
      print(reason)

  def return_to(self, q_target: float) -> bool:
    """Put the joint back where the trial started, then release it again.

    This used to be a Python PD at 100 Hz, gentle and torque-capped because it had
    to be -- it was the only thing available.  The bridge's 1 kHz hold with a
    slewed target does the same job stiffer and with the limit clamp underneath
    it, so between blocks the joint is a locked joint rather than one being nudged
    from userspace.
    """
    ok = self.park_to(q_target, what="return")
    self.release()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# The fit.
# ─────────────────────────────────────────────────────────────────────────────


DQ_MOVING = 0.02  # rad/s below which sign(qdot) is inside the friction dead zone
# Integration windows, in samples, longest first.  One length cannot serve every
# joint: a shoulder sweep holds a plateau for over a second, while a wrist sweep at
# 1 rad/s turns around every 0.24 s, so a single 0.6 s window either straddles a
# reversal on the wrist or wastes the shoulder's plateaus.  Taking all three scales
# and keeping whichever windows pass the uniformity test below covers both without
# a per-joint constant.
FIT_WINDOWS = (60, 30, 15)
# Keep a window only if its slowest sample is at least this fraction of its mean
# speed -- i.e. the joint held a roughly steady speed across it.
#
# This is the criterion that makes damping come out right, and it is not a tuning
# knob: ``tau_c*sgn(qdot)`` is exactly the part of the model that is wrong near zero
# velocity, where real friction is a Stribeck curve and stiction rather than a step.
# A small-amplitude sine spends most of every cycle there, and measured against the
# fake bridge those windows carry 5x the model error of a velocity plateau (0.069 vs
# 0.013 Nm.s rms on joint 1) and return damping of 0.748 against a true 1.50 -- while
# plateau windows alone return 1.482.  Unweighted, the sine windows outnumber the
# plateaus and drag the estimate down by 25-45%.  Requiring speed uniformity drops
# them on the grounds that the model does not describe them, which is the honest
# reason, and leaves inertia within 1% and damping within 3% on all seven joints.
FIT_SPEED_UNIFORMITY = 0.30


def fit_joint(arrays: dict[str, np.ndarray], joint: int, model: ModelRef) -> dict:
  """Identify ``I``, ``d``, ``tau_c`` and a constant offset by the integral method.

  The equation of motion is integrated over windows rather than differentiated
  per sample.  Both forms are algebraically the same fit, but numerically they are
  not: getting qddot out of a 100 Hz velocity signal means differentiating, and
  every differentiator either amplifies quantisation noise or attenuates the band
  the sines actually excite.  Validated against synthetic joints with known
  parameters, a Savitzky-Golay derivative recovered inertia 5% high and Coulomb
  friction 18% low -- the friction step at each velocity zero crossing gets
  smeared into qddot, so it is read as inertia.  Integrating

      I*[qdot] + d*∫qdot + tau_c*∫sgn(qdot) + b*T = ∫tau_cmd

  needs only the *difference* of the measured velocity across the window, and the
  integrals average the noise instead of amplifying it.

  ``b`` is the robot's own gravity-compensation error, which the bridge's on-robot
  compensation leaves behind.  An earlier version also fitted a gain on the
  model's ``qfrc_bias`` to catch a wrong payload, and that was a mistake: these
  trials move a joint by well under 0.1 rad, over which ``tau_grav(q)`` is very
  nearly constant and therefore collinear with ``b``.  On the first hardware run
  the pair blew up into a cancelling +/-, joint 3 reporting a gravity gain error of
  -90 with a -2.2 Nm bias, and dragged the inertia and damping with it.  Detecting
  a payload needs a large, slow sweep of the joint, which is a different
  experiment; do that separately rather than smuggling it in here.

  Windows are taken inside a block and never across one, at each length in
  ``FIT_WINDOWS`` and overlapping at a quarter of their length.  ``sgn(qdot)`` is
  zeroed inside the dead zone rather than dropping those samples, which keeps each
  window contiguous for the integrals.

  Because those windows overlap and repeat at several scales, the rows are strongly
  correlated and the reported standard errors are **optimistic** -- they treat a few
  hundred overlapping windows as a few hundred independent measurements.  Read them
  as relative: a damping standard error that is a large fraction of the damping
  itself means the run did not resolve it, but the absolute figure is not a
  confidence interval.
  """
  q, dq = arrays["q"], arrays["dq"]
  tau_cmd, block = arrays["tau_cmd"], arrays["block"]
  dqj, tauj = dq[:, joint], tau_cmd[:, joint]

  rb = np.array([model.at(q[i])[0][joint] for i in range(len(q))])
  sgn = np.sign(dqj) * (np.abs(dqj) > DQ_MOVING)

  rows: list[list[float]] = []
  ys: list[float] = []
  speeds: list[float] = []
  used = np.zeros(len(dqj), dtype=bool)
  for b in np.unique(block):
    idx = np.flatnonzero(block == b)
    for win in FIT_WINDOWS:
      if win < 10 or len(idx) < win + 2:
        continue
      for a in range(0, len(idx) - win, max(1, win // 4)):
        s = idx[a : a + win + 1]
        speed = np.abs(dqj[s])
        # Skip windows the joint spent mostly stationary: they carry no information
        # about inertia or damping and would let the constant terms absorb them.
        if np.mean(speed > DQ_MOVING) < 0.7:
          continue
        # And skip windows whose speed was not roughly steady -- see
        # FIT_SPEED_UNIFORMITY for why this one is load-bearing.
        if speed.min() < FIT_SPEED_UNIFORMITY * speed.mean():
          continue
        rows.append(
          [
            float(dqj[s[-1]] - dqj[s[0]]),
            float(np.trapezoid(dqj[s], dx=DT)),
            float(np.trapezoid(sgn[s], dx=DT)),
            win * DT,
          ]
        )
        # A left Riemann sum, not a trapezoid: every other signal here is a sampled
        # continuous one, but tau_cmd is *held* for the whole 10 ms tick, so its
        # exact integral is dt times the sum of the held values.  Using the
        # trapezoid on it instead is off by dt*(tau_end - tau_start)/2, which
        # sounds negligible and is not: at a velocity-sweep reversal the command
        # jumps by ~2*gain*vel, 19 Nm on joint 1, so the window picks up ~0.1 Nm.s
        # of torque that was never applied.  That error grows with the torque change
        # across the window, which is the one direction the damping coefficient sits
        # in, and it biased damping low by 25-55% on every joint in the synthetic
        # check -- inertia and Coulomb friction came back fine the whole time, which
        # is exactly why it took so long to find.
        ys.append(float(DT * tauj[s[:-1]].sum()))
        speeds.append(float(speed.mean()))
        used[s] = True

  if len(rows) < 8:
    raise SystemExit(
      f"only {len(rows)} usable integration windows — the joint barely moved.\n"
      f"Raise --tau-frac, or check that the bridge locked the other six joints "
      f"and not this one."
    )
  A, y = np.array(rows), np.array(ys)
  coef, *_ = np.linalg.lstsq(A, y, rcond=None)
  resid = y - A @ coef
  inertia, damping, tau_c, bias = (float(c) for c in coef)
  # Standard errors from the least-squares covariance, so each number arrives with
  # a statement of how much this particular run pins it down.  The tempting
  # cheaper proxy -- the ratio of the fastest to the slowest window speed, on the
  # theory that damping and Coulomb friction separate only through that spread --
  # is worse than useless: on the synthetic joint 4 it read 3.8x, its most
  # confident value of all seven, on the one run whose damping came back negative.
  # A joint pinned near its stop yields a handful of short blocks whose few
  # surviving windows happen to sit at very different speeds, which flatters the
  # spread and says nothing about the conditioning that matters.
  dof = max(len(rows) - A.shape[1], 1)
  sigma2 = float((resid**2).sum() / dof)
  try:
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.abs(np.diagonal(cov)))
  except np.linalg.LinAlgError:
    se = np.full(A.shape[1], np.inf)
  return {
    "joint": joint + 1,
    "inertia_total": inertia,
    "inertia_rigid_body": float(rb[used].mean()),
    "armature": inertia - float(rb[used].mean()),
    "damping": damping,
    "frictionloss": abs(tau_c),
    "bias": bias,
    "se_inertia": float(se[0]),
    "se_damping": float(se[1]),
    "se_frictionloss": float(se[2]),
    # Residual and signal are torque-*integrals*, so each is divided by its own
    # window duration to get back to Nm. That duration is column 3 of the design
    # matrix, which matters now that windows come at three different lengths --
    # dividing them all by one nominal length would understate the short windows.
    "resid_rms": float(np.sqrt(((resid / A[:, 3]) ** 2).mean())),
    "tau_rms": float(np.sqrt(((y / A[:, 3]) ** 2).mean())),
    "n_samples": int(used.sum()),
    "n_windows": len(rows),
  }


def print_fit(f: dict, breakaway: dict | None) -> None:
  j = f["joint"]
  print(f"\n--- joint {j} ---")
  print(f"  windows / samples used  : {f['n_windows']} / {f['n_samples']}")
  print(
    f"  effective inertia       : {f['inertia_total']:8.3f} kg m^2"
    f"   (model rigid body {f['inertia_rigid_body']:.3f})"
  )
  print(
    f"  => armature to set      : {f['armature']:8.3f} kg m^2"
    f"   (+/- {f['se_inertia']:.3f})"
  )
  print(
    f"  Coulomb friction        : {f['frictionloss']:8.3f} Nm"
    f"   (+/- {f['se_frictionloss']:.3f})"
  )
  if breakaway and "pos" in breakaway and "neg" in breakaway:
    bp, bn = breakaway["pos"], breakaway["neg"]
    print(
      f"     cross-check from ramps: {0.5 * (bp - bn):8.3f} Nm"
      f"   (breakaway {bp:+.2f} / {bn:+.2f})"
    )
    print(f"  gravity-comp error       : {0.5 * (bp + bn):8.3f} Nm at the start pose")
  se_d = f["se_damping"]
  loose = se_d > 0.5 * max(abs(f["damping"]), 1e-9)
  print(
    f"  viscous damping         : {f['damping']:8.3f} Nms/rad"
    f"   (+/- {se_d:.3f}{' — NOT resolved' if loose else ''})"
  )
  print(f"  gravity-comp error (fit): {f['bias']:8.3f} Nm")
  frac = f["resid_rms"] / max(f["tau_rms"], 1e-9)
  verdict = "good" if frac < 0.15 else ("usable" if frac < 0.30 else "POOR — see below")
  print(
    f"  fit residual            : {f['resid_rms']:.2f} Nm rms, {100 * frac:.0f}% of signal — {verdict}"
  )
  if frac >= 0.30:
    print(
      "     A poor fit usually means the excursion was too small to excite\n"
      "     qddot, or a neighbouring joint was not locked and its reaction\n"
      "     torque is in the measurement. Check the bridge's --lock-joints."
    )


def report(files: list[Path]) -> None:
  """Merge per-joint runs into the tuples to paste into constants.py."""
  model = ModelRef()
  rows: dict[int, dict] = {}
  for path in files:
    d = np.load(path, allow_pickle=True)
    joint = int(d["joint"])
    arrays = {k: d[k] for k in ("q", "dq", "tau_meas", "tau_ext", "tau_cmd", "block")}
    f = fit_joint(arrays, joint, model)
    bk = {
      k: float(v)
      for k, v in zip(("pos", "neg"), d["breakaway"], strict=True)
      if not np.isnan(v)
    }
    rows[joint + 1] = f
    print_fit(f, bk)

  missing = [j for j in range(1, NUM_JOINTS + 1) if j not in rows]
  print("\n" + "=" * 72)
  if missing:
    print(f"NOTE: no data for joint(s) {missing} — those entries are left at 0.0.")

  def tup(key: str, fmt: str = "{:.2f}") -> str:
    return (
      "("
      + ", ".join(fmt.format(rows[j][key] if j in rows else 0.0) for j in range(1, 8))
      + ")"
    )

  # A run that barely moved the joint fits the little data it has almost
  # perfectly, so a small residual is no evidence the numbers mean anything. The
  # first hardware pass came back at 2-5% residual with rotor inertias of -1.16
  # and -1.05 kg m^2 and damping of 12 Nms/rad, off 9 to 35 windows. Negative
  # rotor inertia is impossible, so say so instead of offering it to be pasted.
  # The gate counts *samples*, not windows: windows overlap and repeat at three
  # scales, so their count inflates with no new information behind it, while the
  # number of distinct samples the fit drew on does not. A healthy run against the
  # fake bridge used 1200-1500 samples per joint; the first hardware pass, whose
  # blocks all aborted, used 52 to 303.
  bad = [
    j
    for j in sorted(rows)
    if rows[j]["armature"] < 0.0
    or rows[j]["n_samples"] < 400
    or rows[j]["se_inertia"] > 0.3 * max(rows[j]["inertia_total"], 1e-9)
  ]
  if bad:
    print(
      f"\nDO NOT USE joint(s) {bad}. Each is either physically impossible (a\n"
      "negative rotor inertia), built on under 400 samples, or has a standard error\n"
      "over 30% of the inertia itself. That means the excitation blocks aborted\n"
      "before they did anything, not that the joint is unusual -- re-run those\n"
      "joints and check each block reports 'completed'."
    )

  print("\nPaste into asset_zoo/robots/flexiv_three_hand/constants.py:\n")
  print(f"FLEXIV_ARM_ARMATURE: tuple[float, ...] = {tup('armature')}")
  print(f"FLEXIV_ARM_FRICTIONLOSS: tuple[float, ...] = {tup('frictionloss')}")
  print(f"FLEXIV_ARM_DAMPING: tuple[float, ...] = {tup('damping', '{:.3f}')}")
  print(
    "\nGravity-comp error left over by the robot: "
    + ", ".join(
      f"j{j}={rows[j]['bias']:+.2f}" if j in rows else f"j{j}=?" for j in range(1, 8)
    )
    + " Nm"
  )
  print(
    "Those are the robot's own gravity-compensation errors at each start pose.\n"
    "A few tenths of a Nm is normal; a whole Nm on a wrist joint means the model\n"
    "is missing the tool's mass, and the arm joints' inertia should not be\n"
    "trusted until it is added."
  )
  weak = [
    j
    for j in sorted(rows)
    if rows[j]["se_damping"] > 0.5 * max(abs(rows[j]["damping"]), 1e-9)
  ]
  if weak:
    print(
      f"\nDamping on joint(s) {weak} has a standard error over half its own value,\n"
      "so it is not resolved — it trades off against Coulomb friction. The\n"
      "velocity-sweep blocks are what resolve it, so this usually means they\n"
      "aborted early on the position window: check that joint had room to move,\n"
      "and re-pose it toward the middle of its range if it did not."
    )


# ─────────────────────────────────────────────────────────────────────────────


def bridge_cmd(slew: float = HOLD_SLEW_DEFAULT) -> str:
  """The one bridge invocation the whole sweep needs.

  No ``--lock-joints``: the sweep sets the lock set at runtime, one joint at a
  time, through the command block.
  """
  return (
    "sudo LD_LIBRARY_PATH=$HOME/rdk_install/lib \\\n"
    "  ~/compliance-grasping/src/mjlab/tasks/compliance/deploy/rt_bridge/build/"
    "rt_torque_bridge \\\n"
    f"  Rizon4s-063501 --enable --hold-slew {slew}"
  )


def identify_one(
  shm: _Shm,
  joint: int,
  tau_frac: float,
  model: ModelRef,
  q_park: np.ndarray,
  slew: float,
  out: Path,
) -> dict | None:
  """Park, excite and re-lock one joint. Returns its fit, or None if unusable.

  The other six joints are held by the bridge throughout, including while this
  one is parked; only between ``release()`` and the final re-lock is any joint
  free, and even then it is the single joint under test with a torque this loop
  guards on every sample.
  """
  tau_cap = tau_frac * EFFORT_LIMIT[joint]
  runner = Runner(shm, joint, tau_cap, q_park, slew=slew)
  q_entry = float(runner._read()["q"][joint])

  print(f"\n{'=' * 72}\nJoint {joint + 1}\n{'=' * 72}")
  print(
    f"  joint limits         : {math.degrees(runner.q_min[joint]):.0f} .. "
    f"{math.degrees(runner.q_max[joint]):.0f} deg"
  )
  print(f"  at entry             : {math.degrees(q_entry):.1f} deg")

  # Park first: everything downstream is sized from the *parked* pose, so the
  # room, the inertia prior and the gravity term all have to be read after the
  # move rather than before it.
  if not runner.park_to(float(q_park[joint]), what="park"):
    print("  SKIPPED: could not park this joint; leaving it locked where it is.")
    return None

  s0 = runner._read()
  q0 = float(s0["q"][joint])
  rb0, grav0 = model.at(s0["q"])
  # The excursion sizing needs a guess at the total inertia. Rigid body plus the
  # XML's declared rotor value is the best prior available before measuring; it
  # only sets the sine amplitudes, and the live window abort is the real bound.
  inertia_guess = float(rb0[joint] + PRIOR_ROTOR[joint])
  # Read after parking, because the rigid-body term depends on the pose. The ramp's
  # overshoot correction needs it too.
  runner.inertia_guess = inertia_guess
  room_pos, room_neg = runner.room(q0)
  x_max = 0.5 * max(room_pos, room_neg)
  ramps = design_ramps(tau_cap, room_pos, room_neg)

  print(f"  parked at            : {math.degrees(q0):.1f} deg")
  print(
    f"  probe torque cap     : {tau_cap:.1f} Nm ({tau_frac:.0%} of "
    f"{EFFORT_LIMIT[joint]:.0f})"
  )
  print(f"  model rigid-body I   : {rb0[joint]:.3f} kg m^2")
  print(f"  model gravity here   : {grav0[joint]:+.2f} Nm")
  print(f"  inertia prior (sizing): {inertia_guess:.3f} kg m^2")
  print(
    f"  travel available     : {room_pos:.3f} rad up / {room_neg:.3f} rad down "
    f"(window {Q_WINDOW}, limit margin {Q_LIMIT_MARGIN})"
  )
  print(
    f"  abort if             : |q-q0| > {Q_WINDOW} rad, |dq| > "
    f"{runner.dq_cap:.2f} rad/s ({DQ_CAP_FRAC:.0%} of rated "
    f"{runner.dq_max[joint]:.2f}), within {Q_LIMIT_MARGIN} rad of a limit, or fault"
  )
  if x_max < Q_WINDOW_MIN:
    print(
      f"  SKIPPED: only {room_pos:.3f} rad above and {room_neg:.3f} rad below even "
      f"after parking — this joint's travel is too short to excite."
    )
    return None
  if len(ramps) < 2:
    print(
      "  NOTE: only one direction has room, so the breakaway cross-check on\n"
      "  Coulomb friction is unavailable (it needs both directions to separate\n"
      "  friction from the gravity-comp error). The regression still identifies\n"
      "  every parameter; it just loses its independent second opinion."
    )
  print(
    f"  blocks               : {len(ramps)} ramps, 2 sines (inertia), "
    f"{len(VSWEEP_LEVELS)} velocity sweeps (damping)\n"
  )

  # Hand this joint over from the bridge's hold to our torque. Everything from
  # here to the finally: block runs with joint `joint` released.
  runner.release()
  try:
    # Ramps first: their breakaway torque is what the sine amplitudes are sized
    # from. A sine below breakaway does not move the joint at all, and one far
    # above it runs straight into the speed cap.
    for exc in ramps:
      if not runner.await_still():
        print("    (joint will not settle at zero torque — it is drifting)")
      runner.run_block(exc, q0)
      runner.return_to(q0)
    # With both directions the pair separates cleanly: Coulomb friction is the
    # half-difference, the gravity-comp residual the half-sum. With only one the
    # two are entangled in that single number, so use it as the friction estimate
    # and assume no residual -- the position window is then the only thing keeping
    # the sine from walking.
    bk = runner.breakaway
    if "pos" in bk and "neg" in bk:
      tau_break = 0.5 * (bk["pos"] - bk["neg"])
      tau_offset = 0.5 * (bk["pos"] + bk["neg"])
    elif bk:
      tau_break = abs(next(iter(bk.values())))
      tau_offset = 0.0
      print("    One direction only: no gravity-comp residual estimate for the sines.")
    else:
      tau_break = 0.15 * tau_cap
      tau_offset = 0.0
      print(
        f"    No usable breakaway; sizing the sines from {tau_break:.2f} Nm "
        f"(15% of the cap) instead."
      )
    blocks = design_sines(
      inertia_guess, tau_break, tau_cap, x_max, runner.dq_cap, offset=tau_offset
    ) + design_vsweeps(
      inertia_guess,
      tau_break,
      tau_cap,
      VSWEEP_REV_FRAC * max(room_pos, room_neg),
      0.8 * runner.dq_cap,
    )
    for exc in blocks:
      if not runner.await_still():
        print("    (joint will not settle at zero torque — it is drifting)")
      runner.run_block(exc, q0)
      runner.return_to(q0)
  finally:
    # However this ends -- finished, guard abort, Ctrl-C, bridge death -- fade the
    # torque and give the joint back to the 1 kHz hold before returning. A joint
    # left released sits at zero stiffness and drifts.
    runner._fade(0.0)
    runner.park_to(q_entry, what="restore")
    runner.lock_mask = ALL_LOCKED
    runner._send(np.zeros(NUM_JOINTS))

  arrays = runner.rec.arrays()
  if len(arrays["q"]) < 50:
    print(f"  only {len(arrays['q'])} samples recorded — nothing written.")
    return None
  np.savez(
    out,
    joint=np.array(joint),
    dt=np.array(DT),
    tau_cap=np.array(tau_cap),
    breakaway=np.array(
      [runner.breakaway.get("pos", np.nan), runner.breakaway.get("neg", np.nan)]
    ),
    # savez is typed (file, *args, allow_pickle=..., **kwds), so a **splat reads
    # as a candidate for the bool keyword.
    **arrays,  # type: ignore[arg-type]
  )
  print(f"\n  {len(arrays['q'])} samples written to {out}")
  f = fit_joint(arrays, joint, model)
  print_fit(f, runner.breakaway)
  return f


def sweep(
  joints: list[int],
  tau_frac: float,
  out_dir: Path,
  slew: float,
  park_mid: bool,
  confirm: bool,
) -> None:
  """Identify every requested joint in one pass, against one bridge instance."""
  model = ModelRef()
  shm = _Shm()
  try:
    lim = shm.limits()
    q_min, q_max = lim["q_min"], lim["q_max"]
    q_start = shm.read_state()["q"][:NUM_JOINTS].copy()
    q_park = plan_park(q_start, q_min, q_max, mid=park_mid)

    print("\nSystem identification sweep")
    print(f"  joints               : {[j + 1 for j in joints]}")
    print(f"  probe torque         : {tau_frac:.0%} of each joint's limit")
    print(f"  hold slew            : {slew} rad/s (must match the bridge's)")
    print(f"  output               : {out_dir}/sysid_j<N>.npz")
    print(
      f"  park strategy        : {'centre of the legal band' if park_mid else 'nearest legal angle'}"
    )
    print("\n  joint   now      park     move    legal band (deg)")
    for j in range(NUM_JOINTS):
      lo = q_min[j] + Q_LIMIT_MARGIN + Q_WINDOW
      hi = q_max[j] - Q_LIMIT_MARGIN - Q_WINDOW
      mark = " <-" if j in joints else "   "
      print(
        f"  j{j + 1}   {math.degrees(q_start[j]):7.1f}  {math.degrees(q_park[j]):7.1f}  "
        f"{math.degrees(q_park[j] - q_start[j]):+7.1f}   "
        f"[{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]{mark}"
      )
    moved = [j + 1 for j in joints if abs(q_park[j] - q_start[j]) > PARK_TOL]
    print(
      f"\n  Joints that will be moved before testing: {moved or 'none'}.\n"
      f"  Each is moved alone, under the bridge's 1 kHz hold at {slew} rad/s, and\n"
      f"  put back afterwards. The other six are held throughout."
    )
    biggest = max((abs(q_park[j] - q_start[j]) for j in joints), default=0.0)
    if biggest > PARK_REFUSE:
      raise SystemExit(
        f"Refusing: parking needs a {math.degrees(biggest):.0f} deg move on some "
        f"joint, over the {math.degrees(PARK_REFUSE):.0f} deg this script will make "
        f"unattended. Check the arm is where you think it is, or move it closer to "
        f"a working pose with home_rizon.py first."
      )

    if confirm and input("\nStart on the REAL arm? type 'go': ").strip() != "go":
      print("Aborted.")
      return

    out_dir.mkdir(parents=True, exist_ok=True)
    done: list[int] = []
    for j in joints:
      try:
        f = identify_one(
          shm, j, tau_frac, model, q_park, slew, out_dir / f"sysid_j{j + 1}.npz"
        )
      except KeyboardInterrupt:
        # identify_one's finally: has already re-locked this joint.
        print("\n  interrupted by the operator — stopping the sweep here.")
        break
      except RuntimeError as e:
        # A fault or a dead bridge is not recoverable by moving to the next joint.
        print(f"\n  ABORTED on joint {j + 1}: {e}")
        break
      if f is not None:
        done.append(j + 1)

    print(f"\n{'=' * 72}")
    print(f"Collected: {done or 'nothing'}")
    if done:
      print(f"\nMerging {len(done)} joint(s):")
      report([out_dir / f"sysid_j{j}.npz" for j in done])
  finally:
    # Leave every joint held about wherever it ended up, then let go of the
    # segment. The bridge keeps holding until the operator stops it.
    try:
      q_now = shm.read_state()["q"][:NUM_JOINTS].copy()
      shm.write_command(np.zeros(NUM_JOINTS), ALL_LOCKED, q_now)
    except Exception as e:  # noqa: BLE001 — best effort on the way out
      print(f"  (could not re-lock on exit: {e})")
    shm.close()


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "--joints",
    default="1-7",
    help="which joints to identify: '1-7', '2', '1,3,5' (default: all seven)",
  )
  ap.add_argument(
    "--tau-frac",
    type=float,
    default=0.25,
    help=f"probe torque as a fraction of the joint's limit (max {TAU_FRAC_MAX})",
  )
  ap.add_argument(
    "--out-dir", type=Path, default=Path("."), help="where the .npz files go"
  )
  ap.add_argument(
    "--hold-slew",
    type=float,
    default=HOLD_SLEW_DEFAULT,
    help=f"rad/s the bridge slews a hold target; must match the bridge's "
    f"--hold-slew (default {HOLD_SLEW_DEFAULT})",
  )
  ap.add_argument(
    "--park",
    choices=("nearest", "mid"),
    default="nearest",
    help="park each joint at the nearest angle with room to be excited (default) "
    "or at the centre of its legal band",
  )
  ap.add_argument("--fit-only", type=Path, default=None, help="re-fit a saved run")
  ap.add_argument("--report", type=Path, nargs="+", default=None, help="merge runs")
  ap.add_argument(
    "--print-bridge-cmd",
    action="store_true",
    help="print the single bridge command the sweep needs, and exit",
  )
  ap.add_argument("--yes", action="store_true", help="skip the pre-motion prompt")
  args = ap.parse_args()

  if args.print_bridge_cmd:
    print(bridge_cmd(args.hold_slew))
    return
  if args.report:
    report(list(args.report))
    return
  if args.fit_only:
    d = np.load(args.fit_only, allow_pickle=True)
    arrays = {k: d[k] for k in ("q", "dq", "tau_meas", "tau_ext", "tau_cmd", "block")}
    bk = {
      k: float(v)
      for k, v in zip(("pos", "neg"), d["breakaway"], strict=True)
      if not np.isnan(v)
    }
    print_fit(fit_joint(arrays, int(d["joint"]), ModelRef()), bk)
    return

  if args.tau_frac > TAU_FRAC_MAX:
    raise SystemExit(f"--tau-frac above {TAU_FRAC_MAX} is refused for this script")
  joints = parse_joints(args.joints)
  sweep(
    joints,
    args.tau_frac,
    args.out_dir,
    args.hold_slew,
    args.park == "mid",
    confirm=not args.yes,
  )


if __name__ == "__main__":
  main()
