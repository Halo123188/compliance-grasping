"""Route-B real-robot deployment skeleton for the Stage-1 compliance policy.

Target checkpoint
-----------------
``logs/rsl_rl/compliance_reach_flexiv/2026-07-23_16-07-17_bare`` — the task
``Mjlab-Compliance-Reach-Flexiv-Bare-Full`` (bare Rizon 4S, ``ablation="full"``):
a GRU actor that, at 100 Hz, emits a 7-D Cartesian-impedance *parameter* vector
``[Δx_ref(3), logK(3), logK_rot(1)]``.  It never emits joint torques.

What "route B" means
--------------------
Instead of reproducing the 1 kHz analytical impedance law ourselves (route A,
``RT_JOINT_TORQUE``), we hand the inner loop to Flexiv's built-in Cartesian
impedance controller (``RT_/NRT_CARTESIAN_MOTION_FORCE``).  Each policy step we:

  * assemble the 37-D observation exactly as the sim did,
  * run the ONNX actor (normalization is baked into the .onnx; GRU state carried),
  * decode the raw action into a target TCP pose + per-axis Cartesian stiffness,
  * command that pose+stiffness to the robot.

This is the fastest, safest way onto the hardware.  It is NOT byte-identical to
sim: Flexiv's impedance dynamics (damping law, virtual mass, null-space) differ
from the ``D = 2ζ√(K·m_eff)``, ``m_eff = 2`` law the policy trained against, so
expect a sim-to-real gap in the *feel* of the yielding.  Use it to validate the
observation pipeline, frames, normalization and GRU wiring first; move to route A
for a faithful reproduction.

RDK version
-----------
The adapter targets **Flexiv RDK v1.9.x** (flat, single-arm API), the line
compatible with the Rizon 4S — RDK v2.x dropped that model (only ``Enlight``/
``MICO`` are accepted) and uses an incompatible joint-group API.  Install with
``uv pip install --python .venv-deploy "flexivrdk==1.9.0"``.

The script imports no ``mjlab`` code (only numpy / onnxruntime / flexivrdk), so
run it by **file path** from a dedicated venv, not via ``-m mjlab...``.  Sanity-
check the policy side with no robot attached:

    .venv-deploy/bin/python \\
      src/mjlab/tasks/compliance/deploy/deploy_rizon_route_b.py --dry-run

On the robot, ``--probe`` first to print the live ``robot.states()`` fields.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────────────────────
# Contract constants — read straight from the run's env.yaml / the task code.
# Do not "clean these up": they define the exact I/O the network was trained on.
# ─────────────────────────────────────────────────────────────────────────────

ONNX_PATH = Path(
  "logs/rsl_rl/compliance_reach_flexiv/2026-07-23_16-07-17_bare/"
  "2026-07-23_16-07-17_bare.onnx"
)

# HOME_KEYFRAME arm joints (constants.py): joint_pos_rel = q - Q_DEFAULT.
Q_DEFAULT = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0], dtype=np.float32)

# CartesianImpedanceActionCfg for this run (env.yaml lines ~896–920).
DELTA_POS_SCALE = 0.03  # m, anchor mode: x_eq = goal + tanh(a)·scale
K_RANGE = (15.0, 2000.0)  # N/m, per-axis translational stiffness
K_ROT_RANGE = (5.0, 200.0)  # Nm/rad, rotational stiffness (learn_rot_stiffness)
# Damping ratios fed to Flexiv as Z_x. Flexiv restricts Z_x to [0.3, 0.8], so
# the sim's rotational ζ=1.0 is capped at 0.8 (a route-B sim-to-real gap).
DAMPING_RATIO = 0.8  # translational ζ (sim damping_ratio); at Flexiv's max
ROT_DAMPING_RATIO = 0.8  # sim rot_damping_ratio is 1.0; clamped to Flexiv max
Z_X_RANGE = (0.3, 0.8)  # Flexiv-allowed Cartesian damping-ratio range
EFFECTIVE_MASS = 2.0  # only used by route A

# Safety clamp on the contact wrench the impedance controller may exert.
# [fx,fy,fz (N), mx,my,mz (Nm)]. Tune down for the first hardware runs.
MAX_CONTACT_WRENCH = [60.0, 60.0, 60.0, 15.0, 15.0, 15.0]

# Cap on the NRT motion generator's linear speed (m/s) chasing target jumps.
# Conservative for first runs; RDK default is 0.5.
MAX_LINEAR_VEL = 0.25
# Cap on the NRT generator's linear acceleration (m/s^2). Lower = softer chase of
# the 10 ms-updated target/stiffness => less jitter at hold. RDK default is 2.0.
MAX_LINEAR_ACC = 1.0
# EMA smoothing of the policy command (target + stiffness), in [0, 1). 0 = off;
# higher = smoother. filtered = (1-s)*new + s*filtered. Tames route-B jitter from
# the policy re-issuing a different target/K every step.
CMD_SMOOTH = 0.0
# Floor on per-axis translational stiffness (N/m). Keeps the arm from going so
# slack at hold that gravity/noise makes it drift. None = use the policy's value.
K_TRANS_FLOOR: float | None = None

# grasp_site (bare) = link7 flange + 0.05 m along link7's local +z (tool axis).
# The RDK TCP MUST be configured to this point, or ee_pos / goal_error are wrong.
BARE_TCP_OFFSET_LINK7 = np.array([0.0, 0.0, 0.05], dtype=np.float32)

DECIMATION = 10  # 100 Hz policy over a nominal 1 kHz inner loop
POLICY_HZ = 100.0
POLICY_DT = 1.0 / POLICY_HZ

OBS_DIM = 37
ACT_DIM = 7
GRU_HIDDEN = 128

_LOG_K_LO = math.log(K_RANGE[0])
_LOG_K_SPAN = math.log(K_RANGE[1]) - math.log(K_RANGE[0])
_LOG_KR_LO = math.log(K_ROT_RANGE[0])
_LOG_KR_SPAN = math.log(K_ROT_RANGE[1]) - math.log(K_ROT_RANGE[0])


# ─────────────────────────────────────────────────────────────────────────────
# [POLICY] Observation assembly — must match reach_env_cfg.py / observations.py.
#
#   layout (37): q_rel(7), dq(7), tau_meas(7), ee_pos(3), ee_vel(3),
#                goal_error(3), last_action(7)
# All quantities are in the robot base frame (== sim env origin; base at origin).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobotState:
  """The minimal real-robot signals the observation needs."""

  q: np.ndarray  # (7,) joint position, rad
  dq: np.ndarray  # (7,) joint velocity, rad/s
  tau_meas: np.ndarray  # (7,) measured joint torque, Nm  (see calibration note)
  ee_pos: np.ndarray  # (3,) TCP (grasp_site) position in base frame, m
  ee_vel: np.ndarray  # (3,) EE linear velocity in base frame, m/s
  ee_quat: np.ndarray  # (4,) TCP orientation, wxyz (for the orientation lock)


def build_observation(
  st: RobotState, goal_pos: np.ndarray, last_action: np.ndarray
) -> np.ndarray:
  """Assemble the 37-D observation. Feed RAW (normalization is inside the ONNX)."""
  q_rel = st.q - Q_DEFAULT
  goal_error = goal_pos - st.ee_pos  # x_g - x_ee, world/base frame
  obs = np.concatenate(
    [
      q_rel,  # 7
      st.dq,  # 7   (joint_vel_rel; default dq is 0)
      st.tau_meas,  # 7
      st.ee_pos,  # 3
      st.ee_vel,  # 3
      goal_error,  # 3
      last_action,  # 7   raw (pre-tanh) previous action
    ]
  ).astype(np.float32)
  assert obs.shape == (OBS_DIM,), obs.shape
  return obs


# ─────────────────────────────────────────────────────────────────────────────
# [POLICY] Action decode — must match CartesianImpedanceAction.process_actions.
# The ONNX outputs the RAW (pre-tanh) action; we tanh-squash and map here, and
# we cache the raw output as next step's last_action (== the sim's raw_action).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Command:
  target_pos: np.ndarray  # (3,) x_eq = goal + Δx_ref, base frame, m
  target_quat: np.ndarray  # (4,) locked orientation, wxyz
  stiffness_trans: np.ndarray  # (3,) per-axis N/m
  stiffness_rot: float  # Nm/rad


def decode_action(
  raw_action: np.ndarray, goal_pos: np.ndarray, lock_quat: np.ndarray
) -> Command:
  s = np.tanh(raw_action)
  delta_pos = s[0:3] * DELTA_POS_SCALE
  log_k = _LOG_K_LO + 0.5 * (s[3:6] + 1.0) * _LOG_K_SPAN
  k_trans = np.exp(log_k)
  log_kr = _LOG_KR_LO + 0.5 * (s[6] + 1.0) * _LOG_KR_SPAN
  k_rot = float(np.exp(log_kr))
  return Command(
    target_pos=goal_pos + delta_pos,
    target_quat=lock_quat,  # orientation is locked to the start pose
    stiffness_trans=k_trans,
    stiffness_rot=k_rot,
  )


# ─────────────────────────────────────────────────────────────────────────────
# [POLICY] ONNX actor with GRU state.
# ─────────────────────────────────────────────────────────────────────────────


class Actor:
  def __init__(self, onnx_path: Path):
    self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = {i.name for i in self.sess.get_inputs()}
    assert {"obs", "h_in"} <= names, f"unexpected ONNX inputs: {names}"
    self.reset()

  def reset(self) -> None:
    self.h = np.zeros((1, 1, GRU_HIDDEN), dtype=np.float32)

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    out = self.sess.run(
      ["actions", "h_out"],
      {"obs": obs[None, :], "h_in": self.h},
    )
    actions, self.h = out
    return actions[0].astype(np.float32)  # (7,) raw action


def _quat_rotate_vec(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate 3-vector ``v`` by wxyz quaternion ``q`` (``v' = R(q)·v``)."""
  w, x, y, z = (float(c) for c in q_wxyz)
  vx, vy, vz = (float(c) for c in v)
  # t = 2·(q_xyz × v);  v' = v + w·t + q_xyz × t
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  return np.array(
    [
      vx + w * tx + (y * tz - z * ty),
      vy + w * ty + (z * tx - x * tz),
      vz + w * tz + (x * ty - y * tx),
    ],
    dtype=np.float32,
  )


# ─────────────────────────────────────────────────────────────────────────────
# Flexiv adapter — aligned to RDK v2.1.0 (introspected on this machine).
# The 1.9.x control API is flat / single-arm (no JointGroup, no NrtCartesianCmd):
#   * states() -> RobotStates struct (attributes, NOT a dict).
#   * enable = Enable(); pose = [x,y,z,qw,qx,qy,qz]; TCP velocity is `tcp_vel`.
#   * SendCartesianMotionForce(pose, wrench=0, velocity=0, max_linear_vel, ...).
#   * SetCartesianImpedance(K_x, Z_x); SetMaxContactWrench(w).
#   * Only NRT_CARTESIAN_MOTION_FORCE exists in 1.9 (no RT variant).
# 1.9.x is the RDK line compatible with the Rizon 4S (2.x dropped that model).
# Run --probe to dump the live states() for your robot.
# ─────────────────────────────────────────────────────────────────────────────


class RizonAdapter:
  """Isolates all Flexiv RDK v1.9.x (flat, single-arm) calls for a Rizon 4S."""

  def __init__(
    self,
    robot_sn: str,
    tcp_offset: np.ndarray | None = None,
    max_contact_wrench: list[float] | None = None,
    max_linear_vel: float = MAX_LINEAR_VEL,
    max_linear_acc: float = MAX_LINEAR_ACC,
    k_trans_floor: float | None = K_TRANS_FLOOR,
  ):
    import flexivrdk  # noqa: PLC0415  (optional dep, only needed on the robot)

    self._rdk = flexivrdk
    self.robot = flexivrdk.Robot(robot_sn)  # 1.9: Robot(serial_number)
    self.Mode = flexivrdk.Mode

    # Fault -> enable -> wait operational. Enable needs the E-stop released
    # (and, in manual mode, the enabling device held).
    if self.robot.fault():
      self.robot.ClearFault()
    self.robot.Enable()
    while not self.robot.operational():
      time.sleep(0.1)

    # TCP handling. Preferred: configure the tool in Flexiv Elements so
    # states().tcp_pose already reports grasp_site (flange +0.05 m along link7
    # z); then leave tcp_offset=None and read_state applies no correction.
    # Fallback: leave the RDK TCP at the flange and pass tcp_offset=
    # BARE_TCP_OFFSET_LINK7 (--flange-tcp) — read_state then shifts point and
    # velocity into grasp_site itself. No Tool-API guessing either way.
    self._tcp_offset = (
      np.zeros(3, np.float32)
      if tcp_offset is None
      else np.asarray(tcp_offset, np.float32)
    )
    self._max_wrench = list(max_contact_wrench or MAX_CONTACT_WRENCH)
    self._max_linear_vel = max_linear_vel
    self._max_linear_acc = max_linear_acc
    self._k_trans_floor = k_trans_floor

  # --- state ---------------------------------------------------------------
  def read_state(self) -> RobotState:
    s = self.robot.states()  # 1.9: RobotStates struct (attributes, NOT a dict)
    q = np.asarray(s.q, dtype=np.float32)
    dq = np.asarray(s.dq, dtype=np.float32)
    tau = np.asarray(s.tau, dtype=np.float32)  # total measured joint torque, Nm
    tcp = np.asarray(s.tcp_pose, dtype=np.float32)  # [x,y,z, qw,qx,qy,qz]
    tcp_vel = np.asarray(s.tcp_vel, dtype=np.float32)  # [vx,vy,vz, wx,wy,wz]
    quat = tcp[3:7]  # wxyz
    # Shift the reported TCP point into grasp_site when the RDK TCP is at the
    # flange (self._tcp_offset != 0). r = R(quat)·offset; velocity picks up the
    # rigid-body term ω × r. Both vanish when the tool is configured in Elements.
    r = _quat_rotate_vec(quat, self._tcp_offset)
    return RobotState(
      q=q[:7],
      dq=dq[:7],
      # tau_meas ≡ sim qfrc_actuator + qfrc_external (total transmitted torque).
      # CALIBRATE first: hold still at home and match sign/offset vs a sim hold,
      # this signal is how the policy senses "a human is pushing me".
      tau_meas=tau[:7],
      ee_pos=(tcp[0:3] + r).astype(np.float32),
      # NOTE: the sim obs ee_vel is link7 *COM* linear velocity, not grasp_site
      # velocity; this ω×r-corrected TCP-point velocity is a close match. For a
      # faithful match compute J_com(link7)·dq via flexivrdk.Model (route A).
      ee_vel=(tcp_vel[0:3] + np.cross(tcp_vel[3:6], r)).astype(np.float32),
      ee_quat=quat,
    )

  # --- mode / command ------------------------------------------------------
  def enter_cartesian_mode(self) -> None:
    # 1.9 has only the NRT Cartesian mode: the robot interpolates toward the
    # last commanded target, which suits a pure-Python ~100 Hz loop.
    self.robot.SwitchMode(self.Mode.NRT_CARTESIAN_MOTION_FORCE)
    # Safety: cap the contact wrench the impedance controller may exert.
    self.robot.SetMaxContactWrench(self._max_wrench)

  def send_command(self, cmd: Command) -> None:
    pose = np.concatenate([cmd.target_pos, cmd.target_quat]).tolist()  # 7-vec
    # Per-axis Cartesian stiffness [kx,ky,kz, krx,kry,krz], with optional floor.
    k = cmd.stiffness_trans
    if self._k_trans_floor is not None:
      k = np.maximum(k, self._k_trans_floor)
    stiffness = [
      float(k[0]),
      float(k[1]),
      float(k[2]),
      cmd.stiffness_rot,
      cmd.stiffness_rot,
      cmd.stiffness_rot,
    ]
    # Z_x = per-axis damping RATIO, clamped to Flexiv's allowed [0.3, 0.8].
    # SetCartesianImpedance(K_x, Z_x) updates online.
    lo, hi = Z_X_RANGE
    damping = [
      min(max(z, lo), hi) for z in ([DAMPING_RATIO] * 3 + [ROT_DAMPING_RATIO] * 3)
    ]
    self.robot.SetCartesianImpedance(stiffness, damping)
    # Pure motion: zero feed-forward wrench, impedance about the target pose.
    # max_linear_vel/acc cap how hard the generator chases target jumps.
    self.robot.SendCartesianMotionForce(
      pose,
      max_linear_vel=self._max_linear_vel,
      max_linear_acc=self._max_linear_acc,
    )

  def probe(self) -> None:
    s = self.robot.states()  # RobotStates struct
    print("robot.states() fields:")
    for name in (
      "q",
      "dq",
      "tau",
      "tau_ext",
      "tcp_pose",
      "flange_pose",
      "tcp_vel",
      "ext_wrench_in_tcp",
    ):
      v = getattr(s, name, None)
      if v is None:
        print(f"  {name}: <missing — inspect dir(states())>")
        continue
      arr = np.asarray(v, dtype=float)
      print(f"  {name}: shape={arr.shape} sample={np.round(arr.ravel()[:4], 4)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DeployCfg:
  robot_sn: str = "Rizon4s-XXXXXX"  # <-- your robot serial number
  goal_pos: np.ndarray = field(
    # A reachable target in the base frame (m). Must be inside the trained
    # workspace box [0.30,0.70]×[-0.20,0.20]×[0.35,0.65] and radial shell.
    default_factory=lambda: np.array([0.50, 0.0, 0.50], dtype=np.float32)
  )


def _smooth_cmd(prev: Command | None, new: Command, s: float) -> Command:
  """EMA-filter the target + stiffness (not the locked quat). s in [0, 1)."""
  if prev is None or s <= 0.0:
    return new
  return Command(
    target_pos=(1.0 - s) * new.target_pos + s * prev.target_pos,
    target_quat=new.target_quat,
    stiffness_trans=(1.0 - s) * new.stiffness_trans + s * prev.stiffness_trans,
    stiffness_rot=(1.0 - s) * new.stiffness_rot + s * prev.stiffness_rot,
  )


def run(
  cfg: DeployCfg,
  adapter: RizonAdapter | None,
  max_steps: int | None,
  confirm: bool = True,
  smooth: float = 0.0,
) -> None:
  actor = Actor(ONNX_PATH)
  last_action = np.zeros(ACT_DIM, dtype=np.float32)  # sim reset: raw_action = 0
  filt_cmd: Command | None = None  # EMA state for command smoothing

  if adapter is not None:
    adapter.enter_cartesian_mode()
    st0 = adapter.read_state()
    lock_quat = st0.ee_quat.copy()  # lock orientation at start
    # The arm is about to move under policy control — summarise and confirm.
    print(
      f"\n  goal (base frame) : {cfg.goal_pos.round(3)}"
      f"\n  start EE (grasp)  : {st0.ee_pos.round(3)}"
      f"  Δgoal={np.linalg.norm(cfg.goal_pos - st0.ee_pos):.3f} m"
      f"\n  max contact wrench: {adapter._max_wrench}"
      f"\n  max linear vel    : {adapter._max_linear_vel} m/s"
      f"\n  steps             : {max_steps}"
    )
    if confirm and input("Start policy on the REAL arm? type 'go': ").strip() != "go":
      print("Aborted.")
      return
  else:
    lock_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

  step = 0
  while max_steps is None or step < max_steps:
    t0 = time.perf_counter()

    if adapter is not None:
      st = adapter.read_state()
    else:  # --dry-run: exercise the policy pipeline with a static fake state
      st = RobotState(
        q=Q_DEFAULT.copy(),
        dq=np.zeros(7, np.float32),
        tau_meas=np.zeros(7, np.float32),
        ee_pos=cfg.goal_pos + np.array([0.1, 0.0, 0.0], np.float32),
        ee_vel=np.zeros(3, np.float32),
        ee_quat=lock_quat,
      )

    obs = build_observation(st, cfg.goal_pos, last_action)
    raw_action = actor(obs)  # (7,) raw, pre-tanh
    cmd = decode_action(raw_action, cfg.goal_pos, lock_quat)
    cmd = _smooth_cmd(filt_cmd, cmd, smooth)  # optional anti-jitter EMA
    filt_cmd = cmd

    if adapter is not None:
      adapter.send_command(cmd)

    if step % 25 == 0:
      print(
        f"[{step:5d}] EE={st.ee_pos.round(3)}  "
        f"Δgoal={np.linalg.norm(cfg.goal_pos - st.ee_pos):.3f}m  "
        f"K={cmd.stiffness_trans.round(0)}  Krot={cmd.stiffness_rot:.0f}  "
        f"|tau|={np.linalg.norm(st.tau_meas):.1f}"
      )

    last_action = raw_action  # cache RAW action as next last_action obs
    step += 1

    # Pace to 100 Hz.
    dt = time.perf_counter() - t0
    if dt < POLICY_DT:
      time.sleep(POLICY_DT - dt)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--robot-sn", default=DeployCfg.robot_sn)
  ap.add_argument("--dry-run", action="store_true", help="no robot; test pipeline")
  ap.add_argument("--probe", action="store_true", help="print robot.states() keys")
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
  ap.add_argument(
    "--flange-tcp",
    action="store_true",
    help="RDK TCP left at the flange; apply the +0.05m link7-z grasp_site "
    "offset in code (omit if the tool is configured in Flexiv Elements)",
  )
  ap.add_argument("--yes", action="store_true", help="skip the pre-motion prompt")
  ap.add_argument(
    "--smooth",
    type=float,
    default=CMD_SMOOTH,
    help="EMA smoothing of target+stiffness in [0,1); higher=smoother (anti-jitter)",
  )
  ap.add_argument("--max-linear-vel", type=float, default=MAX_LINEAR_VEL, metavar="MPS")
  ap.add_argument(
    "--max-linear-acc", type=float, default=MAX_LINEAR_ACC, metavar="MPS2"
  )
  ap.add_argument(
    "--k-floor",
    type=float,
    default=K_TRANS_FLOOR,
    metavar="N_PER_M",
    help="floor on translational stiffness (N/m); default: use the policy value",
  )
  args = ap.parse_args()

  cfg = DeployCfg(robot_sn=args.robot_sn)
  if args.goal is not None:
    cfg.goal_pos = np.array(args.goal, dtype=np.float32)

  if args.dry_run:
    run(cfg, adapter=None, max_steps=args.steps or 200)
    return

  adapter = RizonAdapter(
    args.robot_sn,
    tcp_offset=BARE_TCP_OFFSET_LINK7 if args.flange_tcp else None,
    max_linear_vel=args.max_linear_vel,
    max_linear_acc=args.max_linear_acc,
    k_trans_floor=args.k_floor,
  )
  if args.probe:
    adapter.probe()
    return
  run(
    cfg,
    adapter=adapter,
    max_steps=args.steps,
    confirm=not args.yes,
    smooth=args.smooth,
  )


if __name__ == "__main__":
  main()
