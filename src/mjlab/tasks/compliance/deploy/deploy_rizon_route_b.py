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

IMPORTANT — this is a skeleton
------------------------------
Everything on the *policy* side (obs layout, decode, GRU state) is implemented to
match the training code exactly and is marked ``# [POLICY]``.  Every call into
Flexiv RDK is isolated in ``RizonAdapter`` and marked ``# [RDK — CONFIRM]``:
RDK v2.x method / state-key names vary across minor versions, so run
``--probe`` first to print your ``robot.states()`` keys and adapt the adapter.

Run the pipeline with no robot attached to sanity-check the policy side:

    uv run python -m mjlab.tasks.compliance.deploy.deploy_rizon_route_b --dry-run
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
DAMPING_RATIO = 0.8  # only used by route A; here for reference
EFFECTIVE_MASS = 2.0  # only used by route A

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


# ─────────────────────────────────────────────────────────────────────────────
# [RDK — CONFIRM] Flexiv adapter. Every method here touches version-specific RDK
# API. Confirm names against YOUR installed flexivrdk (run --probe first).
# ─────────────────────────────────────────────────────────────────────────────


class RizonAdapter:
  """Isolates all Flexiv RDK calls. Adapt to your RDK v2.x minor version."""

  def __init__(self, robot_sn: str):
    import flexivrdk  # noqa: PLC0415  (optional dep, only needed on the robot)

    self._rdk = flexivrdk
    self.robot = flexivrdk.Robot(robot_sn)  # [RDK — CONFIRM] ctor signature

    # Clear faults / enable / wait until operational.
    if self.robot.fault():  # [RDK — CONFIRM]
      self.robot.ClearFault()
    self.robot.Enable()
    while not self.robot.operational():  # [RDK — CONFIRM]
      time.sleep(0.1)

    # IMPORTANT: configure the tool/TCP so tcp_pose == grasp_site, i.e. the
    # flange frame translated +0.05 m along link7 z. Either set it here via the
    # Tool API, or read flange_pose below and add the offset yourself.
    # [RDK — CONFIRM] e.g. flexivrdk.Tool(robot).Switch(...) or robot.SetTool(...)

    self.Mode = flexivrdk.Mode

  # --- state ---------------------------------------------------------------
  def read_state(self) -> RobotState:
    s = self.robot.states()  # returns a dict in Python (v2.x)
    # [RDK — CONFIRM] the dict keys below. Run --probe to print them.
    q = np.asarray(s["q"], dtype=np.float32)
    dq = np.asarray(s["dq"], dtype=np.float32)
    tau = np.asarray(s["tau"], dtype=np.float32)  # measured joint torque
    tcp = np.asarray(s["tcp_pose"], dtype=np.float32)  # [x,y,z, qw,qx,qy,qz]
    tcp_vel = np.asarray(s["tcp_vel"], dtype=np.float32)  # [vx,vy,vz, wx,wy,wz]
    return RobotState(
      q=q[:7],
      dq=dq[:7],
      # tau_meas ≡ sim qfrc_actuator + qfrc_external (total transmitted torque).
      # CALIBRATE first: hold still at home and match sign/offset vs a sim hold,
      # this signal is how the policy senses "a human is pushing me".
      tau_meas=tau[:7],
      ee_pos=tcp[0:3],
      # NOTE: the sim obs ee_vel is link7 *COM* linear velocity, not TCP vel.
      # TCP linear velocity is a close approximation; for a faithful match
      # compute J_com(link7)·dq via flexivrdk.Model (that's a route-A refinement).
      ee_vel=tcp_vel[0:3],
      ee_quat=tcp[3:7],  # wxyz
    )

  # --- mode / command ------------------------------------------------------
  def enter_cartesian_mode(self) -> None:
    # NRT is friendlier for a pure-Python ~100 Hz loop (the robot interpolates
    # and holds the last target); RT expects a tight ~1 kHz stream. Pick one.
    self.robot.SwitchMode(self.Mode.NRT_CARTESIAN_MOTION_FORCE)  # [RDK — CONFIRM]

  def send_command(self, cmd: Command) -> None:
    pose = np.concatenate([cmd.target_pos, cmd.target_quat]).tolist()  # 7-vec
    # Per-axis Cartesian stiffness [kx,ky,kz, krx,kry,krz].
    stiffness = [
      float(cmd.stiffness_trans[0]),
      float(cmd.stiffness_trans[1]),
      float(cmd.stiffness_trans[2]),
      cmd.stiffness_rot,
      cmd.stiffness_rot,
      cmd.stiffness_rot,
    ]
    # [RDK — CONFIRM] exact names. Likely:
    #   self.robot.SetCartesianImpedance(stiffness)     # online stiffness
    #   self.robot.SendCartesianMotionForce(pose)       # NRT target pose
    # (RT variant: self.robot.StreamCartesianMotionForce(pose))
    self.robot.SetCartesianImpedance(stiffness)
    self.robot.SendCartesianMotionForce(pose)

  def probe(self) -> None:
    s = self.robot.states()
    print("robot.states() keys:", list(s.keys()))
    for k, v in s.items():
      shape = np.asarray(v).shape if hasattr(v, "__len__") else "scalar"
      print(f"  {k}: {shape}")


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


def run(cfg: DeployCfg, adapter: RizonAdapter | None, max_steps: int | None) -> None:
  actor = Actor(ONNX_PATH)
  last_action = np.zeros(ACT_DIM, dtype=np.float32)  # sim reset: raw_action = 0

  if adapter is not None:
    adapter.enter_cartesian_mode()
    lock_quat = adapter.read_state().ee_quat.copy()  # lock orientation at start
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

    if adapter is not None:
      adapter.send_command(cmd)
    else:
      if step % 25 == 0:
        print(
          f"[{step:5d}] Δgoal={np.linalg.norm(cfg.goal_pos - st.ee_pos):.3f}m  "
          f"K={cmd.stiffness_trans.round(0)}  Krot={cmd.stiffness_rot:.0f}  "
          f"tgt={cmd.target_pos.round(3)}"
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
  args = ap.parse_args()

  cfg = DeployCfg(robot_sn=args.robot_sn)
  if args.goal is not None:
    cfg.goal_pos = np.array(args.goal, dtype=np.float32)

  if args.dry_run:
    run(cfg, adapter=None, max_steps=args.steps or 200)
    return

  adapter = RizonAdapter(args.robot_sn)
  if args.probe:
    adapter.probe()
    return
  run(cfg, adapter=adapter, max_steps=args.steps)


if __name__ == "__main__":
  main()
