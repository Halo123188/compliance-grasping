"""Real-robot deployment for the two **direct joint-torque** compliance policies.

Target checkpoints
------------------
``--policy tracking``
  ``logs/rsl_rl/compliance_tracking_flexiv_a_torquebare/
  2026-07-28_16-52-56_bare_A_as70_ws25_s0`` — task
  ``Mjlab-ComplianceTracking-StageA-TorqueBare-Flexiv``.  Bare Rizon 4S, Stage A
  (no gripper/object/weld).  Follows the teacher's reach-and-lift path and yields
  when pushed, resuming from where it left off.

``--policy reach``
  ``logs/rsl_rl/compliance_reach_flexiv/2026-07-27_20-49-19_torque_smoothbare_
  curriculum`` — task ``Mjlab-Compliance-Reach-Flexiv-TorqueSmoothBare``.  Bare
  Rizon 4S, reach a Cartesian goal under a stubborn "human" pull.

Both are GRU actors that, at 100 Hz, emit a 7-D **residual joint torque**:
``tau = clip(tanh(a) * tau_max + g(q, qdot), +-tau_max)`` (``gravity_comp=True``).
Neither emits impedance parameters — this is *route A*, so
``deploy_rizon_route_b.py`` (Cartesian impedance) does not apply to them.

The two differ in exactly three places, all captured in ``PolicySpec``: the home
pose ``q_default`` that ``joint_pos_rel`` is measured against, the meaning of the
``ee_vel`` observation (grasp-site velocity vs. link7 COM velocity), and the goal
box.  Everything else — the 37-D observation layout, the action decode, the
torque limits — is shared.

Observation, exactly as trained (37):
  ``q - q_default (7), qdot (7), tau_meas (7), x_ee (3), xdot_ee (3),
  x_goal - x_ee (3), a_prev (7 raw, pre-tanh)``

Frames
------
``x_ee`` is the **sim** ``grasp_site`` (link7 flange + 5 cm along link7's local
+z), expressed in the robot base frame.  Rather than trust the RDK's configured
TCP, this script recomputes it from the measured joint angles with an exact
forward-kinematic copy of the MuJoCo chain (``_CHAIN`` below, transcribed from
``assets/flexiv_rizon4s/flexiv_rizon4s.xml``).  That makes the observation
byte-comparable to sim regardless of how the tool is configured in Flexiv
Elements, and it is the only way to get the link7 *COM* velocity the ``reach``
policy was trained on — the RDK does not report it.  ``--probe`` prints the FK
result next to ``states().flange_pose`` so the base-frame convention can be
checked against the real robot before anything moves.

Control backends
----------------
``rt_torque`` (faithful)
  ``RT_JOINT_TORQUE`` + ``StreamJointTorque(tau, enable_gravity_comp=True)`` at
  1 kHz, holding each policy torque for 10 ticks — exactly the sim's
  ``decimation=10``.  **Needs a Linux host**: every macOS flexivrdk wheel that
  supports the Rizon 4S has the RT layer stripped, so this backend is simply
  absent there (``flexivrdk.Mode`` has no ``RT_*`` entries).  The 1.9.0
  ``manylinux_2_35`` wheels (glibc >= 2.35, i.e. Ubuntu 22.04+; cp310/cp312/
  cp313, x86_64 or aarch64) do ship ``RT_JOINT_TORQUE`` and do list ``Rizon4s``
  as a supported model — verified against the wheel binary.  The robot faults if
  more than ``--timeliness-limit`` percent of commands arrive late, so watch the
  overrun figure printed on exit.

``joint_impedance`` (fallback, approximate)
  ``NRT_JOINT_IMPEDANCE``: the robot runs ``tau = K_q (q_d - q) - D_q qdot +
  g(q)``, so commanding ``q_d = q_meas + tau_res / K_q`` makes it a torque source
  to first order.  This *does* run on macOS.  Deviations from sim, in order of
  size: the ``-D_q qdot`` term adds damping the policy never trained against; the
  command is NRT (100 Hz, interpolated) rather than a 1 kHz hold; and torque
  saturates at ``K_q * max_offset`` (printed at startup).  Use it to validate the
  observation pipeline and to see the gross behaviour, not to judge the feel.

Running it
----------
The script imports no ``mjlab`` code (numpy / onnxruntime / flexivrdk only), so
run it by **file path** from the dedicated venv:

    .venv-deploy/bin/python \\
      src/mjlab/tasks/compliance/deploy/deploy_rizon_torque.py \\
      --policy tracking --self-test          # FK check, no robot, no ONNX
    ... --policy tracking --dry-run          # + the ONNX/GRU pipeline
    ... --policy tracking --probe            # on the robot: frames + limits
    ... --policy tracking --robot-sn Rizon4s-063501

Home the arm to the policy's own start pose first — ``q_default`` differs between
the two policies and ``joint_pos_rel`` is measured against it:

    .venv-deploy/bin/python src/mjlab/tasks/compliance/deploy/home_rizon.py \\
      --robot-sn Rizon4s-063501 --policy tracking

The ONNX is written next to the checkpoints on every ``save()`` (see
``ManipulationOnPolicyRunner.save``), named after the run directory, and tracks
the *last* saved checkpoint.  Copy that file down from the training host; point
``--onnx`` anywhere else.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────────────────────
# Kinematics — an exact transcription of the MuJoCo chain in
# assets/flexiv_rizon4s/flexiv_rizon4s.xml.  Each entry is one body:
# (translation from the parent frame, fixed rotation as wxyz, hinge axis).  The
# hinge sits at the body-frame origin in every case, so the body pose is
#   X_body = X_parent . T(pos, quat) . Rot(axis, q_i).
# Do not "tidy" these numbers: they define the frame the policy was trained in.
# ─────────────────────────────────────────────────────────────────────────────

_Vec3 = tuple[float, float, float]
_Quat = tuple[float, float, float, float]

_CHAIN: tuple[tuple[_Vec3, _Quat, _Vec3], ...] = (
  ((0.0, 0.0, 0.155), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),  # link1
  ((0.0, 0.03, 0.21), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # link2
  ((0.0, 0.035, 0.205), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # link3
  ((-0.02, -0.03, 0.19), (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # link4
  ((-0.02, 0.025, 0.195), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),  # link5
  ((0.0, 0.03, 0.19), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # link6
  ((-0.015, 0.073, 0.11), (0.707107, 0.0, -0.707107, 0.0), (0.0, 0.0, 1.0)),  # link7
)

# grasp_site on the bare arm: link7 + 5 cm along its local +z (constants.py
# _BARE_EE_POS).  This is the point the ee_pos / goal_error observations use.
_EE_LOCAL = np.array([0.0, 0.0, 0.05])
# link7's inertial (COM) offset, from the XML <inertial pos=...>.  The `reach`
# policy's ee_vel observation is this body's COM velocity, not the site's.
_LINK7_COM_LOCAL = np.array([0.0, 0.0001, 0.0544])

NUM_JOINTS = 7


def _quat_to_mat(q: _Quat) -> np.ndarray:
  """Rotation matrix from a wxyz quaternion (MuJoCo's convention)."""
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
  """Rodrigues rotation about a unit ``axis`` by ``angle`` rad."""
  k = np.array(
    [
      [0.0, -axis[2], axis[1]],
      [axis[2], 0.0, -axis[0]],
      [-axis[1], axis[0], 0.0],
    ]
  )
  return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


# Precomputed constant transforms, so the per-step FK only builds the hinges.
_FIXED = tuple((np.array(p), _quat_to_mat(q), np.array(a)) for p, q, a in _CHAIN)


@dataclass
class Frames:
  """Forward kinematics of the bare arm at one joint configuration."""

  link7_pos: np.ndarray  # (3,) link7 body-frame origin, base frame
  link7_rot: np.ndarray  # (3, 3) link7 orientation, base frame
  joint_pos: np.ndarray  # (7, 3) hinge anchors, base frame
  joint_axis: np.ndarray  # (7, 3) hinge axes, base frame

  def point(self, local: np.ndarray) -> np.ndarray:
    """A point given in link7's local frame, expressed in the base frame."""
    return self.link7_pos + self.link7_rot @ local

  def point_vel(self, local: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Linear velocity of that point: ``sum_i (axis_i x (p - anchor_i)) qdot_i``."""
    p = self.point(local)
    return (np.cross(self.joint_axis, p - self.joint_pos) * dq[:, None]).sum(axis=0)


def forward_kinematics(q: np.ndarray) -> Frames:
  """Base-frame FK of the bare Rizon 4S, matching the MuJoCo model exactly."""
  pos = np.zeros(3)
  rot = np.eye(3)
  anchors = np.zeros((NUM_JOINTS, 3))
  axes = np.zeros((NUM_JOINTS, 3))
  for i, (offset, fixed_rot, axis) in enumerate(_FIXED):
    pos = pos + rot @ offset
    rot = rot @ fixed_rot
    anchors[i] = pos
    axes[i] = rot @ axis
    rot = rot @ _axis_angle_to_mat(axis, float(q[i]))
  return Frames(link7_pos=pos, link7_rot=rot, joint_pos=anchors, joint_axis=axes)


# ─────────────────────────────────────────────────────────────────────────────
# Contract constants — read straight from the task code.  Do not "clean these
# up": they define the exact I/O each network was trained on.
# ─────────────────────────────────────────────────────────────────────────────

# Per-joint torque limit (Rizon 4S datasheet), also the action scale: raw +-1
# maps to +-effort_limit.  constants.FLEXIV_ARM_EFFORT_LIMIT.
EFFORT_LIMIT = np.array([123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0])

OBS_DIM = 37
ACT_DIM = 7
GRU_HIDDEN = 128

POLICY_HZ = 100.0
POLICY_DT = 1.0 / POLICY_HZ

# --- joint_impedance backend ------------------------------------------------
# Cap on the commanded position offset q_d - q.  Also sets the reachable torque:
# tau_max_effective = K_q * MAX_JOINT_OFFSET (printed at startup).
MAX_JOINT_OFFSET = 0.06  # rad
KQ_SCALE = 1.0  # multiplier on the robot's nominal K_q
# NRT generator caps while chasing the offset target.  The offsets are small, so
# these only bound how violently a step change is chased.
NRT_MAX_VEL = 1.5  # rad/s
NRT_MAX_ACC = 6.0  # rad/s^2

# --- safety -----------------------------------------------------------------
DQ_GUARD_FRAC = 0.5  # trip if |qdot| exceeds this fraction of the robot's dq_max
Q_MARGIN = 0.10  # rad, trip this close to a joint limit
# Cartesian keep-out: the EE must stay inside this base-frame box.  Wide enough
# for both policies' goal boxes plus the yield excursions, tight enough to catch
# a runaway before it reaches the table or the robot's own base.
EE_GUARD_MIN = np.array([0.10, -0.50, 0.08])
EE_GUARD_MAX = np.array([0.85, 0.50, 1.00])

_LOG_DIR = Path("logs/rsl_rl")


@dataclass(frozen=True)
class PolicySpec:
  """Everything that differs between the two torque policies."""

  name: str
  run_dir: Path
  q_default: np.ndarray
  """Home pose ``joint_pos_rel`` is measured against (the task's init_state)."""
  ee_vel_source: str
  """``"site"`` (grasp-site velocity) or ``"link7_com"`` (body COM velocity)."""
  goal_box: tuple[np.ndarray, np.ndarray]
  """The box the training goal was sampled from, base frame."""
  default_goal: np.ndarray
  warm_start_steps: int
  """Training-time residual ramp after reset (``ArmTorqueActionCfg``)."""
  default_steps: int
  """Policy steps in one training episode (``episode_length_s`` * 100)."""
  description: str

  @property
  def onnx_path(self) -> Path:
    """The exporter names the ONNX after the run directory."""
    return self.run_dir / f"{self.run_dir.name}.onnx"


POLICIES: dict[str, PolicySpec] = {
  # Mjlab-ComplianceTracking-StageA-TorqueBare-Flexiv.  Home is the retracted
  # start pose in compliance_tracking/config/flexiv/env_cfg._HOME_ARM_POSE; the
  # goal box is _BARE_GRASP_BOX_{MIN,MAX}; ee_vel is the grasp *site* velocity
  # (teacher.ee_vel_w).  Trained with --warm-start-steps 25 (the "ws25" in the
  # run name) and --admittance-stiffness 70 ("as70", a teacher-side parameter
  # with no deploy-time counterpart).
  "tracking": PolicySpec(
    name="tracking",
    run_dir=_LOG_DIR
    / "compliance_tracking_flexiv_a_torquebare"
    / "2026-07-28_16-52-56_bare_A_as70_ws25_s0",
    q_default=np.array(
      [-0.225705, 0.003626, 0.507476, 2.438249, 0.002759, 0.864276, -1.290931]
    ),
    ee_vel_source="site",
    goal_box=(np.array([0.42, -0.10, 0.18]), np.array([0.58, 0.10, 0.24])),
    default_goal=np.array([0.50, 0.0, 0.21]),
    warm_start_steps=25,
    default_steps=800,  # episode_length_s = 8.0
    description=(
      "reach-and-lift along the teacher path, yielding to a push and resuming; "
      "the path is replayed from the GRU's own clock (approach 1.2 s, descend "
      "0.8 s, dwell 0.6 s, lift 0.8 s to +10 cm), so the arm will keep moving "
      "after it reaches the goal"
    ),
  ),
  # Mjlab-Compliance-Reach-Flexiv-TorqueSmoothBare.  Home is BARE_HOME_KEYFRAME;
  # the goal box is ReachCommandCfg.box_{min,max}; ee_vel is the link7 *body COM*
  # velocity (compliance.mdp.ee_lin_vel with asset_cfg body_names=("link7",)).
  "reach": PolicySpec(
    name="reach",
    run_dir=_LOG_DIR
    / "compliance_reach_flexiv"
    / "2026-07-27_20-49-19_torque_smoothbare_curriculum",
    q_default=np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0]),
    ee_vel_source="link7_com",
    goal_box=(np.array([0.30, -0.20, 0.35]), np.array([0.70, 0.20, 0.65])),
    default_goal=np.array([0.50, 0.0, 0.50]),
    warm_start_steps=0,  # no ramp in training; --ramp-steps 25 for a first run
    default_steps=700,  # episode_length_s = 7.0
    description=(
      "reach the goal and hold it against a stubborn pull; the episode ends in "
      "sim on success, so on hardware it simply keeps holding"
    ),
  ),
}


# ─────────────────────────────────────────────────────────────────────────────
# [POLICY] Observation assembly and action decode.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobotState:
  """The real-robot signals the observation needs, in sim conventions."""

  q: np.ndarray  # (7,) joint position, rad
  dq: np.ndarray  # (7,) joint velocity, rad/s
  tau_meas: np.ndarray  # (7,) measured joint torque, Nm
  ee_pos: np.ndarray  # (3,) grasp_site position, base frame, m
  ee_vel: np.ndarray  # (3,) per spec.ee_vel_source, base frame, m/s


def state_from_joints(
  q: np.ndarray, dq: np.ndarray, tau: np.ndarray, spec: PolicySpec
) -> RobotState:
  """Build the observation-side state from joint measurements alone."""
  frames = forward_kinematics(q)
  local = _EE_LOCAL if spec.ee_vel_source == "site" else _LINK7_COM_LOCAL
  return RobotState(
    q=q,
    dq=dq,
    # tau_meas === sim qfrc_actuator + qfrc_external (total transmitted torque).
    # This is the only channel through which the policy senses a push, so
    # CALIBRATE it: hold still at home, confirm sign and offset against a sim
    # hold at the same pose before trusting the yielding behaviour.
    tau_meas=tau,
    ee_pos=frames.point(_EE_LOCAL),
    ee_vel=frames.point_vel(local, dq),
  )


def build_observation(
  st: RobotState, goal_pos: np.ndarray, last_action: np.ndarray, spec: PolicySpec
) -> np.ndarray:
  """Assemble the 37-D observation. Feed RAW (normalization is inside the ONNX)."""
  obs = np.concatenate(
    [
      st.q - spec.q_default,  # 7  joint_pos_rel
      st.dq,  # 7  joint_vel_rel (default qdot is 0)
      st.tau_meas,  # 7
      st.ee_pos,  # 3
      st.ee_vel,  # 3
      goal_pos - st.ee_pos,  # 3  goal_error
      last_action,  # 7  raw (pre-tanh) previous action
    ]
  ).astype(np.float32)
  assert obs.shape == (OBS_DIM,), obs.shape
  return obs


def decode_action(raw_action: np.ndarray, ramp: float, scale: float) -> np.ndarray:
  """Raw action -> residual joint torque, mirroring ``ArmTorqueAction``.

  Sim computes ``tanh(a) * tau_max``, ramps it by the warm-start weight, then
  adds ``qfrc_bias`` and clips to ``+-tau_max``.  Here the robot adds its own
  gravity term, so we return the *residual* and clip that; ``scale`` is a
  deploy-only safety derate with no sim counterpart (1.0 = as trained).
  """
  safe = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
  tau = np.tanh(safe) * EFFORT_LIMIT * ramp * scale
  return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)


class Actor:
  """The exported GRU actor. Normalization is baked into the ONNX graph."""

  def __init__(self, onnx_path: Path):
    if not onnx_path.exists():
      raise SystemExit(
        f"ONNX not found: {onnx_path}\n"
        "It is written next to the checkpoints on every save(); copy the run "
        "directory's <run_name>.onnx down from the training host, or pass "
        "--onnx <path>."
      )
    self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = {i.name for i in self.sess.get_inputs()}
    assert {"obs", "h_in"} <= names, f"unexpected ONNX inputs: {names}"
    self.reset()

  def reset(self) -> None:
    self.h = np.zeros((1, 1, GRU_HIDDEN), dtype=np.float32)

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    actions, self.h = self.sess.run(
      ["actions", "h_out"], {"obs": obs[None, :], "h_in": self.h}
    )
    return actions[0].astype(np.float32)  # (7,) raw action


# ─────────────────────────────────────────────────────────────────────────────
# Flexiv adapter + the two torque backends.
#
# RDK v1.9.x is the line compatible with the Rizon 4S (2.x dropped the model).
# Its control API is flat / single-arm: states() -> RobotStates struct
# (attributes, not a dict), Enable(), SwitchMode(Mode.X), no joint-group args.
# ─────────────────────────────────────────────────────────────────────────────


class RizonAdapter:
  """Connect, enable, read state; owns the safety limits read from the robot."""

  def __init__(self, robot_sn: str, spec: PolicySpec):
    import flexivrdk  # noqa: PLC0415  (optional dep, only needed on the robot)

    self._rdk = flexivrdk
    self.spec = spec
    self.robot = flexivrdk.Robot(robot_sn)
    self.Mode = flexivrdk.Mode

    if self.robot.fault():
      print("Fault present; clearing...")
      self.robot.ClearFault()
    self.robot.Enable()
    print("Enabling; waiting until operational...")
    while not self.robot.operational():
      time.sleep(0.1)

    info = self.robot.info()
    self.dof = int(info.DoF)
    if self.dof != NUM_JOINTS:
      raise SystemExit(f"robot DoF {self.dof} != {NUM_JOINTS}; wrong robot model?")
    self.q_min = np.asarray(info.q_min, dtype=float)[:NUM_JOINTS]
    self.q_max = np.asarray(info.q_max, dtype=float)[:NUM_JOINTS]
    self.dq_max = np.asarray(info.dq_max, dtype=float)[:NUM_JOINTS]
    self.tau_max = np.asarray(info.tau_max, dtype=float)[:NUM_JOINTS]
    self.kq_nom = np.asarray(info.K_q_nom, dtype=float)[:NUM_JOINTS]

  def read_state(self) -> RobotState:
    s = self.robot.states()
    return state_from_joints(
      np.asarray(s.q, dtype=float)[:NUM_JOINTS],
      np.asarray(s.dq, dtype=float)[:NUM_JOINTS],
      np.asarray(s.tau, dtype=float)[:NUM_JOINTS],
      self.spec,
    )

  def probe(self) -> None:
    """Print the live state next to our FK, so the frames can be checked."""
    s = self.robot.states()
    q = np.asarray(s.q, dtype=float)[:NUM_JOINTS]
    frames = forward_kinematics(q)
    print("\n--- robot.info() ---")
    print(f"  model      : {self.robot.info().model_name}")
    print(f"  q_min (deg): {np.degrees(self.q_min).round(1)}")
    print(f"  q_max (deg): {np.degrees(self.q_max).round(1)}")
    print(f"  dq_max     : {self.dq_max.round(2)}")
    print(f"  tau_max    : {self.tau_max.round(1)}")
    print(f"  K_q_nom    : {self.kq_nom.round(0)}")
    print("\n--- states() ---")
    for name in ("q", "dq", "tau", "tau_ext", "tcp_pose", "flange_pose", "tcp_vel"):
      v = getattr(s, name, None)
      if v is None:
        print(f"  {name}: <missing>")
        continue
      print(f"  {name}: {np.asarray(v, dtype=float).round(4)}")
    print("\n--- our FK (sim base frame) ---")
    print(f"  grasp_site pos : {frames.point(_EE_LOCAL).round(4)}")
    print(f"  link7 origin   : {frames.link7_pos.round(4)}")
    print(f"  link7 COM      : {frames.point(_LINK7_COM_LOCAL).round(4)}")
    print(
      "\nCompare 'link7 origin' against flange_pose[:3]: they are different\n"
      "points (the flange plate sits further along link7's +z), but a\n"
      "disagreement in *direction* means the RDK world frame and the MuJoCo\n"
      "base frame differ, and every Cartesian term here would be wrong."
    )

  def stop(self) -> None:
    try:
      self.robot.Stop()
      self.robot.SwitchMode(self.Mode.IDLE)
    except Exception as exc:  # noqa: BLE001 — best effort on the way out
      print(f"[WARN] stop failed: {exc}")


class RtTorqueBackend:
  """``RT_JOINT_TORQUE``: stream the residual at 1 kHz, gravity comp on-robot.

  The faithful backend. Available only where the wheel ships the RT layer
  (Linux); ``supported()`` is False on macOS.

  The robot monitors command *timeliness* and faults once too large a fraction
  of commands arrive late (default 2%).  A pure-Python 1 kHz loop on a stock
  kernel will miss some deadlines, so ``timeliness_limit`` raises that tolerance
  — at the cost of the robot genuinely running on stale commands when it fires.
  Watch the overrun count this script prints on exit before raising it.
  """

  send_hz = 1000.0

  def __init__(self, adapter: RizonAdapter, timeliness_limit: float | None = None):
    self.adapter = adapter
    self.robot = adapter.robot
    self.timeliness_limit = timeliness_limit

  @staticmethod
  def supported(adapter: RizonAdapter) -> bool:
    return hasattr(adapter.Mode, "RT_JOINT_TORQUE") and hasattr(
      adapter.robot, "StreamJointTorque"
    )

  def describe(self) -> str:
    limit = (
      "robot default" if self.timeliness_limit is None else f"{self.timeliness_limit}%"
    )
    return (
      f"RT_JOINT_TORQUE @ 1 kHz, gravity comp by the robot (faithful to sim); "
      f"timeliness limit {limit}"
    )

  def start(self) -> None:
    if self.timeliness_limit is not None:
      self.robot.SetTimelinessFailureLimit(float(self.timeliness_limit))
    self.robot.SwitchMode(self.adapter.Mode.RT_JOINT_TORQUE)

  def send(self, tau_residual: np.ndarray) -> None:
    tau = [float(t) for t in tau_residual]
    try:
      self.robot.StreamJointTorque(tau, True, True)
    except TypeError:  # older/newer binding: keyword form
      self.robot.StreamJointTorque(
        tau, enable_gravity_comp=True, enable_soft_limits=True
      )


class JointImpedanceBackend:
  """``NRT_JOINT_IMPEDANCE``: emulate a torque source with a position offset.

  The robot runs ``tau = K_q (q_d - q) - D_q qdot + g(q)``, so ``q_d = q +
  tau_res / K_q`` delivers ``tau_res`` to first order.  Approximate — see the
  module docstring for what this costs.
  """

  send_hz = POLICY_HZ

  def __init__(
    self,
    adapter: RizonAdapter,
    kq_scale: float = KQ_SCALE,
    max_offset: float = MAX_JOINT_OFFSET,
  ):
    self.adapter = adapter
    self.robot = adapter.robot
    self.k_q = adapter.kq_nom * kq_scale
    self.max_offset = max_offset
    self.tau_ceiling = self.k_q * max_offset

  @staticmethod
  def supported(adapter: RizonAdapter) -> bool:
    return hasattr(adapter.Mode, "NRT_JOINT_IMPEDANCE")

  def describe(self) -> str:
    return (
      f"NRT_JOINT_IMPEDANCE @ {self.send_hz:.0f} Hz, K_q="
      f"{self.k_q.round(0)}, |q_d - q| <= {self.max_offset} rad "
      f"=> torque ceiling {self.tau_ceiling.round(1)} Nm (vs limit "
      f"{EFFORT_LIMIT.round(0)})"
    )

  def start(self) -> None:
    self.robot.SwitchMode(self.adapter.Mode.NRT_JOINT_IMPEDANCE)
    self.robot.SetJointImpedance([float(k) for k in self.k_q])

  def send(self, tau_residual: np.ndarray) -> None:
    q = np.asarray(self.robot.states().q, dtype=float)[:NUM_JOINTS]
    offset = np.clip(tau_residual / self.k_q, -self.max_offset, self.max_offset)
    target = q + offset
    self.robot.SendJointPosition(
      [float(v) for v in target],
      [0.0] * NUM_JOINTS,
      [NRT_MAX_VEL] * NUM_JOINTS,
      [NRT_MAX_ACC] * NUM_JOINTS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safety guards.  Any trip means: stop commanding, drop to IDLE, exit.
# ─────────────────────────────────────────────────────────────────────────────


def check_guards(st: RobotState, adapter: RizonAdapter | None) -> str | None:
  """Return a reason string if a safety limit is violated, else ``None``."""
  if not np.all(np.isfinite(st.q)) or not np.all(np.isfinite(st.tau_meas)):
    return "non-finite robot state"
  if np.any(st.ee_pos < EE_GUARD_MIN) or np.any(st.ee_pos > EE_GUARD_MAX):
    return f"EE left the keep-in box: {st.ee_pos.round(3)}"
  if adapter is None:
    return None
  if adapter.robot.fault():
    return "robot reported a fault"
  dq_limit = DQ_GUARD_FRAC * adapter.dq_max
  if np.any(np.abs(st.dq) > dq_limit):
    worst = int(np.argmax(np.abs(st.dq) - dq_limit))
    return f"joint{worst + 1} speed {st.dq[worst]:.2f} > {dq_limit[worst]:.2f} rad/s"
  lo = adapter.q_min + Q_MARGIN
  hi = adapter.q_max - Q_MARGIN
  if np.any(st.q < lo) or np.any(st.q > hi):
    worst = int(np.argmax(np.maximum(lo - st.q, st.q - hi)))
    return f"joint{worst + 1} at {math.degrees(st.q[worst]):.1f} deg, near its limit"
  return None


# ─────────────────────────────────────────────────────────────────────────────
# Main loop.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DeployCfg:
  spec: PolicySpec
  goal_pos: np.ndarray
  onnx_path: Path
  steps: int
  torque_scale: float = 1.0
  ramp_steps: int = 0
  log_every: int = 25


def run(
  cfg: DeployCfg,
  adapter: RizonAdapter | None,
  backend: RtTorqueBackend | JointImpedanceBackend | None,
  confirm: bool = True,
) -> None:
  spec = cfg.spec
  actor = Actor(cfg.onnx_path)
  last_action = np.zeros(ACT_DIM, dtype=np.float32)  # sim reset: raw_action = 0
  tau_residual = np.zeros(NUM_JOINTS)

  send_hz = backend.send_hz if backend is not None else POLICY_HZ
  ticks_per_policy = max(1, int(round(send_hz / POLICY_HZ)))
  tick_dt = 1.0 / send_hz

  if adapter is not None:
    st0 = adapter.read_state()
    q_err = np.abs(st0.q - spec.q_default)
    print(
      f"\n  policy            : {spec.name} — {spec.description}"
      f"\n  backend           : {backend.describe() if backend else 'none'}"
      f"\n  goal (base frame) : {cfg.goal_pos.round(3)}"
      f"\n  start EE (grasp)  : {st0.ee_pos.round(3)}"
      f"  dgoal={np.linalg.norm(cfg.goal_pos - st0.ee_pos):.3f} m"
      f"\n  q vs home (deg)   : {np.degrees(st0.q - spec.q_default).round(1)}"
      f"\n  torque scale      : {cfg.torque_scale}"
      f"\n  residual ramp     : {cfg.ramp_steps} steps"
      f"\n  steps             : {cfg.steps} ({cfg.steps / POLICY_HZ:.1f} s)"
    )
    if q_err.max() > 0.10:
      worst = int(np.argmax(q_err))
      print(
        f"\n  WARNING: joint{worst + 1} is {math.degrees(q_err[worst]):.1f} deg "
        f"from the policy's home pose. joint_pos_rel is measured against it, so "
        f"the observation is out of distribution. Run home_rizon.py --policy "
        f"{spec.name} first."
      )
    if confirm and input("\nStart policy on the REAL arm? type 'go': ").strip() != "go":
      print("Aborted.")
      return
    assert backend is not None
    backend.start()

  step = 0
  tick = 0
  overruns = 0
  worst_late = 0.0
  stop_reason = "completed"
  t_next = time.perf_counter()
  try:
    while step < cfg.steps:
      # Sense once per *policy* step, as sim does: the inner ticks only re-send
      # the held torque (the sim's decimation), they do not re-observe.
      if tick % ticks_per_policy == 0:
        if adapter is not None:
          st = adapter.read_state()
        else:  # --dry-run: a static fake state, to exercise the pipeline
          st = state_from_joints(
            spec.q_default.copy(), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), spec
          )

        reason = check_guards(st, adapter)
        if reason is not None:
          stop_reason = f"GUARD TRIPPED: {reason}"
          break

        obs = build_observation(st, cfg.goal_pos, last_action, spec)
        raw_action = actor(obs)
        ramp = (
          1.0 if cfg.ramp_steps <= 0 else min(1.0, (step + 1) / float(cfg.ramp_steps))
        )
        tau_residual = decode_action(raw_action, ramp, cfg.torque_scale)
        last_action = raw_action  # cache the RAW action as next last_action obs

        if step % cfg.log_every == 0:
          print(
            f"[{step:5d}] EE={st.ee_pos.round(3)}  "
            f"dgoal={np.linalg.norm(cfg.goal_pos - st.ee_pos):.3f}m  "
            f"|tau_cmd|={np.linalg.norm(tau_residual):6.1f}  "
            f"|tau_meas|={np.linalg.norm(st.tau_meas):6.1f}  "
            f"|dq|={np.linalg.norm(st.dq):.2f}"
          )
        step += 1

      if backend is not None:
        backend.send(tau_residual)

      tick += 1
      t_next += tick_dt
      sleep = t_next - time.perf_counter()
      if sleep > 0:
        time.sleep(sleep)
      else:
        # Missed the deadline. In RT mode the robot counts this as a late
        # command and faults past its timeliness limit, so it is worth
        # measuring rather than silently swallowing.
        overruns += 1
        worst_late = max(worst_late, -sleep)
        t_next = time.perf_counter()  # resync rather than spiral
  except KeyboardInterrupt:
    stop_reason = "interrupted by the operator"
  finally:
    if adapter is not None:
      # Fade the residual out over 0.2 s so the arm settles onto the pure
      # gravity hold instead of dropping the command in one step.
      if backend is not None:
        for i in range(int(0.2 * send_hz), 0, -1):
          backend.send(tau_residual * (i / (0.2 * send_hz)))
          time.sleep(tick_dt)
      adapter.stop()
  print(f"\nStopped after {step} policy steps: {stop_reason}")
  if tick:
    print(
      f"Loop timing at {send_hz:.0f} Hz: {overruns}/{tick} ticks late "
      f"({100.0 * overruns / tick:.1f}%), worst {worst_late * 1e3:.2f} ms."
    )
    if overruns / tick > 0.02 and send_hz > POLICY_HZ:
      print(
        "  That is above the robot's default 2% timeliness budget. Use a "
        "low-latency/PREEMPT_RT kernel, pin this process (chrt/taskset), or "
        "raise --timeliness-limit knowing the robot will act on stale commands."
      )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: the FK against poses the task code states independently.
# ─────────────────────────────────────────────────────────────────────────────


def self_test() -> None:
  """Check the FK transcription against known sim geometry."""
  print("Forward-kinematics self-test (no robot, no ONNX).\n")
  ok = True

  q = POLICIES["tracking"].q_default
  ee = forward_kinematics(q).point(_EE_LOCAL)
  # env_cfg._HOME_ARM_POSE documents the *gripper* grasp site at
  # (0.38, 0.00, 0.24), and the bare flange EE as ~17.5 cm higher (the tool axis
  # points down at this top-down pose), i.e. ~(0.38, 0.00, 0.415).
  want = np.array([0.38, 0.00, 0.415])
  err = float(np.abs(ee - want).max())
  ok &= err < 0.01
  print(f"  tracking home grasp_site = {ee.round(4)}  (expected ~{want}) ")
  print(
    f"    max component error {err * 1000:.1f} mm -> {'OK' if err < 0.01 else 'FAIL'}"
  )

  q = POLICIES["reach"].q_default
  ee = forward_kinematics(q).point(_EE_LOCAL)
  # Reach is quoted from the shoulder (the joint1/joint2 origin at z = 0.155),
  # not from the base plate: 820 mm for the Rizon 4S.
  shoulder = float(np.linalg.norm(ee - np.array([0.0, 0.0, 0.155])))
  ok &= shoulder < 0.82
  print(f"\n  reach home grasp_site    = {ee.round(4)}")
  print(
    f"    {shoulder:.3f} m from the shoulder, within the 0.82 m reach -> "
    f"{'OK' if shoulder < 0.82 else 'FAIL'}"
  )
  print(
    f"    NOTE: {100 * shoulder / 0.82:.0f}% extended, and "
    f"{np.linalg.norm(ee):.3f} m from the base — outside ReachCommandCfg's "
    "[0.35, 0.80] goal shell.\n    The reach policy starts near full extension "
    "and works inward; gravity torques are\n    largest and the arm is closest "
    "to its boundary singularity exactly at t=0."
  )

  # A finite-difference check of the analytic point velocity.
  rng = np.random.default_rng(0)
  q = POLICIES["reach"].q_default + rng.uniform(-0.3, 0.3, NUM_JOINTS)
  dq = rng.uniform(-1.0, 1.0, NUM_JOINTS)
  h = 1e-6
  for label, local in (("grasp_site", _EE_LOCAL), ("link7 COM", _LINK7_COM_LOCAL)):
    analytic = forward_kinematics(q).point_vel(local, dq)
    numeric = (
      forward_kinematics(q + h * dq).point(local)
      - forward_kinematics(q - h * dq).point(local)
    ) / (2 * h)
    err = float(np.abs(analytic - numeric).max())
    ok &= err < 1e-6
    print(f"\n  {label} velocity: analytic {analytic.round(5)}")
    print(
      f"    vs finite difference, max error {err:.2e} -> "
      f"{'OK' if err < 1e-6 else 'FAIL'}"
    )

  print(f"\n{'ALL CHECKS PASSED' if ok else 'SELF-TEST FAILED'}")
  if not ok:
    raise SystemExit(1)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--policy", choices=sorted(POLICIES), default="tracking")
  ap.add_argument("--robot-sn", default="Rizon4s-063501")
  ap.add_argument("--onnx", type=Path, default=None, help="override the ONNX path")
  ap.add_argument("--self-test", action="store_true", help="FK checks only, no robot")
  ap.add_argument("--dry-run", action="store_true", help="no robot; test the pipeline")
  ap.add_argument("--probe", action="store_true", help="print states + FK, then exit")
  ap.add_argument("--steps", type=int, default=None, help="policy steps (100 Hz)")
  ap.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
  ap.add_argument(
    "--backend",
    choices=("auto", "rt_torque", "joint_impedance"),
    default="auto",
    help="auto: RT_JOINT_TORQUE if the wheel has it, else NRT_JOINT_IMPEDANCE",
  )
  ap.add_argument(
    "--torque-scale",
    type=float,
    default=1.0,
    help="safety derate on the residual torque; 1.0 = as trained. Gravity "
    "compensation is unaffected, so <1 makes the arm react more weakly, not sag",
  )
  ap.add_argument(
    "--ramp-steps",
    type=int,
    default=None,
    help="ramp the residual in over N policy steps after start (default: the "
    "policy's own warm_start_steps; 25 is a good value for a first run)",
  )
  ap.add_argument(
    "--timeliness-limit",
    type=float,
    default=None,
    metavar="PCT",
    help="rt_torque backend: percentage of late commands the robot tolerates "
    "before faulting (RDK default 2.0). Raise only after seeing the overrun "
    "figure this script prints on exit",
  )
  ap.add_argument(
    "--kq-scale",
    type=float,
    default=KQ_SCALE,
    help="joint_impedance backend: multiplier on the robot's nominal K_q",
  )
  ap.add_argument(
    "--max-offset",
    type=float,
    default=MAX_JOINT_OFFSET,
    help="joint_impedance backend: cap on |q_d - q| (rad)",
  )
  ap.add_argument("--yes", action="store_true", help="skip the pre-motion prompt")
  args = ap.parse_args()

  if args.self_test:
    self_test()
    return

  spec = POLICIES[args.policy]
  goal = spec.default_goal if args.goal is None else np.array(args.goal, dtype=float)
  lo, hi = spec.goal_box
  if np.any(goal < lo) or np.any(goal > hi):
    print(
      f"[WARN] goal {goal.round(3)} is outside the box the {spec.name} policy was "
      f"trained on ({lo} .. {hi}); the behaviour is extrapolation."
    )
  cfg = DeployCfg(
    spec=spec,
    goal_pos=goal,
    onnx_path=args.onnx or spec.onnx_path,
    steps=args.steps if args.steps is not None else spec.default_steps,
    torque_scale=args.torque_scale,
    ramp_steps=spec.warm_start_steps if args.ramp_steps is None else args.ramp_steps,
  )

  if args.dry_run:
    run(cfg, adapter=None, backend=None, confirm=False)
    return

  adapter = RizonAdapter(args.robot_sn, spec)
  if args.probe:
    adapter.probe()
    return

  rt_ok = RtTorqueBackend.supported(adapter)
  backend: RtTorqueBackend | JointImpedanceBackend
  if args.backend == "rt_torque" or (args.backend == "auto" and rt_ok):
    if not rt_ok:
      raise SystemExit(
        "This flexivrdk wheel has no RT layer (Mode has no RT_JOINT_TORQUE and "
        "Robot has no StreamJointTorque). Every macOS wheel that supports the "
        "Rizon 4S ships without it — run from a Linux host with flexivrdk 1.9, "
        "or use --backend joint_impedance (approximate)."
      )
    backend = RtTorqueBackend(adapter, args.timeliness_limit)
  else:
    if not JointImpedanceBackend.supported(adapter):
      raise SystemExit("Neither RT_JOINT_TORQUE nor NRT_JOINT_IMPEDANCE is available.")
    backend = JointImpedanceBackend(adapter, args.kq_scale, args.max_offset)
    print(
      "\n[NOTE] Using the NRT joint-impedance emulation, not true torque control.\n"
      "       The robot's own joint damping is added on top of the policy's\n"
      "       torque and the command is interpolated at 100 Hz, so the arm will\n"
      "       feel stiffer and slower than sim. Good for checking the pipeline\n"
      "       and the gross behaviour; not for judging the compliance."
    )

  run(cfg, adapter=adapter, backend=backend, confirm=not args.yes)


if __name__ == "__main__":
  main()
