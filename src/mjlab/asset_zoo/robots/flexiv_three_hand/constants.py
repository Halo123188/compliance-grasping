"""Flexiv Rizon4S arm with UMI parallel-jaw gripper (mesh visual + collision).

The UMI gripper (``assets/umi_gripper``) ships as a *free-floating*
model: its root body carries six positional DOFs (``gripper_joint_x/y/z`` plus
``rx/ry/rz``) plus matching actuators that fly the gripper through space, and a
tendon-coupled finger pair driven by a single ``fingers_actuator``.  Here we
strip the floating base and the cosmetic hardware (GoPro, mirrors, aruco
stickers, screws, rails) but keep the base shell so the gripper visually bridges
the wrist to the jaws, rigidly mount it on the arm's ``link7``, and re-expose it
the same way the previous myCobot gripper was:

  * body ``gripper_base``  — the mount frame on link7 (contact-sensor target).
  * geoms ``left_finger_col`` / ``right_finger_col`` — the finger meshes act as
    the colliders.
  * site ``grasp_site``    — grasp centre between the jaws.
  * joint ``left_finger_joint`` — the single driven DOF; ``right_finger_joint``
    mirrors it via an mjEQ_JOINT equality so the jaws stay symmetric.

Keeping those names identical means the pick-and-place env config is unchanged.
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
  XmlActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Repo-relative asset root: <repo>/assets (this file is 5 levels below the root).
_ASSETS: Path = Path(__file__).resolve().parents[5] / "assets"

FLEXIV_XML: Path = _ASSETS / "flexiv_rizon4s" / "flexiv_rizon4s.xml"
assert FLEXIV_XML.exists(), f"Flexiv XML not found: {FLEXIV_XML}"

UMI_GRIPPER_XML: Path = _ASSETS / "umi_gripper" / "umi_gripper.xml"
assert UMI_GRIPPER_XML.exists(), f"UMI gripper XML not found: {UMI_GRIPPER_XML}"

# ── Mount pose of the gripper base frame in link7's local frame ──────────────
# link7's +z is the tool approach axis (same convention the myCobot mount used).
# After we zero the UMI base body's own pose, the gripper frame == this site
# frame: jaws point along +z (link7's approach axis).  This small offset seats
# the UMI base shell's top right against link7's flange so it bridges the wrist
# to the jaws (the shell has no collider, so a flush visual seam is fine).
_MOUNT_POS = (0.0, 0.0, 0.05)  # m, in link7 frame
_MOUNT_QUAT = (1.0, 0.0, 0.0, 0.0)  # wxyz; identity == aligned with link7

# ── Finger DOF ──────────────────────────────────────────────────────────────
# left/right_finger_joint slide range is 0 (fully open, jaw gap ~87 mm) to
# 0.041 (fully closed).  The cap stops just before the fingertips interpenetrate
# (collision distance crosses zero at ~0.0425).  A 40 mm cube is gripped at ~0.02.
_JOINT_RANGE = 0.041  # m

# Grasp site (lift-reward target) between the jaws, in the gripper_base frame.
# Placed near the fingertip closing zone where a grasped object is held (tuned
# against the rendered jaws at a ~40 mm grasp aperture).
_GRASP_SITE_POS = (0.0, 0.003, 0.175)

GRIPPER_STIFFNESS = 200.0  # N/m
GRIPPER_DAMPING = 20.0  # N·s/m
GRIPPER_EFFORT_LIMIT = 20.0  # N

# ── Force/torque sensor mounting sites ──────────────────────────────────────
# One 6-axis FT sensor per finger, mounted on the inner contact face of each
# finger holder near the collider. The site's body is the finger holder, so a
# MuJoCo force/torque sensor here reads the wrench transmitted between the finger
# and the gripper base — i.e. the grasp contact wrench. The finger joint slides
# along the holder-frame X axis, so the site's +X is the closing / contact-normal
# direction. Placement is on each finger collider's inner face (X nudged toward
# the jaw centre); the force reading itself is position-invariant.
FT_SITE_NAMES: tuple[str, str] = ("left_ft_site", "right_ft_site")
FT_NORMAL_AXIS: int = 0  # holder-frame X == finger closing / contact-normal axis
FT_FORCE_SENSOR_NAMES: tuple[str, str] = ("ft_force_left", "ft_force_right")
FT_TORQUE_SENSOR_NAMES: tuple[str, str] = ("ft_torque_left", "ft_torque_right")
_FT_SITE_SIZE = 0.008  # visual radius (m)
_LEFT_FT_POS = (-0.0435, 0.1244, -0.003)  # left_finger_holder frame, inner face
_RIGHT_FT_POS = (0.0435, 0.1244, -0.003)  # right_finger_holder frame, inner face

_FLOATING_JOINTS = (
  "gripper_joint_x",
  "gripper_joint_y",
  "gripper_joint_z",
  "gripper_joint_rx",
  "gripper_joint_ry",
  "gripper_joint_rz",
)

# Cosmetic geoms to drop (everything that isn't the base shell or the jaws).
_COSMETIC_GEOMS = (
  "linear_guide_rail",
  "left_mirror",
  "right_mirror",
  "left_marker",
  "right_marker",
  "left_rail_block",
  "right_rail_block",
)


def _strip_leading_slash(spec: mujoco.MjSpec) -> None:
  """Remove the leading "/" that MjSpec.attach() namespaces onto child names.

  Mirrors elephant_hand: the entity layer and contact sensor expect canonical
  names (e.g. "robot/gripper_base"), not the doubled "robot//gripper_base".
  """
  for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    body.name = body.name.lstrip("/")
  for j in spec.joints:
    j.name = j.name.lstrip("/")
  for g in spec.geoms:
    g.name = g.name.lstrip("/")
  for s in spec.sites:
    s.name = s.name.lstrip("/")
  for eq in spec.equalities:
    eq.name = eq.name.lstrip("/")
    eq.name1 = eq.name1.lstrip("/")
    eq.name2 = eq.name2.lstrip("/")


def _build_gripper_spec() -> mujoco.MjSpec:
  """Load the UMI gripper and reduce it to a rigidly-mountable jaw mechanism."""
  spec = mujoco.MjSpec.from_file(str(UMI_GRIPPER_XML))

  # The single body directly under worldbody (unnamed) is the floating base.
  base = next(b for b in spec.worldbody.bodies if b.name in ("", "gripper_base"))
  base.name = "gripper_base"
  # Fold the whole mount transform into the link7 site (set later); the base
  # frame then coincides with the mount site.
  base.pos = [0.0, 0.0, 0.0]
  base.quat = [1.0, 0.0, 0.0, 0.0]

  # Drop the 6 floating DOFs, every actuator, and the finger tendon — the arm
  # positions the gripper and we re-actuate the jaws with one driven joint.
  for j in list(spec.joints):
    if j.name in _FLOATING_JOINTS:
      spec.delete(j)
  for a in list(spec.actuators):
    spec.delete(a)
  for t in list(spec.tendons):
    spec.delete(t)
  for c in list(spec.cameras):
    spec.delete(c)

  # Keep the base shell + holders + jaws as visuals; drop only the cosmetic
  # hardware (GoPro, mirrors, aruco, screws, rails) and every collision mesh
  # EXCEPT the two finger pads — the fingers themselves are the colliders
  # (their own mesh, not a box).
  for g in list(spec.geoms):
    is_finger = g.meshname in ("left_finger", "right_finger")
    if is_finger:
      continue  # keep finger visual + collision; renamed below
    is_collision_mesh = g.group == 3
    is_screw = g.name.endswith("screw")
    is_gopro = g.meshname == "gopro"
    if is_collision_mesh or is_screw or is_gopro or g.name in _COSMETIC_GEOMS:
      spec.delete(g)

  # Name the finger collision meshes so the CollisionCfg can target them.
  for g in spec.geoms:
    if g.group == 3 and g.meshname == "left_finger":
      g.name = "left_finger_col"
    elif g.group == 3 and g.meshname == "right_finger":
      g.name = "right_finger_col"

  # Re-range the jaws and let the position actuator (not passive stiffness)
  # drive them.
  for joint_name in ("left_finger_joint", "right_finger_joint"):
    joint = next(j for j in spec.joints if j.name == joint_name)
    joint.range = [0.0, _JOINT_RANGE]
    joint.stiffness = [0.0, 0.0, 0.0]

  # Grasp site between the jaws.
  grasp_site = base.add_site()
  grasp_site.name = "grasp_site"
  grasp_site.pos = list(_GRASP_SITE_POS)
  grasp_site.size = [0.01, 0.0, 0.0]
  grasp_site.rgba = [1.0, 0.0, 0.0, 0.9]

  # FT sensor sites, one on each finger holder's inner contact face.
  for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    if body.name == "left_finger_holder":
      s = body.add_site()
      s.name, s.pos, s.rgba = "left_ft_site", list(_LEFT_FT_POS), [0.0, 0.9, 0.9, 1.0]
      s.type, s.size = mujoco.mjtGeom.mjGEOM_SPHERE, [_FT_SITE_SIZE, 0.0, 0.0]
    elif body.name == "right_finger_holder":
      s = body.add_site()
      s.name, s.pos, s.rgba = "right_ft_site", list(_RIGHT_FT_POS), [0.9, 0.0, 0.9, 1.0]
      s.type, s.size = mujoco.mjtGeom.mjGEOM_SPHERE, [_FT_SITE_SIZE, 0.0, 0.0]

  # Mirror the right jaw onto the left so a single command closes both.
  eq = spec.add_equality()
  eq.type = mujoco.mjtEq.mjEQ_JOINT
  eq.name1 = "right_finger_joint"
  eq.name2 = "left_finger_joint"  # reference (driven) joint
  eq.data[0] = 0.0
  eq.data[1] = 1.0  # right_q = +1 × left_q

  return spec


def get_spec() -> mujoco.MjSpec:
  """Load the Flexiv arm and rigidly mount the UMI jaw mechanism on link7."""
  spec = mujoco.MjSpec.from_file(str(FLEXIV_XML))

  link7 = next(
    (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "link7"),
    None,
  )
  assert link7 is not None, "link7 not found in Flexiv spec"

  mount = link7.add_site()
  mount.name = "gripper_mount"
  mount.pos = list(_MOUNT_POS)
  mount.quat = list(_MOUNT_QUAT)

  gripper = _build_gripper_spec()
  spec.attach(gripper, site="gripper_mount")
  _strip_leading_slash(spec)

  return spec


ARM_ACTUATOR = XmlActuatorCfg(
  target_names_expr=(
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
  ),
)

GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("left_finger_joint",),
  stiffness=GRIPPER_STIFFNESS,
  damping=GRIPPER_DAMPING,
  effort_limit=GRIPPER_EFFORT_LIMIT,
)

# q=0 → fully open; equality drives right_finger_joint automatically.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 1.57,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
    "left_finger_joint": 0.0,
    "right_finger_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=("left_finger_col", "right_finger_col"),
  condim={
    "left_finger_col": 6,
    "right_finger_col": 6,
  },
  friction={
    "left_finger_col": (1.0, 5e-3, 5e-4),
    "right_finger_col": (1.0, 5e-3, 5e-4),
  },
  solref={
    "left_finger_col": (0.01, 1),
    "right_finger_col": (0.01, 1),
  },
  priority={
    "left_finger_col": 1,
    "right_finger_col": 1,
  },
  disable_other_geoms=True,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_ACTUATOR, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_flexiv_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


# ── Torque-controlled variant (for Cartesian impedance control) ──────────────
# The stock arm ships with <position> actuators (joint-space PD). Cartesian
# impedance control needs joint *torque* input, so this variant strips those and
# drives each arm joint as a <motor>. Per-joint torque limits (Rizon4S datasheet
# order): J1..J7 = 123/123/64/64/39/39/39 Nm.
FLEXIV_ARM_EFFORT_LIMIT: tuple[float, ...] = (
  123.0,
  123.0,
  64.0,
  64.0,
  39.0,
  39.0,
  39.0,
)
_ARM_JOINTS: tuple[str, ...] = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
)


def get_torque_spec() -> mujoco.MjSpec:
  """Like :func:`get_spec` but with the arm's position actuators removed.

  The arm joints are re-actuated as motors by ``ARM_MOTOR_ACTUATOR``; the gripper
  keeps its builtin position actuator.
  """
  spec = get_spec()
  for a in list(spec.actuators):
    if a.name.lstrip("/") in _ARM_JOINTS:
      spec.delete(a)
  return spec


ARM_MOTOR_ACTUATOR = BuiltinMotorActuatorCfg(
  target_names_expr=_ARM_JOINTS,
  # Generous headroom; the impedance controller clips to FLEXIV_ARM_EFFORT_LIMIT
  # per joint before sending, so this never binds first.
  effort_limit=200.0,
)

TORQUE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_MOTOR_ACTUATOR, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_flexiv_torque_robot_cfg() -> EntityCfg:
  """Flexiv + UMI with torque (motor) arm actuators for impedance control."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_torque_spec,
    articulation=TORQUE_ARTICULATION,
  )


# ── Bare arm (no UMI gripper) ────────────────────────────────────────────────
# The compliance reach task needs no gripper: the "human" is a virtual wrench,
# not a grasped object, and the end-effector is just the flange tool point.  This
# variant loads only the Flexiv arm and exposes a ``grasp_site`` on link7 so the
# reach command / EE observation resolve exactly as they do with the gripper.
# EE frame sits a short way down link7's local +z (its tool approach axis), where
# a flange tool plate would be -- shorter reach than the ~0.22 m UMI stack.
_BARE_EE_POS = (0.0, 0.0, 0.05)  # m, link7 local frame (tool approach axis)


def get_bare_torque_spec() -> mujoco.MjSpec:
  """Flexiv arm alone, torque-actuated, with a ``grasp_site`` EE on link7."""
  spec = mujoco.MjSpec.from_file(str(FLEXIV_XML))

  link7 = next(
    (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "link7"),
    None,
  )
  assert link7 is not None, "link7 not found in Flexiv spec"
  ee = link7.add_site()
  ee.name = "grasp_site"
  ee.pos = list(_BARE_EE_POS)
  ee.size = [0.01, 0.0, 0.0]
  ee.rgba = [1.0, 0.0, 0.0, 0.9]

  # Strip the stock position actuators so the arm joints take motor torque.
  for a in list(spec.actuators):
    if a.name.lstrip("/") in _ARM_JOINTS:
      spec.delete(a)
  return spec


# Home pose without the finger joints (they do not exist on the bare arm).
BARE_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 1.57,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
  },
  joint_vel={".*": 0.0},
)

BARE_TORQUE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_MOTOR_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_flexiv_bare_torque_robot_cfg() -> EntityCfg:
  """Flexiv Rizon 4S alone (no gripper), torque-actuated for impedance control."""
  return EntityCfg(
    init_state=BARE_HOME_KEYFRAME,
    collisions=(),  # free-space reach; the human wrench is virtual, no contacts
    spec_fn=get_bare_torque_spec,
    articulation=BARE_TORQUE_ARTICULATION,
  )


# ── Hardware-limited bare arm (sim-to-real velocity fix) ─────────────────────
# MuJoCo does not enforce joint velocity limits, so a torque policy learns
# motions no real actuator can execute (the wrist-roll null-space DOF spun to
# >150 rad/s in sim, tripping the deploy velocity guardrail). Two structural
# fixes bake the hardware envelope into training:
#   * dof damping on each arm joint, sized so full torque reaches ~dq_max, not
#     infinity -- a physically honest torque-speed droop the policy cannot game;
#   * j7 (wrist roll) is null-space for the EE task, so it is locked: near-zero
#     torque authority (via the action effort limit below) plus heavy damping.
# Rizon4S rated max joint speed (rad/s), datasheet order J1..J7:
FLEXIV_ARM_DQ_MAX: tuple[float, ...] = (2.09, 2.09, 2.44, 2.44, 4.89, 4.89, 4.89)

# Physical dof damping d = effort_limit / dq_max, so sustained full torque
# reaches the rated speed rather than diverging. j7 gets heavy damping to freeze
# the (locked) wrist roll.
_ARM_DAMPING_PHYS: tuple[float, ...] = tuple(
  e / v for e, v in zip(FLEXIV_ARM_EFFORT_LIMIT[:6], FLEXIV_ARM_DQ_MAX[:6], strict=True)
) + (60.0,)

# Effort limit with j7 authority zeroed out (locked roll): the policy's j7 output
# is clamped to ~gravity-comp only; damping holds it at home.
FLEXIV_ARM_EFFORT_LIMIT_J7LOCK: tuple[float, ...] = FLEXIV_ARM_EFFORT_LIMIT[:6] + (2.0,)


def _make_bare_hwlimited_spec_fn(damping_scale: float):
  """Bare torque spec with dof damping on the arm joints (scaled)."""

  def _spec() -> mujoco.MjSpec:
    spec = get_bare_torque_spec()
    damp = {
      nm: _ARM_DAMPING_PHYS[i] * damping_scale for i, nm in enumerate(_ARM_JOINTS)
    }
    for j in spec.joints:
      nm = j.name.lstrip("/")
      if nm in damp:
        # MjSpec stores hinge damping as a 3-vector; index 0 is the DOF.
        j.damping = np.array([damp[nm], 0.0, 0.0])
    return spec

  return _spec


def get_flexiv_bare_torque_hwlimited_robot_cfg(
  damping_scale: float = 1.0,
) -> EntityCfg:
  """Bare Flexiv arm with hardware-limited dof damping (j7 lock via effort cfg)."""
  return EntityCfg(
    init_state=BARE_HOME_KEYFRAME,
    collisions=(),
    spec_fn=_make_bare_hwlimited_spec_fn(damping_scale),
    articulation=BARE_TORQUE_ARTICULATION,
  )


def get_ft_sensor_cfgs(entity: str = "robot") -> tuple:
  """Two 6-axis FT sensors: a force+torque pair on each fingertip site.

  Each pair reads the wrench transmitted between the finger and the gripper base
  in the finger (site) frame — the grasp contact wrench. Returns four
  ``BuiltinSensorCfg`` (force_left, force_right, torque_left, torque_right); each
  ``force``/``torque`` sensor outputs a 3-vector, so the two FT sensors together
  give 12 values.
  """
  from mjlab.sensor import BuiltinSensorCfg, ObjRef

  cfgs = []
  for sensor_name, site_name in zip(FT_FORCE_SENSOR_NAMES, FT_SITE_NAMES, strict=True):
    cfgs.append(
      BuiltinSensorCfg(
        name=sensor_name,
        sensor_type="force",
        obj=ObjRef(type="site", name=site_name, entity=entity),
      )
    )
  for sensor_name, site_name in zip(FT_TORQUE_SENSOR_NAMES, FT_SITE_NAMES, strict=True):
    cfgs.append(
      BuiltinSensorCfg(
        name=sensor_name,
        sensor_type="torque",
        obj=ObjRef(type="site", name=site_name, entity=entity),
      )
    )
  return tuple(cfgs)


# Arm joints: 0.25 * effort_limit / stiffness (covers ~±25% of joint torque budget)
# Gripper: use _JOINT_RANGE directly so action=±1 maps to full open/close stroke
FLEXIV_ACTION_SCALE: dict[str, float] = {
  "joint1": 0.25 * 123 / 289,
  "joint2": 0.25 * 123 / 673,
  "joint3": 0.25 * 64 / 224,
  "joint4": 0.25 * 64 / 373,
  "joint5": 0.25 * 39 / 237,
  "joint6": 0.25 * 39 / 232,
  "joint7": 0.25 * 39 / 186,
  "left_finger_joint": _JOINT_RANGE,  # action=+1 → full close, action=0 → open
}
