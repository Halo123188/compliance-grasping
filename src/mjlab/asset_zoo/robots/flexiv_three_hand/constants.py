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

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
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
