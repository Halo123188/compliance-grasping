"""Flexiv Rizon4S arm with the custom two-finger hand (mesh visual + COACD collision).

The two-finger hand (``assets/two_finger_hand``) is a fully-actuated 4-DOF
mechanism: each finger has two independently-driven hinge joints
(``left_1``/``left_2`` and ``right_1``/``right_2``, all about the local Y axis),
so there is no tendon coupling or mimic equality -- four joints, four actuators.

Here we load the hand as its own spec, rigidly mount its ``base_link`` (palm) on
the arm's ``link7`` using the CAD-derived flange transform, and re-expose the
grasp interface the way the other grippers do:

  * body  ``base_link``   -- the palm/mount frame on link7.
  * geoms ``<seg>_c<i>``  -- COACD convex parts of each finger segment act as
    the colliders (group 3).
  * joints ``left_1/left_2/right_1/right_2`` -- the four driven DOFs.
  * site  ``grasp_site``  -- grasp centre between the fingertips.

PLACEHOLDERS (replace once the real values are known):
  * Link masses/inertia come from a uniform PLA density (1240 kg/m^3) baked into
    the collision meshes -- see ``two_finger_hand.xml``. Re-export from Onshape
    with dynamics enabled for true inertials.
  * Actuator ``stiffness``/``damping``/``effort_limit`` are rough guesses pending
    the DC15 servo datasheet (torque/speed/gear ratio).
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

HAND_XML: Path = _ASSETS / "two_finger_hand" / "two_finger_hand.xml"
assert HAND_XML.exists(), f"Two-finger hand XML not found: {HAND_XML}"

# ── Mount pose of the hand base frame in link7's local frame ─────────────────
# Salvaged from the CAD stack model (rizon_hand.xml: hand_base under link7). The
# quat seats the palm against link7's flange with the fingers pointing along the
# tool approach axis.
_MOUNT_POS = (-0.00799, 0.00797, 0.15550)  # m, in link7 frame
_MOUNT_QUAT = (0.0, -0.923880, -0.382683, 0.0)  # wxyz

# Grasp site (lift-reward target) between the fingertips, in the base_link frame.
# The fingers hang along -z from the palm; the second segments reach ~-0.09 m.
# PLACEHOLDER: tune against the rendered jaws at a representative grasp aperture.
_GRASP_SITE_POS = (0.0, -0.0037, -0.09)

# ── Actuator gains (PLACEHOLDER -- pending DC15 servo datasheet) ──────────────
FINGER_STIFFNESS = 5.0  # N·m/rad
FINGER_DAMPING = 0.5  # N·m·s/rad
FINGER_EFFORT_LIMIT = 2.0  # N·m

_FINGER_JOINTS: tuple[str, ...] = ("left_1", "left_2", "right_1", "right_2")


def _strip_leading_slash(spec: mujoco.MjSpec) -> None:
  """Remove the leading "/" that MjSpec.attach() namespaces onto child names.

  The entity layer and contact sensors expect canonical names (e.g.
  "robot/base_link"), not the doubled "robot//base_link".
  """
  for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    body.name = body.name.lstrip("/")
  for j in spec.joints:
    j.name = j.name.lstrip("/")
  for g in spec.geoms:
    g.name = g.name.lstrip("/")
  for s in spec.sites:
    s.name = s.name.lstrip("/")


def _build_hand_spec() -> mujoco.MjSpec:
  """Load the two-finger hand and add the grasp site."""
  spec = mujoco.MjSpec.from_file(str(HAND_XML))

  base = next(b for b in spec.worldbody.bodies if b.name == "base_link")
  # Fold the whole mount transform into the link7 site; the base frame then
  # coincides with the mount site.
  base.pos = [0.0, 0.0, 0.0]
  base.quat = [1.0, 0.0, 0.0, 0.0]

  # Drop the hand's own actuators -- mjlab re-actuates the four finger joints via
  # FINGER_ACTUATOR after attach.
  for a in list(spec.actuators):
    spec.delete(a)

  grasp_site = base.add_site()
  grasp_site.name = "grasp_site"
  grasp_site.pos = list(_GRASP_SITE_POS)
  grasp_site.size = [0.01, 0.0, 0.0]
  grasp_site.rgba = [1.0, 0.0, 0.0, 0.9]

  return spec


def get_spec() -> mujoco.MjSpec:
  """Load the Flexiv arm and rigidly mount the two-finger hand on link7."""
  spec = mujoco.MjSpec.from_file(str(FLEXIV_XML))

  link7 = next(
    (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "link7"),
    None,
  )
  assert link7 is not None, "link7 not found in Flexiv spec"

  mount = link7.add_site()
  mount.name = "hand_mount"
  mount.pos = list(_MOUNT_POS)
  mount.quat = list(_MOUNT_QUAT)

  hand = _build_hand_spec()
  spec.attach(hand, site="hand_mount")
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

FINGER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_FINGER_JOINTS,
  stiffness=FINGER_STIFFNESS,
  damping=FINGER_DAMPING,
  effort_limit=FINGER_EFFORT_LIMIT,
)

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
    "left_1": 0.0,
    "left_2": 0.0,
    "right_1": 0.0,
    "right_2": 0.0,
  },
  joint_vel={".*": 0.0},
)

# All COACD convex parts (named "<segment>_c<i>") are the colliders.
#
# Non-interpenetration is guaranteed two ways that reinforce each other:
#   1. Joint limits (in the XML) cap the proximal joints at ~0.28 rad, the angle
#      at which the two fingertips just meet at the grasp centre (measured: they
#      begin to cross past ~0.30).  The fingers therefore *cannot* rotate into a
#      crossed configuration.
#   2. Self-collision is left ENABLED (default contype/conaffinity) so finger-vs-
#      finger contact is a physical backstop.  The coarse COACD hulls first touch
#      right at the visual contact angle (~0.25 rad), so this stops the jaws at
#      the moment the real surfaces meet rather than firing spuriously early.
# Together: the limits keep the joints in a non-crossing range, and contact halts
# them exactly at touch even under actuator force.
HAND_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_c\d+$",),
  condim={r".*_c\d+$": 6},
  friction={r".*_c\d+$": (1.0, 5e-3, 5e-4)},
  disable_other_geoms=True,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_ACTUATOR, FINGER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_two_finger_hand_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(HAND_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )
