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

# Grasp site (lift-reward target) between the fingertips, in the base_link frame,
# down near where the jaws close on an object.
# The fingertips reach ~22 mm below this site. At -0.090 the site sat so deep
# that driving it to a cube centre (15 mm above the table) put the fingertips
# 7.1 mm BELOW the table surface -- a pose the arm can never reach, so every
# approach rammed the tips into the table and shoved the cube ~4 cm away before
# the jaws could close. Measured with scripts/oracle_lift.py. At -0.099 the tips
# clear the table by ~2 mm when the site is at the cube centre, and the jaws
# straddle the cube. NOTE the sign: moving the site DOWN (more negative, toward
# the tips) makes the hand sit HIGHER for the same site target.
#
# The y is NOT zero: the two pads are not symmetric about the palm's y=0 plane.
# Measuring the midpoint of the left/right inner pad faces over the working
# range of left_1 puts the jaw centreline at y = +0.003..+0.0075 (mean ~0.005),
# x = 0.000. Leaving the site at y=0 biased every approach ~5 mm to one side, so
# one pad reached the cube first and shoved it instead of pinching -- visible in
# the env as a 48 N force on the right fingertip with 0 N on the left, and hence
# a bilateral-grasp reward of exactly zero.
_GRASP_SITE_POS = (0.0, 0.005, -0.099)

# The arm is bolted to the table top; mounting the base at this world height
# puts the whole arm + workspace on the table, with the terrain plane acting as
# the floor below. The task's table box (in the env cfg) uses the same height.
TABLE_HEIGHT = 0.40  # m

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


# ── Fixed RealSense D435 camera on the static camera mount ────────────────────
# Modelled as one aligned RGB-D camera at the D435 depth vertical FOV
# (58 deg -> ~87 deg horizontal at 16:9), on the as-modelled Static Camera Mount
# V1.4.3. Derived from the real mount chain in robot-actuator-model
# (urdf/rizon_baseplate_camera.urdf):
#
#   base_plate -> camera_mount        rpy (0, 0, -pi/2)      xyz (-0.092, 0, 0)
#   camera_mount -> camera_optical    rpy (-2.049889, 0, pi) xyz (0, -0.013912, 0.204879)
#
# which puts the optical frame at (-0.105912, 0, 0.204879) in the base-plate
# frame -- on the base's X axis (y = 0), 20.5 cm up -- looking 27.45 deg below
# horizontal (the mount's as-modelled bolt-face tilt, not a round-number guess).
#
# The one deviation: the real chain looks along -X, i.e. the physical workcell
# has its workspace *behind* the base plate, whereas this scene's table + cube
# region are on +X. We therefore yaw the whole mount assembly 180 deg about the
# base Z axis -- the same bracket bolted to the opposite side of the base plate.
# Every real dimension (0.106 m offset, 0.205 m height, 27.45 deg tilt) is
# preserved; only which side of the base it overlooks changes. The optical axis
# then meets the table top at x = 0.500, y = 0 -- the centre of the cube region.
_D435_POS = (0.105912, 0.0, 0.204879)  # m, in the arm base frame
_D435_QUAT = (
  0.604354,
  0.367092,
  -0.367092,
  -0.604354,
)  # wxyz, looks +X, 27.45 deg down
D435_FOVY = 58.0  # deg (depth module vertical FOV)


def get_spec() -> mujoco.MjSpec:
  """Load the Flexiv arm, mount the two-finger hand on link7, add the D435."""
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

  # Exclude self-collision between adjacent (jointed) finger links so bending a
  # finger doesn't self-collide. Opposing fingers (left vs right) are NOT
  # excluded, so their capsules still stop the fingers passing through each
  # other. Added after attach so the (namespaced-then-stripped) body names match.
  for b1, b2 in (
    ("base_link", "left_1"),
    ("base_link", "right_1"),
    ("left_1", "left_2"),
    ("right_1", "right_2"),
  ):
    spec.add_exclude(bodyname1=b1, bodyname2=b2)

  # Fixed D435 on the arm base (base sits at the entity origin).
  base = next(
    (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "base"),
    None,
  )
  assert base is not None, "arm base body not found in Flexiv spec"

  # NOTE: do NOT set base.pos here to mount the arm on the table. A fixed-base
  # entity is driven as a mocap body: the scene writes
  # ``mocap_pos = init_state.pos + env_origin`` on every reset, and that stacks
  # on top of the body's own pos -- setting both applies the table height twice.
  # ``EntityCfg.InitialStateCfg.pos`` (HOME_KEYFRAME / the task cfg) is the single
  # source of truth for where the arm is mounted.

  cam = base.add_camera()
  cam.name = "d435"
  cam.pos = list(_D435_POS)
  cam.quat = list(_D435_QUAT)
  cam.fovy = D435_FOVY

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
  # Fixed base: this pos IS the arm's world mount point (it overwrites the
  # spec's base.pos), so it must carry the table height.
  pos=(0.0, 0.0, TABLE_HEIGHT),
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
# The jaws can close fully: finger self-collision (below) is the physical
# backstop, so the fingers close until their surfaces meet and stop there rather
# than passing through each other.  The XML joint limits (+-0.42 proximal, +-0.55
# distal) are a generous safety cap, not the closing stop -- contact governs the
# actual close.  Self-collision is left ENABLED (default contype/conaffinity);
# the coarse COACD hulls first touch right at the visual contact angle, so
# closing halts exactly when the real surfaces meet.  Verified: a 60-gain
# actuator slammed to the proximal limit settles at surface contact
# (|proximal| < 0.32), no interpenetration blow-up.
# All hand colliders share contype=2 / conaffinity=1. Robot-vs-robot pairs then
# test (2 & 1) | (2 & 1) == 0 and DON'T collide, so the coarse COACD finger hulls
# never self-collide -- that mesh-vs-mesh case overflowed mujoco-warp's fixed EPA
# polytope ("EPA horizon = 24 isn't large enough") and returned garbage normals
# that spiked the solver into NaNs when the fingers closed in empty air. Hand vs
# world (cube/table/terrain, all contype=conaffinity=1) still tests to 1, so
# grasping and ground contact are unaffected. The fingers now close against the
# object (or their joint limits in empty air) rather than against each other.
HAND_COLLISION = CollisionCfg(
  # The colliders are the real finger meshes (group 3, named "<seg>_col"),
  # matching the stock UMI gripper's setup. MuJoCo collides them as convex hulls,
  # capped by maxhullvert in the hand XML to UMI-like complexity so mujoco-warp's
  # fixed EPA polytope does not overflow. This reproduces the true jaw geometry
  # exactly (verified 0.0 mm error vs the visual mesh across the whole range),
  # including the flat 27x21 mm inner pad and its 4.3 deg slope -- the capsule
  # primitives this replaces gave line contact instead of pad contact and read
  # 22 mm narrower than the real fingers when open.
  #
  # condim=6 + tuned friction give a grippy contact; solimp is hardened to match
  # the table/cube (see HARD_SOLIMP in the task cfg) so a pressed object does not
  # sink into whatever it is pressed against.
  geom_names_expr=(r".*_col$",),
  condim={r".*_col$": 6},
  friction={r".*_col$": (1.0, 5e-3, 5e-4)},
  solimp={r".*_col$": (0.98, 0.999, 0.001, 0.5, 2.0)},
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
