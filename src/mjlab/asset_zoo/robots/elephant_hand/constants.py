"""mygripper_H100_R ("elephant") hand — mjlab asset_zoo entity configs.

Two variants:
  * get_elephant_hand_cfg()  — standalone hand fixed at world origin (no arm,
    no MuJoCo Menagerie dependency).  Good for hand-only tasks or debugging.
  * get_elephant_arm_cfg()   — Flexiv Rizon 4S arm + elephant hand, same
    assembly as mygripper_arm_viewer.py --weld.  Requires the Rizon 4S XML at
    FLEXIV_XML (symlinked / checked out in compliance/assets/).

Physical model (weld variant):
  The URDF is a four-bar linkage cut open into a tree (15 joints, 3 duplicate
  coupler bodies).  We close each four-bar loop with one mjEQ_WELD constraint
  per finger and regularise the mass matrix via dof_armature=0.01 on every
  gripper joint.  The 6 driven DOFs (2 per finger: base swing + curl input) get
  BuiltinPositionActuatorCfg; the 9 passive loop joints are weld-slaved.
"""

from __future__ import annotations

import functools
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# ---------------------------------------------------------------------------
# Asset paths.
# ---------------------------------------------------------------------------

# Repo-relative asset root: <repo>/assets (this file is 5 levels below the root).
_ASSETS: Path = Path(__file__).resolve().parents[5] / "assets"

HAND_URDF: Path = (
  _ASSETS / "elephant-hand-mujoco/assets/robots/mygripper_H100_R/robot.urdf"
)
assert HAND_URDF.exists(), f"Elephant-hand URDF not found: {HAND_URDF}"

FLEXIV_XML: Path = _ASSETS / "flexiv_rizon4s" / "flexiv_rizon4s.xml"

# ---------------------------------------------------------------------------
# Weld (four-bar loop closure) parameters — ported from mygripper_arm_viewer.py.
# ---------------------------------------------------------------------------

# Tuned mount pose of the hand's root body (base_step) on link7 (flange) of
# the Rizon 4S.  These are body_pos / body_quat (wxyz) in link7's local frame.
# Recovered via the interactive Mount-Adjustment slider in mygripper_arm_viewer.
MOUNT_POS = np.array([-0.01000, -0.00010, 0.12100])
MOUNT_QUAT = np.array([0.65397, -0.34044, -0.31195, -0.59926])  # wxyz

# Stiff weld solver params (validated: anchor distance <0.2 mm across full ROM).
_WELD_SOLREF = [0.002, 1.0]
_WELD_SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]

# Weld pairs: b1 ← b2 (b2 fused onto b1 at relpos/relquat in b1's frame).
# Names are the hand-alone URDF names; attach(suffix="_h") retargets them.
_WELD_FINGERS = [
  {
    "name": "F1",
    "b1": "s_link2_step",
    "b2": "s_link2_step_2",
    "relpos": [0.035707, 0.012457, 0.009],
    "relquat": [0.924902, 0.0, 0.0, 0.380206],
  },
  {
    "name": "F2",
    "b1": "finger3_step",
    "b2": "finger3_step_2",
    "relpos": [0.03668, -0.011137, 0.0088],
    "relquat": [0.947288, 0.0, 0.0, -0.320384],
  },
  {
    "name": "F3",
    "b1": "finger4_step",
    "b2": "finger4_step_2",
    "relpos": [0.03668, 0.011137, 0.0089],
    "relquat": [0.972724, 0.0, 0.0, 0.231967],
  },
]

# Assembly angles (rad) that satisfy the weld to <0.2 mm.  Used to seed qpos0
# so the welds start satisfied and avoid a diverging snap on the first step.
_WELD_ASSEMBLY = {
  "Revolute 4": 0.11850651675287728,
  "nRevolute 5": 0.19694636697490253,
  "nCylindrical 1": 0.0774500849888125,
  "nRevolute 6": -0.0009871188288751393,
  "Revolute 7": 0.1831730332809386,
  "nRevolute 8": 0.32215584872455566,
  "nCylindrical 2": 0.3312176032729121,
  "nRevolute 9": 0.19223044142564735,
  "Revolute 10": -0.0019317300325050431,
  "nRevolute 11": -0.0034647938206841134,
  "nCylindrical 3": 0.0021822067379099544,
  "nRevolute 12": -0.0006524893470445766,
}

# Derived per-joint ROM limits (driven joints from four-bar singularity analysis;
# passive joints from swept ROM + 0.1 rad margin).  See JOINT_LIMITS_PROGRESS.md.
_JOINT_LIMITS: dict[str, tuple[float, float]] = {
  "Revolute 1": (-0.70, 0.90),
  "Revolute 2": (-0.70, 0.90),
  "Revolute 3": (-0.70, 0.90),
  "nRevolute 5": (-1.54, 1.88),
  "nRevolute 8": (-1.61, 1.20),
  "nRevolute 11": (-1.54, 1.88),
  "Revolute 4": (-0.85, 1.18),
  "nCylindrical 1": (-1.06, 0.28),
  "nRevolute 6": (-0.73, 0.10),
  "Revolute 7": (-0.83, 0.54),
  "nCylindrical 2": (-0.83, 1.25),
  "nRevolute 9": (-0.30, 0.79),
  "Revolute 10": (-0.76, 0.96),
  "nCylindrical 3": (-0.43, 1.27),
  "nRevolute 12": (-0.74, 0.12),
}

# Weld stability regularisation (the key fix for tiny-inertia loop links).
_WELD_ARMATURE = 0.01  # reflected inertia at every gripper joint
_WELD_DAMPING = 0.3  # viscous damping coefficient

# ---------------------------------------------------------------------------
# Joint-coupling variant (mjEQ_JOINT) — replaces the four-bar weld closure.
# ---------------------------------------------------------------------------
#
# A four-bar linkage is 1-DOF: once the driven "master" curl joint is fixed,
# every passive joint angle is determined.  Instead of cutting the loop and
# holding it shut with a stiff 6-DOF weld (small-dt, ill-conditioned), we delete
# the ghost coupler body to make the chain a tree, then slave each surviving
# passive joint to the master with a scalar mjEQ_JOINT polynomial coupling
#   theta_slave = a0 + a1*theta_master + a2*theta_master^2 + a3*..^3 + a4*..^4
# (qpos0 == 0 so absolute angles map straight onto the polycoef).
#
# Coefficients were measured by sweeping each master across a trimmed working
# range (the singular ROM tips are excluded) and recording where the weld pulls
# the passive joints; quartic fits track to <0.2 deg over that range.  See
# scripts under the repo history / JOINT_LIMITS_PROGRESS for the calibration.
#
# Per finger we slave two joints: the nCylindrical_k (carries the contact pad)
# and the Revolute_k coupler stub (collision-free, kept visually consistent).
# Re-calibrated from a quasi-static ground-truth sweep of the weld linkage
# (verify/weld_sweep.py: both directions from the assembly seed, overdamped, the
# usable range ends where the master stops tracking = the four-bar binds).
# The previous F2 entry was over-trimmed (master upper 0.694) and had a mis-fit
# nCylindrical coefficient, which clamped F2 to ~half the closing travel of its
# mirror twin F3; these swept values restore symmetric closing.  Cubic fits
# (a4=0) track the linkage to ≈2-5 deg over the full range.
_JOINT_COUPLINGS: tuple[dict, ...] = (
  {
    "master": "nRevolute 5",
    "master_range": (-1.549, 1.108),
    "slaves": (
      ("nCylindrical 1", (-0.00014, 0.42840, -0.26745, 0.01583, 0.0)),
      ("Revolute 4", (-0.00358, 0.61856, 0.00205, -0.01514, 0.0)),
    ),
  },
  {
    "master": "nRevolute 8",
    "master_range": (-1.992, 1.074),
    "slaves": (
      ("nCylindrical 2", (-0.00867, 0.95389, 0.35536, 0.02645, 0.0)),
      ("Revolute 7", (0.00484, 0.54672, -0.04352, -0.02976, 0.0)),
    ),
  },
  {
    "master": "nRevolute 11",
    "master_range": (-1.578, 1.891),
    "slaves": (
      ("nCylindrical 3", (-0.00929, -0.62717, 0.31960, -0.00809, 0.0)),
      ("Revolute 10", (-0.01602, 0.54206, 0.00609, -0.00803, 0.0)),
    ),
  },
)

# Stiff scalar coupling: a 1-D joint equality stays well-conditioned even when
# stiff, so we keep it tight (tracks the linkage to <0.05 deg).  A soft setting
# (e.g. timeconst 0.01) leaks a steady-state offset of several degrees.
_COUPLING_SOLREF = [0.002, 1.0]
_COUPLING_SOLIMP = [0.999, 0.9999, 0.001, 0.5, 2.0]


def _convert_welds_to_couplings(spec: mujoco.MjSpec, suffix: str = "") -> None:
  """Replace the 3 four-bar welds with scalar mjEQ_JOINT couplings (in-place).

  Deletes each weld equality and its ghost coupler body (breaking the loops into
  a tree), then adds one scalar joint coupling per surviving passive joint and
  tightens the master curl joints to their calibrated working range.  ``suffix``
  is "_h" after attach() to the arm, "" for the standalone hand.
  """
  for eq in list(spec.equalities):
    if eq.name.startswith("weld_"):
      spec.delete(eq)
  bodies = {b.name: b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)}
  for f in _WELD_FINGERS:
    ghost_name = str(f["b2"]) + suffix
    ghost = bodies.get(ghost_name)
    assert ghost is not None, f"ghost body {ghost_name} not found"
    spec.delete(ghost)  # also removes its joint (nRevolute 6/9/12)
  joints = {j.name: j for j in spec.joints}
  for fc in _JOINT_COUPLINGS:
    master = fc["master"] + suffix
    mj = joints.get(master)
    assert mj is not None, f"master joint {master} not found"
    mj.range = list(fc["master_range"])
    mj.limited = True
    for slave, poly in fc["slaves"]:
      eq = spec.add_equality()
      eq.name = f"couple_{slave}{suffix}".replace(" ", "")
      eq.type = mujoco.mjtEq.mjEQ_JOINT
      eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
      eq.name1 = slave + suffix  # dependent joint (y)
      eq.name2 = master  # reference joint (x)
      data = np.zeros(11)
      data[0:5] = poly
      eq.data = data
      eq.solref = np.array(_COUPLING_SOLREF)
      eq.solimp = np.array(_COUPLING_SOLIMP)


# ---------------------------------------------------------------------------
# Spec builders — return MjSpec (not compiled).
# ---------------------------------------------------------------------------


# b1 coupler bodies (weld primaries) — kept collision-free now that the
# rubber pad output bodies below are the true contact surfaces.
_FINGERTIP_BODIES = {
  f["b1"] for f in _WELD_FINGERS
}  # {"s_link2_step", "finger3_step", "finger4_step"}
# b2 bodies: always kept collision-free (they're the weld "ghost" duplicates).
_WELD_GHOST_BODIES = {f["b2"] for f in _WELD_FINGERS}
# Rubber pad output bodies: the distal link of each four-bar finger chain.
# These are the actual contact surfaces that touch the grasped object.
_FINGERTIP_PAD_BODIES = {"finger1_step", "finger1_step_2", "finger1_step_3"}


def _configure_collision(hand_spec: mujoco.MjSpec) -> None:
  """Enable contact only on rubber pad output bodies; silence everything else.

  The four-bar mechanism has overlapping coupler geoms (b1/b2 pairs) that
  must not collide with each other.  The actual contact surfaces are the
  distal output-link bodies (finger1_step, finger1_step_2, finger1_step_3)
  — these carry the rubber fingertip pads and form the grasping triangle.
  """
  for body in hand_spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    is_pad = body.name in _FINGERTIP_PAD_BODIES
    for geom in body.geoms:
      if not is_pad:
        geom.contype = 0  # disable contact initiation; keep conaffinity for rendering


def _add_weld_equalities(hand_spec: mujoco.MjSpec) -> None:
  """Add 3 mjEQ_WELD constraints to hand_spec (in-place, before attach)."""
  for f in _WELD_FINGERS:
    eq = hand_spec.add_equality()
    eq.name = f"weld_{f['name']}"
    eq.name1 = f["b1"]
    eq.name2 = f["b2"]
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.type = mujoco.mjtEq.mjEQ_WELD
    data = np.zeros(11)
    data[3:6] = f["relpos"]
    data[6:10] = f["relquat"]
    data[10] = 1.0
    eq.data = data
    eq.solref = np.array(_WELD_SOLREF)
    eq.solimp = np.array(_WELD_SOLIMP)


def _set_joint_limits(hand_spec: mujoco.MjSpec) -> None:
  """Apply derived ROM limits to joints in hand_spec (in-place, before attach)."""
  for j in hand_spec.joints:
    if j.name in _JOINT_LIMITS:
      lo, hi = _JOINT_LIMITS[j.name]
      j.range = [lo, hi]
      j.limited = True


def get_hand_only_spec(coupling: bool = False) -> mujoco.MjSpec:
  """Standalone mygripper_H100_R spec: 15 joints, 3 weld equalities, no arm.

  mjlab attaches this to the scene's world body; the root body (base_step)
  is fixed at the entity origin.  Actuators are added separately via
  EntityArticulationInfoCfg — do NOT add them here.

  A grasp_site is added to base_step at an approximate grasping centre that
  can be tuned after inspecting the assembled scene.

  If ``coupling`` is True, the three four-bar welds are replaced with scalar
  mjEQ_JOINT couplings (loops → tree); this drops 3 DOF and is far cheaper /
  more stable than the stiff weld closure.
  """
  hand = mujoco.MjSpec.from_file(str(HAND_URDF))

  # Structural edits (all before any attach call):
  _add_weld_equalities(hand)
  _set_joint_limits(hand)

  # Selectively disable collision to avoid self-collision while keeping fingertip contact.
  #
  # The four-bar mechanism has duplicate coupler bodies (b1 and b2) that physically
  # overlap.  Strategy:
  #   - Fingertip bodies (b1 of each weld pair): keep contype=1 → can contact cube.
  #   - All other bodies (including b2 ghost copies): set contype=0 → no contact.
  #
  # Only contype is cleared (not conaffinity) so MuJoCo retains the geoms for
  # rendering.  conaffinity=1 means these geoms *accept* contact from others, but
  # since their contype=0 they can't initiate any — the net result is no contact.
  _configure_collision(hand)

  # Grasp site: approximate centre of the three jaw tips in assembly pose.
  root_body = next(
    b
    for b in hand.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
    if b.name == "base_step"
  )
  gs = root_body.add_site()
  gs.name = "grasp_site"
  gs.pos = [0.07, 0.0, 0.03]
  gs.size = [0.005, 0.0, 0.0]

  if coupling:
    _convert_welds_to_couplings(hand, suffix="")

  return hand


def get_arm_hand_spec(coupling: bool = False) -> mujoco.MjSpec:
  """Flexiv Rizon 4S + mygripper_H100_R spec: 7 arm + 15 gripper joints.

  The arm's XML already contains position actuators for joint1–joint7.
  Gripper actuators are added separately via EntityArticulationInfoCfg.

  If ``coupling`` is True, the three four-bar welds are replaced with scalar
  mjEQ_JOINT couplings after attach() (loops → tree, 3 DOF dropped).

  Requires FLEXIV_XML to exist (set the path at the top of this file).
  """
  assert FLEXIV_XML.exists(), (
    f"Flexiv XML not found: {FLEXIV_XML}.\n"
    "Either symlink the Rizon 4S XML there or use get_elephant_hand_cfg() "
    "(standalone hand, no arm)."
  )
  arm = mujoco.MjSpec.from_file(str(FLEXIV_XML))

  # Mount site on link7 — set directly to the tuned mount pose so that after
  # attach() the hand root body is already at the correct position.  This
  # replaces the post-compile _bake_mount() from mygripper_arm_viewer.py.
  link7 = next(
    (b for b in arm.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "link7"),
    None,
  )
  assert link7 is not None, "link7 not found in Flexiv spec"
  site = link7.add_site()
  site.name = "hand_mount"
  site.pos = list(MOUNT_POS)
  site.quat = list(MOUNT_QUAT)

  hand = mujoco.MjSpec.from_file(str(HAND_URDF))
  _add_weld_equalities(hand)  # before attach so suffix "_h" is applied automatically
  _set_joint_limits(hand)
  _configure_collision(
    hand
  )  # fingertip b1 bodies keep contact; everything else disabled

  # Grasp site on the hand root body (will be renamed base_step_h after attach).
  root_body = next(
    b
    for b in hand.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
    if b.name == "base_step"
  )
  gs = root_body.add_site()
  gs.name = "grasp_site"
  gs.pos = [0.117, -0.058, 0.079]
  gs.size = [0.02, 0.0, 0.0]
  gs.rgba = [1.0, 0.0, 0.0, 0.8]

  arm.attach(hand, suffix="_h", site="hand_mount")

  # MuJoCo's attach() namespaces every element from the child spec with a
  # leading "/" (e.g. "base_step_h" → "/base_step_h").  The entity layer
  # strips this via split("/")[-1] for its own lookups, but the contact
  # sensor's _prefix_name() does entity + "/" + stripped_name, producing
  # "robot/base_step_h" while the actual compiled MuJoCo name would be
  # "robot//base_step_h".  Strip the leading "/" here so all names are
  # canonical and the sensor resolves correctly.
  for body in arm.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    if body.name.startswith("/"):
      body.name = body.name[1:]
  for j in arm.joints:
    if j.name.startswith("/"):
      j.name = j.name[1:]
  for s in arm.sites:
    if s.name.startswith("/"):
      s.name = s.name[1:]
  for eq in arm.equalities:
    if eq.name.startswith("/"):
      eq.name = eq.name[1:]
    if eq.name1.startswith("/"):
      eq.name1 = eq.name1[1:]
    if eq.name2.startswith("/"):
      eq.name2 = eq.name2[1:]

  # Delete the arm's default keyframe — stale now that the model has 22 DOF.
  for k in list(arm.keys):
    arm.delete(k)

  if coupling:
    _convert_welds_to_couplings(arm, suffix="_h")

  return arm


# ---------------------------------------------------------------------------
# Actuator configs.
# ---------------------------------------------------------------------------

# Arm joints: reuse XML actuators already in the Rizon 4S MJCF.
_ARM_ACTUATOR = XmlActuatorCfg(
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

# Gripper: 6 driven DOFs (3 base-swing + 3 curl-input), one per finger.
# After attach(suffix="_h") the joint names gain the "_h" suffix.
# KP=8, KD=0.4 from mygripper_arm_viewer.py; armature/viscous_damping are
# the weld stability regularisers (the real fix for tiny-inertia loop links).
_GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "Revolute [123]_h",  # 3 base-swing joints (aperture)
    "nRevolute (5|8|11)_h",  # 3 curl-input joints (wrap / conform)
  ),
  stiffness=8.0,
  damping=0.4,
  armature=_WELD_ARMATURE,
  viscous_damping=_WELD_DAMPING,
)

# Same actuator config for the standalone hand (joint names WITHOUT "_h").
_GRIPPER_ACTUATOR_STANDALONE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "Revolute [123]",
    "nRevolute (5|8|11)",
  ),
  stiffness=8.0,
  damping=0.4,
  armature=_WELD_ARMATURE,
  viscous_damping=_WELD_DAMPING,
)

# ---------------------------------------------------------------------------
# Initial state configs.
# ---------------------------------------------------------------------------

# Arm joints: hover-forward pre-grasp (same as mygripper_arm_viewer.py qpos[:7]).
# Hand joints: assembly pose (the only pose where the welds are satisfied at t=0;
# deviating from this before a forward pass causes a ~39 mm snap → NaN risk).
# Keys are entity-internal joint names (without entity prefix).
_ARM_HAND_JOINT_POS: dict[str, float] = {
  "joint4": 1.57,  # elbow up; other arm joints default to 0.0 via catch-all
  # Assembly angles for the 15 hand joints (with "_h" attach suffix).
  # Driven joints start at their assembly values so ctrl==qpos at t=0.
  "nRevolute 5_h": _WELD_ASSEMBLY["nRevolute 5"],
  "nRevolute 8_h": _WELD_ASSEMBLY["nRevolute 8"],
  "nRevolute 11_h": _WELD_ASSEMBLY["nRevolute 11"],
  # Passive joints — must be seeded at assembly or the welds snap.
  "Revolute 4_h": _WELD_ASSEMBLY["Revolute 4"],
  "nCylindrical 1_h": _WELD_ASSEMBLY["nCylindrical 1"],
  "nRevolute 6_h": _WELD_ASSEMBLY["nRevolute 6"],
  "Revolute 7_h": _WELD_ASSEMBLY["Revolute 7"],
  "nCylindrical 2_h": _WELD_ASSEMBLY["nCylindrical 2"],
  "nRevolute 9_h": _WELD_ASSEMBLY["nRevolute 9"],
  "Revolute 10_h": _WELD_ASSEMBLY["Revolute 10"],
  "nCylindrical 3_h": _WELD_ASSEMBLY["nCylindrical 3"],
  "nRevolute 12_h": _WELD_ASSEMBLY["nRevolute 12"],
  # Catch-all: all other joints (arm joints 1-3,5-7, base swing joints) → 0.
  ".*": 0.0,
}

_HAND_ONLY_JOINT_POS: dict[str, float] = {
  "nRevolute 5": _WELD_ASSEMBLY["nRevolute 5"],
  "nRevolute 8": _WELD_ASSEMBLY["nRevolute 8"],
  "nRevolute 11": _WELD_ASSEMBLY["nRevolute 11"],
  "Revolute 4": _WELD_ASSEMBLY["Revolute 4"],
  "nCylindrical 1": _WELD_ASSEMBLY["nCylindrical 1"],
  "nRevolute 6": _WELD_ASSEMBLY["nRevolute 6"],
  "Revolute 7": _WELD_ASSEMBLY["Revolute 7"],
  "nCylindrical 2": _WELD_ASSEMBLY["nCylindrical 2"],
  "nRevolute 9": _WELD_ASSEMBLY["nRevolute 9"],
  "Revolute 10": _WELD_ASSEMBLY["Revolute 10"],
  "nCylindrical 3": _WELD_ASSEMBLY["nCylindrical 3"],
  "nRevolute 12": _WELD_ASSEMBLY["nRevolute 12"],
  ".*": 0.0,
}

# Coupling variant: the ghost-body joints (nRevolute 6/9/12) are deleted, so
# drop them from the init pose.  Masters + surviving slaves start at assembly so
# the couplings begin (near-)satisfied.
_ARM_HAND_JOINT_POS_COUPLED: dict[str, float] = {
  k: v
  for k, v in _ARM_HAND_JOINT_POS.items()
  if k not in {"nRevolute 6_h", "nRevolute 9_h", "nRevolute 12_h"}
}
_HAND_ONLY_JOINT_POS_COUPLED: dict[str, float] = {
  k: v
  for k, v in _HAND_ONLY_JOINT_POS.items()
  if k not in {"nRevolute 6", "nRevolute 9", "nRevolute 12"}
}

# ---------------------------------------------------------------------------
# Public EntityCfg factories.
# ---------------------------------------------------------------------------


def get_elephant_hand_cfg(coupling: bool = False) -> EntityCfg:
  """Standalone elephant hand (no arm).  Fixed at world origin."""
  return EntityCfg(
    spec_fn=functools.partial(get_hand_only_spec, coupling=coupling),
    init_state=EntityCfg.InitialStateCfg(
      joint_pos=_HAND_ONLY_JOINT_POS_COUPLED if coupling else _HAND_ONLY_JOINT_POS,
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(_GRIPPER_ACTUATOR_STANDALONE,),
      soft_joint_pos_limit_factor=0.9,
    ),
  )


def get_elephant_arm_cfg(coupling: bool = False) -> EntityCfg:
  """Flexiv Rizon 4S arm + elephant hand.

  ``coupling=False`` → 22 DOF, three stiff four-bar welds (legacy).
  ``coupling=True``  → 19 DOF, welds replaced by scalar mjEQ_JOINT couplings.
  """
  return EntityCfg(
    spec_fn=functools.partial(get_arm_hand_spec, coupling=coupling),
    init_state=EntityCfg.InitialStateCfg(
      joint_pos=_ARM_HAND_JOINT_POS_COUPLED if coupling else _ARM_HAND_JOINT_POS,
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(_ARM_ACTUATOR, _GRIPPER_ACTUATOR),
      soft_joint_pos_limit_factor=0.9,
    ),
  )


# ---------------------------------------------------------------------------
# Action scale (used in JointPositionActionCfg.scale).
# ---------------------------------------------------------------------------

# Arm: same as FLEXIV_ACTION_SCALE (0.25 * effort_limit / stiffness per joint).
# Gripper: half the joint operating range → action ±1 ≈ half the full ROM.
ELEPHANT_ACTION_SCALE: dict[str, float] = {
  "joint1": 0.25 * 123 / 289,
  "joint2": 0.25 * 123 / 673,
  "joint3": 0.25 * 64 / 224,
  "joint4": 0.25 * 64 / 373,
  "joint5": 0.25 * 39 / 237,
  "joint6": 0.25 * 39 / 232,
  "joint7": 0.25 * 39 / 186,
  # Base swing: range (-0.70, 0.90) → half-range ≈ 0.80; use 0.40 for safety.
  "Revolute [123]_h": 0.40,
  # Curl: range ≈ (-1.54, 1.88) / (-1.61, 1.20) → use 0.80 for safety.
  "nRevolute (5|8|11)_h": 0.80,
}
