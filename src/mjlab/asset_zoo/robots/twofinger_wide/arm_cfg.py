"""EntityCfg for the Flexiv Rizon 4S + the custom two-finger claw.

Assembles one mjlab entity: sys-id'd arm, the gripper read from CAD, the base
plate it stands on, and the static camera mount. Everything the *task* adds --
table, foam, cube, rewards, observations -- lives in
``tasks/cube_grasp/config/twofinger/env_cfgs.py``.

ARM (system-identified)
    The Menagerie ``flexiv_rizon4s`` model with per-joint passive parameters
    OVERRIDDEN by the sys-id values from ``robot-actuator-model``
    (``rizon/rizon_hand.xml``): damping, frictionloss, armature and
    actuatorfrcrange exactly as identified. Armature is 0 on joints 1-4 -- the
    fit only found reflected inertia on the wrist joints; that is a result, not
    an omission. Menagerie's position servos (kp 186..673) are kept.

HAND (read from CAD, not transcribed)
    ``hand_urdf.py`` parses ``hands/<CG_HAND>/hand.urdf`` at build time and this
    file assembles the spec from it: link masses, centres of mass and full
    inertia tensors (off-diagonals included), joint origins, axes and travel.

    Nothing about the gripper's geometry or inertia is a constant in this file.
    That is deliberate and it is the point of the arrangement -- the sim and the
    CAD are the same object, so a hardware revision is a file copy and a re-run.
    The former 56 mm claw did carry estimates here, and they were wrong by 2.1x
    in total mass. See ``src/mjlab/asset_zoo/robots/twofinger_wide/assets/hands/README.md`` and ``docs/hand.md``.

    Two things are still this file's own, because they are not CAD properties:
    the actuator gains (``_FINGER_KP``/``_FINGER_KV``, with the URDF's effort
    limit applied as the force range) and the collision boxes, which are FITTED
    to the CAD meshes rather than inherited -- the URDF ships a 151-hull COACD
    decomposition that is the wrong representation for 1024 parallel envs.

GRAFT
    ``hand_base`` attaches to a site on link7. The pose is the calibrated one
    from the repo XML plus a DERIVED offset; see the block above
    ``_FLANGE_REF_POS`` for the derivation and the one assumption it rests on.
    This is the only pose in the file that is not directly measured.

BASE PLATE
    The Onshape base plate (225 x 225 x 15 mm) is bolted under the arm base, so
    the arm origin sits at the plate's top centre. The plate rests ON the
    manipulation table, which is why the whole arm is lifted to ``ARM_BASE_Z``
    and the plate ends up flush with the surface the cube sits on.

COLLISION (two-bit scheme; the CAD model carries none)
    Palm and finger links get box geoms with ``contype=FINGER_BIT(2)`` and
    ``conaffinity=OBJECT_BIT(4)|FINGER_BIT(2)``, so fingers hit the object and
    each other but never the arm or the floor. A graspable object uses
    ``contype=OBJECT_BIT, conaffinity=3``; a table uses ``contype=5,
    conaffinity=3`` so the fingers collide with it too.

SENSORS
    A touch site plus ``mjSENS_TOUCH`` on each distal link, so grasp rewards key
    on REAL contact force rather than proximity. Massless ``left_pad`` /
    ``right_pad`` marker bodies sit at the measured contact patch and are what
    the reward's distance gating reads.

    NOTE when reading these: ``data.sensordata`` is indexed by
    ``model.sensor_adr``, NOT by sensor id. The 3-vector force/torque sensors
    precede the touch sensors here, so the two indices differ.

Public surface
    ``get_robot_cfg``, ``twofinger_arm_spec``, ``PAD_BODIES``,
    ``TOUCH_SENSORS``, ``FINGER_JOINTS``, ``ARM_JOINTS``, ``FINGER_OPEN_POS``,
    ``FINGER_OPEN_ANGLE``, ``aperture_mm``, ``open_angle_for``,
    ``TWOFINGER_ARM_HOME`` / ``_SETTLED`` / ``_PREGRASP``, ``CG_HAND``.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import XmlActuatorCfg
from mjlab.asset_zoo.robots.twofinger_wide.hand_urdf import (
  ASSETS_DIR,
  COLLISION_DIR,
  HANDS_DIR,
  fit_collision_boxes,
  load_hand_cad,
  pad_marker_pos,
  tip_marker_pos,
)
from mjlab.asset_zoo.robots.twofinger_wide.hand_urdf import (
  MESH_DIR as WIDE_MESH_DIR,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# --------------------------------------------------------------------------- #
# mujoco_menagerie is an external checkout (github.com/google-deepmind/mujoco_menagerie),
# not vendored here. Point MUJOCO_MENAGERIE at your clone; the default is only a
# convenience for the machine this was developed on.
_MENAGERIE = Path(
  os.environ.get("MUJOCO_MENAGERIE") or Path.home() / "mujoco_menagerie"
)
_ARM_XML = _MENAGERIE / "flexiv_rizon4s" / "flexiv_rizon4s.xml"
# The base plate and the static camera mount belong to no hand and sit at the
# top of the package's asset tree; the gripper's own meshes come from
# assets/hands/<CG_HAND>/meshes via hand_urdf.MESH_DIR. Both are PACKAGE DATA,
# resolved through importlib.resources so they survive an install.
_ASSET_DIR = ASSETS_DIR  # base plate, camera mount

_SUFFIX = "_tf"

# --- which hand is on the robot -------------------------------------------- #
# One directory per gripper under hands/. Only the wide (70 mm knuckle) claw
# ships: it is what is fitted. The 56 mm claw it replaced was removed on
# 2026-08-04 -- see src/mjlab/asset_zoo/robots/twofinger_wide/assets/hands/README.md for what re-introducing a second hand takes.
#
# Validated against the directory rather than a hardcoded list, so adding a
# gripper is a directory plus its constants and nothing to update here.
CG_HAND = (os.environ.get("CG_HAND") or "wide").lower()
if not (HANDS_DIR / CG_HAND / "hand.urdf").exists():
  _avail = sorted(d.name for d in HANDS_DIR.iterdir() if (d / "hand.urdf").exists())
  raise ValueError(
    f"CG_HAND={CG_HAND!r} has no assets/hands/{CG_HAND}/hand.urdf; available: {_avail}"
  )

# --- where the hand bolts to the flange ------------------------------------ #
# TOOL_MOUNT_SITE is the site on link7 that the hand spec is attached to.
#
# _MOUNT_QUAT and _FLANGE_REF_POS are the CALIBRATED pose from
# robot-actuator-model's rizon_hand.xml -- but they describe where the *former*
# 56 mm claw's base_link sat, because that is the assembly that was calibrated.
#
# The fitted wide claw's base_link sits elsewhere in its own frame, and the CAD
# repo publishes no flange-to-hand transform for it (rizon_baseplate_camera.urdf
# stops at the flange; its wide-hand MJCF is a standalone hand). So the offset
# below is DERIVED, and it is the one number in this file that is not measured.
#
# It rests on the ADAPTER being the same part in both assemblies -- same mesh,
# same mass to 9 significant figures, same rpy (pi, 0, 0) -- so it bolts to the
# flange identically and T_link7->adapter is invariant:
#
#     T_link7->base_wide = M . N . inv(W)
#         M = the calibrated pose below
#         N = T(0, 0.0113, 0.0315) . Rx(pi)    former claw's adapter_fixed
#         W = T(0, 0,      0.0240) . Rx(pi)    wide   claw's adapter_fixed
#
# The rotations cancel exactly, leaving a pure translation in the hand frame and
# an UNCHANGED orientation. VERIFIED: with either hand's spec attached, the
# adapter and shim land at identical world positions, 0.0 mm delta.
#
# If the physical hand turns out to be clocked differently on the flange, THIS
# is the constant to fix, and it is the first thing to check against the real
# arm. See docs/hand.md.
TOOL_MOUNT_SITE = "tool_mount"
_FLANGE_REF_POS = (-0.00799, 0.00797, 0.15550)
_MOUNT_QUAT = (0.0, -0.923880, -0.382683, 0.0)
_MOUNT_OFFSET_HAND = (0.0, 0.0113, 0.0075)


def _mount_pos() -> tuple[float, float, float]:
  """Hand-base position on link7: the calibrated pose plus the derived offset."""
  r = mujoco_quat_to_mat(_MOUNT_QUAT)
  return tuple(np.array(_FLANGE_REF_POS) + r @ np.array(_MOUNT_OFFSET_HAND))


def mujoco_quat_to_mat(q) -> np.ndarray:
  """wxyz quaternion -> 3x3 rotation matrix (mujoco's convention)."""
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.array(q, dtype=float))
  return m.reshape(3, 3)


# --- collision two-bit scheme ---------------------------------------------- #
FINGER_BIT = 2
OBJECT_BIT = 4

# --- base plate (robot-actuator-model rizon/meshes/baseplate_flat.stl) ------ #
# The exported plate is laid flat here (see src/mjlab/asset_zoo/robots/twofinger_wide/assets/baseplate.stl): thin
# axis -> z, centred in x/y, TOP FACE AT z = 0, i.e. expressed directly in the
# arm-base frame, so the visual mesh mounts at the origin with no offset.
BASE_PLATE_MESH = "baseplate"
BASE_PLATE_THICKNESS = 0.015  # 15 mm plate
BASE_PLATE_HALF = (0.1125, 0.1125, BASE_PLATE_THICKNESS / 2)  # 225 mm square
BASE_PLATE_MASS = 1.47  # 545 cm^3 (holes included) of 6061 alloy

# Height of the surface the plate is bolted to. The plate sits on the SAME
# table as the cube, so this is the table top; the arm base origin is one plate
# thickness above it. env_cfgs.py derives TABLE_TOP_Z from this constant.
MOUNT_SURFACE_Z = 0.35
ARM_BASE_Z = MOUNT_SURFACE_Z + BASE_PLATE_THICKNESS  # 0.365

# --- static camera mount ("Static Camera Mount V1.4.3") --------------------- #
# Weld pose is AUTHORITATIVE, from the repo's combined URDF
# (urdf/rizon_baseplate_camera.urdf, joint base_plate_to_camera_mount):
#     xyz = (-0.092, 0, 0)   rpy = (0, 0, -pi/2)
# Do not re-derive it. An earlier version of this file inferred +0.090 / +pi/2
# from the foot's concave arc going tangent to the arm base -- right magnitude,
# WRONG SIDE. The tangency condition is symmetric in x (the arc nests equally
# well in front or behind), and that ambiguity was resolved by assuming the
# camera must face the workspace rather than by evidence. The URDF settles it.
CAMERA_MOUNT_MESH = "camera_mount"
# Alok's call (2026-07-24): mirror BOTH of the URDF's values -> (+0.092, 0, 0)
# / (0, 0, +pi/2), so the mount sits on the +x side and the shelf faces the
# workspace. Yaw is what aims the camera: the shelf normal's horizontal
# component maps to -x at yaw -pi/2 and +x at +pi/2, for ANY x -- sliding x only
# moves the origin along the rail. At the URDF's own -92/-90 it looks the other
# way entirely (cube ~119 deg off-axis), so this DEPARTS from the URDF
# deliberately. With V1.4.3's 27.45 deg shelf the axis meets the foam top at
# x = +0.433 and the cube-centre plane at x = +0.385; TABLE_X sits a deliberate
# 75 mm beyond that at 0.46, so the cube's NEAR CORNER clears the D435's minimum
# range everywhere in the spawn box (see env_cfgs.py) -- 5.7 deg off-axis, against
# a 29 deg vertical half-FOV.
# For the record, V1.4's 45 deg shelf aimed at x = +0.240, so with foam the cube
# at TABLE_X = 0.41 sat at 74% of frame height and the +100 mm lift target was
# entirely OUT OF FRAME (v = +1.54). The shallower mount is what makes a lifted
# cube visible at all; it is not a cosmetic change.
CAMERA_MOUNT_OFFSET = 0.092  # +x offset in the arm-base frame
CAMERA_MOUNT_YAW = np.pi / 2
# Collision proxies (mount-local, pre-yaw), z-slab AABBs of the V1.4.3 mesh: it
# is 48.7 k faces, far too expensive for warp. Boxes as (halfsize, pos).
_CAM_BOXES = (
  ("cam_foot_col", (0.07792, 0.03201, 0.02250), (0.0, 0.00451, 0.00750)),
  ("cam_neck_col", (0.04036, 0.01979, 0.01500), (0.0, -0.00771, 0.04500)),
  ("cam_post_col", (0.02923, 0.00500, 0.05000), (0.0, 0.01305, 0.11000)),
  ("cam_head_col", (0.03500, 0.01978, 0.02921), (0.0, -0.00173, 0.18921)),
)
# Static Camera Mount V1.4.3 (robot-actuator-model @ dbfac5d): the apex bracket
# faces moved from 45 deg to 27.45/62.55 deg and the apex rose 214.64 -> 218.43
# mm. The camera bolt face is the 1064 mm^2, 55 x 21 mm wall whose outward
# normal is 27.449992 deg below horizontal. A camera bolted flat to it looks
# along that normal, so the optical axis is baked into the part, not chosen here.
#
# Both numbers are the URDF's (joint camera_mount_to_camera, commit ac2c373) and
# were CONFIRMED against the mesh independently: 252 coplanar faces, area
# 1064.0 mm^2 to four figures, area-weighted centroid identical to 3 decimals of
# a mm, max deviation from the URDF plane 0.5 um. Do not re-derive them by
# eyeballing the STL -- the part still carries 13.8 mm^2 of leftover 45 deg
# facets and 6.5 mm^2 at the complementary 62.55 deg, so a normal-only search
# finds decoys.


def camera_mount_xy_aabb(z_lo: float, z_hi: float) -> tuple[float, float, float, float]:
  """World-frame (x0, x1, y0, y1) of mount structure between world z_lo..z_hi.

  Exported because anything laid ON the bench has to be cut around the mount,
  not through it. The 50 mm foam layer found this the hard way: the mount foot
  reaches world x = 119.5 mm while the base-plate cutout stops at 112.5, so
  7 mm of the foot -- 1382 mesh vertices -- sat buried inside the foam slab.
  Nothing exploded, because foam, plate, mount and arm base are all welded to
  the world and MuJoCo filters every one of those pairs, so the overlap was
  silent. It was still wrong: on the real bench the foam gets knife-cut around
  the foot.

  Derived from the collision proxies rather than the mesh so callers do not
  need trimesh, and from the yaw/offset constants rather than hard-coded
  numbers so it follows the mount if it is ever repositioned.
  """
  xs: list[float] = []
  ys: list[float] = []
  for _, half, pos in _CAM_BOXES:
    zc = ARM_BASE_Z + pos[2]
    if zc + half[2] <= z_lo or zc - half[2] >= z_hi:
      continue
    # yaw +pi/2 maps mount-local (x, y) -> world (-y, x), so the half-extents
    # swap axes along with the centre.
    cx, cy = CAMERA_MOUNT_OFFSET - pos[1], pos[0]
    xs += [cx - half[1], cx + half[1]]
    ys += [cy - half[0], cy + half[0]]
  if not xs:
    raise ValueError(f"no camera-mount structure between z {z_lo} and {z_hi}")
  return min(xs), max(xs), min(ys), max(ys)


_CAM_FACE_NORMAL = (0.0, -0.8874135122, -0.4609742492)
_CAM_FACE_CENTRE = (0.0, -0.013912, 0.204879)  # centroid of the bolt face

# How far below horizontal the optical axis comes out, DERIVED from the bolt
# face rather than typed. The mount's own geometry sets the camera angle, so
# this follows a CAD revision instead of needing to be remembered: it went
# 45 -> 27.45 deg with the V1.4.3 mount, and the self-test that asserted 45
# kept asserting 45 for three generations afterwards.
CAMERA_MOUNT_TILT_DEG = float(
  np.degrees(np.arcsin(-_CAM_FACE_NORMAL[2] / np.linalg.norm(_CAM_FACE_NORMAL)))
)

# --- measured correction on top of the CAD chain ----------------------------- #
# Everything above is bolt-face geometry: where the shelf is SUPPOSED to point.
# The print tolerance, the screw-hole clearance and how hard the bolts were done
# up all sit between that and where the lens actually looks.
#
# Measured 2026-08-04 on Rizon4s-063501 by fitting the table plane twice -- once
# in the camera frame from the D435 point cloud, once in the base frame by touch-
# probing 9 points with the bare flange -- and solving for the transform between
# them. Protocol and scripts: hierarchical_vla/sim2real/calib (s2, s3, s4).
#
#   optical axis   27.450 deg (CAD)  ->  28.494 deg below horizontal
#   correction     pitch -1.044, pan +0.090, roll -0.181 deg, in camera axes
#   position       +2.13 mm, almost entirely along the table normal
#
# Corroborated by two numbers that do NOT feed this correction, each from a
# different instrument:
#   * the arm base sits 14.2 mm above the table (arm encoders, touch probe)
#     against BASE_PLATE_THICKNESS = 15.0 mm above -- which came from Onshape,
#     not from any fit
#   * the lens sits 210.1 mm above the table (point cloud) against 208.9 mm out
#     of this CAD chain -- and that one never touches the arm
#
# A PLANE ONLY PINS 3 OF THE 6 DoF: the two rotations that hold pitch, plus the
# height along its own normal. Yaw about the table normal and the two in-plane
# translations are invisible to it -- a plane cannot see rotation about itself or
# translation within itself -- so they sat at CAD until stage B below.
#
# Applied to the camera only, not to the D435i body: tilting the whole assembly
# by 1.044 deg would move the lens 25.1 mm * sin(1.044 deg) = 0.46 mm, well inside
# the measurement noise, so rotating the mesh and its collision boxes would be
# churn without fidelity.
CALIB_CAM_DROT = np.array(
  [  # MuJoCo camera axes: +x right,
    [0.999993765, -0.003146045, -0.001603944],  # +y up, -z forward. Right-
    [0.003174744, 0.999829036, 0.018215942],  # multiplies the CAD rotation.
    [0.001546362, -0.018220920, 0.999832789],
  ]
)

# --- stage B: the in-plane translation the plane fit could not see ------------ #
# Measured 2026-08-07 on Rizon4s-063501 with deploy/capture_sweep.py (nine poses,
# spanning 31.6 deg of joint1 and 0.37-0.48 m of range) and scripts/fit_camera_
# pose.py, which registers the real claw's point cloud onto the rendered one at
# each pose and asks which single camera error explains all nine at once.
#
#   camera sits 19.94 mm to the image-LEFT of where the CAD chain puts it
#   -> base frame (-0.99, +19.94, -1.58) mm, i.e. one lateral offset
#
# It is a TRANSLATION and not a rotation, which the fit reports rather than
# assumes: position alone leaves a 1.79 mm residual over 27 equations, rotation
# alone leaves 3.44 mm, both together 1.64 mm, and the uncorrected extrinsic
# 11.69 mm. Six parameters buy 0.15 mm over three, so three is what goes in --
# especially as the rotational DoF are the ones degenerate with joint1 (below).
# Condition number 17, so the poses did separate what they claim to.
#
# The size is what _D435I_IR_LENS already warns about: the depth origin is the
# LEFT IR imager and the two imagers are 50 mm apart, so a nominal lens position
# is wrong by exactly this order. It is also 6.6x _CAM_POS_JITTER (+-3 mm), the
# camera-position domain randomisation the student trained under -- SO THE
# CURRENT CHECKPOINT WAS TRAINED AGAINST THE OLD, WRONG CAMERA and does not
# inherit this fix. It applies from the next retrain; until then the correction
# makes the sim match the bench, which is what any diff against real is worth.
#
# A JOINT1 ZERO-OFFSET WOULD LOOK IDENTICAL TO THE SWEEP, and the sweep alone
# cannot separate them: substituting p_base = R^T p_cam + cam_pos into
# dj*(z x p_base) splits it exactly into a camera rotation plus a camera
# translation, so it lies in the span of the camera's own six numbers at every
# pose. The design matrix is singular by construction, not for want of data --
# everything joint1 moves is one rigid body, and a rigid body cannot say what it
# is rigid WITH RESPECT TO.
#
# Settled the same day by the one feature that does NOT turn with joint1: a cube
# at a measured place on the table (deploy/check_camera.py). At (0.508, 0.102) on
# the bare bench the two hypotheses predict pixels 5.1 px apart, and the cube
# landed 1.79 px from the camera prediction against 4.30 px from the joint1 one.
# So the camera really is the cause, and this correction is right for the table
# and the cube as well as for the arm. The same frame put a left-right mirror
# 44.5 px away and a 180 deg rotation 45.3 px, both comprehensively excluded.
#
# Corroborating, from the same sweep: a joint1-only fit needs dj = +1.99 deg and
# still leaves 3.13 mm against camera-position's 1.79 mm, and when both are free
# joint1 collapses to +0.11 deg. Two degrees is also absurd for a zero offset on
# encoders good to 3e-5 rad.
CALIB_CAM_OFFSET = (0.019931531, 0.000989788, 0.000549401)  # camera_mount frame, m

# --- RealSense D435i body (mujoco_menagerie) --------------------------------- #
# Menagerie's model is geometry only -- 9 textured visual meshes, a fitted
# capsule collision, mass 0.072 kg, and NO <camera> element, so the optics are
# ours to define. Its body frame (verified by transforming the mesh verts):
# extents 89.9 x 25.0 x 25.1 mm (= the 90 x 25 x 25 datasheet body), lens row
# along x, thin axis y, body spanning z in [-25.1, 0] with the lens plane at
# z = 0 -- i.e. the camera looks along +z.
_D435I_XML = _MENAGERIE / "realsense_d435i" / "d435i.xml"
_D435I_DEPTH = 25.1e-3  # back face to lens plane
# IR (depth) lens centroid in the D435i body frame. The left imager is the
# depth origin on real hardware; this is close enough for a nominal model and
# should be replaced by the extrinsics from calibration.
_D435I_IR_LENS = (-0.0057, 0.0, -0.00123)

# --- camera intrinsics: NOMINAL GUESS, replace with a real calibration ------- #
# NOT fovy. With fovy alone MuJoCo leaves cam_intrinsic at a placeholder
# (0.01, 0.01) and cam_resolution at 1x1, so there are no true intrinsics to
# match against hardware. The calibrated path is focalpixel + principalpixel +
# resolution + sensorsize, where (verified empirically):
#   * sensorsize is MANDATORY alongside focalpixel (compile error without it);
#   * focal_metric = focalpixel * sensorsize / resolution, exactly;
#   * principalpixel is an OFFSET FROM THE IMAGE CENTRE, not absolute cx/cy, AND
#     IT IS NEGATED relative to image axes: use (-(cx - W/2), -(cy - H/2)).
#     This note previously omitted the negation and was wrong. Image indices grow
#     right and DOWN while MuJoCo's camera +y is UP, and render_util builds
#     top = ify*(sh/2 - cy) / bottom = -ify*(sh/2 + cy), so positive cy tilts the
#     image-centre ray DOWN. Inverting the kernel's own arithmetic for "which
#     pixel does dir=(0,0,-1) land on" reproduces librealsense's ppx/ppy to
#     0.0000 px with the negation and misses by (-0.95, -7.25) px without it.
#     scripts/calib/compare_intrinsics.py does the conversion and shows the residual.
# CALIBRATED 2026-07-29 from the actual device, replacing the datasheet placeholder
# (which was fx = fy = 433 with a perfectly centred principal point). Read off the
# 848x480 depth profile: fx = fy = 428.3054, ppx = 424.4773, ppy = 243.6254, which
# the device self-reports as 89.42 x 58.52 deg -- consistent, since
# 2*atan(424/428.3054) = 89.42 and 2*atan(240/428.3054) = 58.52.
#
# What actually changed, per scripts/calib/compare_intrinsics.py: focal 1.1% shorter, so
# the field is 0.6-0.9 deg WIDER (more framing margin, not less), and the principal
# point is 3.63 px below centre in y, which puts the scene 2.72 mm off where the
# placeholder had it at the 321 mm working distance. That last one is the reason to
# bother: it is the same order as the +-3 mm camera-pose jitter, so leaving it at
# zero was quietly asserting something false.
#
# `sensorsize` is set to 4.77 x 2.70 mm so its ASPECT equals 848/480 exactly. Only
# the aspect and the implied pixel pitch matter; matching the resolution's aspect
# keeps the renderer's crop branch (below) attributable to the render size alone.
#
# ASPECT RATIO: the student renders 160x120 (4:3) while this sensor is 16:9, and
# mujoco_warp/_src/render_util.py:96-103 shrinks whichever sensor dimension is too
# large for the render aspect. So the width is CROPPED 4.77 -> 3.60 mm and the
# student sees 73.53 deg H x 58.53 deg V, not the 89.42 x 58.52 these intrinsics
# describe.
#
# That crop is EXACTLY a 640x480 centre-crop of the calibrated stream, and this is
# arithmetic rather than a near-coincidence: 3.60/4.77 = 0.754717 and
# 848 * 0.754717 = 640.0000 px, giving 73.5291 deg by either route. A centred crop
# also leaves the principal-point offset unchanged (+0.4773 px before and after),
# so nothing about the calibration above needs adjusting.
#
# Consequently the deployment recipe is one line and no retrain:
#     848x480  ->  centre-crop to 640x480 (drop 104 columns each side)  ->  160x120
# The only genuinely wrong thing is resizing the FULL 848x480 to 160x120 on the
# robot: that squashes 89.42 deg into pixels the student learned as 73.53 deg.
#
# Keeping the crop is also the better choice on the merits, not a compromise -- at
# 160 px wide it is 2.176 px/deg against 1.789 for a 16:9 render, i.e. 1.48x the
# cube's pixel area, and the cube is the small thing that has to be resolved. The
# wider field buys nothing: the entire spawn jitter box already sits inside the
# cropped frame with the worst coordinate at 0.628 of half-frame
# (scripts/calib/aim_camera.py). Alok's call, 2026-07-29.
CAMERA_RESOLUTION = (848, 480)
CAMERA_SENSORSIZE = (4.77e-3, 2.70e-3)  # aspect == 848/480 exactly
CAMERA_FOCALPIXEL = (428.3054, 428.3054)  # device, 848x480 depth profile
CAMERA_PRINCIPALPIXEL = (-0.4773, -3.6254)  # -(ppx - W/2), -(ppy - H/2); see above

# --- sys-id'd arm joint params (robot-actuator-model rizon_hand.xml) -------- #
# damping / frictionloss / armature / actuatorfrcrange, verbatim.
_SYSID_JOINTS: dict[str, tuple[float, float, float, float]] = {
  #          damping  frictionloss  armature  frcrange
  "joint1": (0.61, 0.53, 0.0, 123.0),
  "joint2": (0.0, 0.63, 0.0, 123.0),
  "joint3": (0.0, 0.47, 0.0, 64.0),
  "joint4": (0.0, 0.30, 0.0, 64.0),
  "joint5": (0.20, 0.13, 0.02, 39.0),
  "joint6": (0.0, 0.21, 0.02, 39.0),
  "joint7": (0.14, 0.13, 0.02, 39.0),
}

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = tuple(
  f"{n}{_SUFFIX}" for n in ("left_1", "left_2", "right_1", "right_2")
)

# --- finger servo (DC15-A01 class) ----------------------------------------- #
_FINGER_KP = 20.0
# Servo velocity gain. Unlike the arm joints, the finger params are DC15-A01
# class ESTIMATES, not sys-id'd, so this one is a legitimate knob. At 0.5 the
# actuator stays saturated at 2 N.m until the joint reaches ~10 rad/s, i.e. the
# pads sweep shut at ~0.42 m/s -- enough that a one-sided touch launches a 60 g
# cube before the far pad arrives. Raising it caps closing speed (kv 4.0 ->
# ~1.25 rad/s, ~0.05 m/s at the pad), which is also closer to what a real hobby
# servo does. Overridable so the two effects can be separated in one sweep.
#
# DEFAULT IS 4.0, NOT 0.5. 0.5 is the value the ejection failure was diagnosed
# AT, not the value it was fixed to, and leaving it as the default made every
# script that forgot the flag silently run broken physics. That is not
# hypothetical -- it cost most of a day. The same checkpoint, 512 envs, 199
# steps, nothing different but this number:
#
#     kv 0.5 -> held 61%, ejected 39%, peak pad force 174 N
#     kv 4.0 -> held 100%, ejected  0%, peak pad force  20.7 N
#
# Evaluated at 0.5, a policy trained at 4.0 looks like it punts the cube off the
# table in 40% of episodes, and the failure is convincing: it reproduces, it is
# spatially structured, and it survives every sample-size check. It was
# scripts/diag/diag_policy_behaviour.py, scripts/render/render_many.py and slurm/train/distill.sh
# all defaulting to a damping the policy had never trained under. A default that
# is wrong for every current caller is a trap, so the validated value is now the
# default and the broken one has to be asked for explicitly.
_FINGER_KV = float(os.environ.get("CG_FINGER_KV") or "4.0")
_FINGER_FORCERANGE = (-2.0, 2.0)  # repo URDF: effort=2
_FINGER_RANGE = (-1.6, 1.6)  # repo joint range
_FINGER_ARMATURE = 0.01
_FINGER_DAMPING = 0.1
_FINGER_FRICTIONLOSS = 0.05

# Pad marker bodies (contact patch on each distal link) + touch sensors.
PAD_BODIES = (f"left_pad{_SUFFIX}", f"right_pad{_SUFFIX}")
TOUCH_SENSORS = (f"left_touch{_SUFFIX}", f"right_touch{_SUFFIX}")
PREFIXED_TOUCH_SENSORS = tuple(f"robot/{s}" for s in TOUCH_SENSORS)

# --- aperture ---------------------------------------------------------------
# Pad separation is affine in the proximal angle. MEASURED off the compiled
# model, not derived: set the two proximal joints, read the pad-marker
# separation, fit a line.
#
#     sep(q) = 50.1 + 134.0 q   mm        (fit residual < 0.25 mm over -0.15..0.40)
#
# The same procedure applied to the former 56 mm claw returned
# sep = 40.1 + 168.3 q, reproducing that claw's own independently derived
# "40 + 168.6 q" -- which is what established that the procedure is sound.
#
# Sign convention: left_1 POSITIVE / right_1 NEGATIVE swings the pads APART.
APERTURE_A, APERTURE_B = 50.1, 134.0


def aperture_mm(q: float) -> float:
  """Pad separation in mm at proximal angle q."""
  return APERTURE_A + APERTURE_B * q


def open_angle_for(sep_mm: float) -> float:
  """Proximal angle giving a pad separation of sep_mm."""
  return (sep_mm - APERTURE_A) / APERTURE_B


# The open default is set by ONE requirement: at FULL CLOSE the commanded angle
# must land a ~10 mm squeeze on the 50 mm cube -- firm grip, no drive-through.
# The action is q = default + _HAND_SCALE * a, so
#
#     FINGER_OPEN_ANGLE = open_angle_for(40 mm) + _HAND_SCALE
#                       = -0.0754 + 0.35
#
# _HAND_SCALE is deliberately NOT the knob that moves to satisfy this. It is
# coupled to init_std -- the noise injected per joint is scale * std against a
# validated 0.15 rad budget -- so changing it changes exploration as well as
# reach, and env_cfgs.py records a run that was lost that way.
#
# This claw is FLUSH on the cube at q = 0 (50.1 mm, zero squeeze), so its open
# pose sits at 0.275 and the open aperture is 86.9 mm. That is 18.5 mm of
# clearance per side against the straddle gate's own 15 mm xy tolerance, so a
# pose that passes the gate still clears the cube -- but it is tight. If
# training shows the fingers clipping the cube on approach, raise _HAND_SCALE to
# ~0.40 and this to ~0.325 TOGETHER (93.7 mm open, same 10 mm squeeze).
FINGER_OPEN_ANGLE = round(open_angle_for(40.0) + 0.35, 4)

FINGER_OPEN_POS: dict[str, float] = {
  f"left_1{_SUFFIX}": FINGER_OPEN_ANGLE,
  f"left_2{_SUFFIX}": 0.0,
  f"right_1{_SUFFIX}": -FINGER_OPEN_ANGLE,
  f"right_2{_SUFFIX}": 0.0,
}

# NOTE: this file no longer carries any hand mass, inertia or collision-box
# constants. They were estimates for the former 56 mm claw ("everything but the
# motors is 3D printed", 0.7 g/cm^3) and were wrong by 2.1x in total: 434 g
# estimated against the fitted claw's CAD-computed 209 g, with base_link alone
# 3.8x heavy. Everything is now read from hands/<CG_HAND>/hand.urdf by
# hand_urdf.py, and the collision boxes are fitted to the meshes there.

# The whole claw renders in ONE blue, against a white cube (see CUBE_RGBA in the
# task's env_cfgs). The old scheme gave every part its own colour -- green
# adapter, purple shim, orange mount, blue left finger, red right finger -- which
# was useful for spotting a mis-seated mesh but is the wrong choice for an RGB
# student: it hands the network four incidental colour boundaries inside the
# gripper and only a weak one at the object. Uniform claw, uniform cube, and the
# only strong colour edge in the image is the one that matters. Depth is
# unaffected either way, so this changes nothing for the current depth student
# or for the state-based teacher.
_CLAW_BLUE = (0.13, 0.34, 0.85, 1.0)


def _add_visual(body: mujoco.MjsBody, mesh: str, pos, quat, rgba) -> None:
  g = body.add_geom()
  g.name = f"{body.name}_{mesh}_vis"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = mesh
  g.pos = np.array(pos)
  g.quat = np.array(quat)
  g.rgba = np.array(rgba)
  g.contype = 0
  g.conaffinity = 0
  g.group = 1


def _add_collision_box(body: mujoco.MjsBody, name: str, halfsize, pos) -> None:
  g = body.add_geom()
  g.name = name
  g.type = mujoco.mjtGeom.mjGEOM_BOX
  g.size = np.array(halfsize)
  g.pos = np.array(pos)
  g.contype = FINGER_BIT
  g.conaffinity = OBJECT_BIT | FINGER_BIT
  g.friction = np.array([1.0, 0.05, 0.001])  # printed pad; object priority wins pair
  g.rgba = np.array([0.3, 0.3, 0.3, 0.4])
  g.group = 3


# Which collision representation the hand is built from. "boxes" is the default
# and the one every checkpoint was trained against; the coacd_* sets are the
# ones in assets/hands/wide/collision/, whose README carries the measurements
# behind this choice.
#
# MEASURED on the gripping band of `right_2`, sampled across the pad patch
# rather than at the bounding vertices (the bounds measurement is trivially
# zero and says nothing about contact):
#
#   fitted boxes   0.90 mm     coacd_t0.2   0.35 mm     coacd_t0.4   0.35 mm
#
# COACD is the better fit and it plateaus at +0.346 mm at EVERY threshold --
# that floor is the remesh of three non-watertight parts, not the decomposition,
# so buying a finer set buys nothing.
# NOT an environment variable, and that is the point. This changes the PHYSICS
# -- the same full-close command grips at 12.4 N on boxes and 7.0 N on
# coacd_t0.2 -- so a checkpoint is only valid under the setting it was trained
# with. An env var makes the mismatch SILENT: the policy would simply look
# worse, with nothing raising. It is threaded from the task registration instead
# so the task id records it and the two can never be paired wrongly.
HAND_COLLISION_SETS = ("boxes", "coacd_t0.15", "coacd_t0.2", "coacd_t0.4")


def _add_collision_mesh(body: mujoco.MjsBody, name: str, meshname: str) -> None:
  r"""One convex hull, wearing the same contact identity as `_add_collision_box`.

  The NAME MATTERS and must keep the `_col<k>` shape: `FINGERTIP_GEOMS` and the
  three `fingertip_friction_*` events select the fingertips by the regex
  `(left_2|right_2)_col\d_tf`, and a pattern that matches nothing reports no
  contact rather than raising. Naming the hulls the way the boxes were named is
  what lets the whole task config stay untouched by this switch.
  """
  g = body.add_geom()
  g.name = name
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = meshname
  g.contype = FINGER_BIT
  g.conaffinity = OBJECT_BIT | FINGER_BIT
  g.friction = np.array([1.0, 0.05, 0.001])  # printed pad; object priority wins pair
  g.rgba = np.array([0.3, 0.3, 0.3, 0.4])
  g.group = 3


def _coacd_parts(tag: str) -> dict[str, list[str]]:
  """{part: [mesh name, ...]}, registering nothing -- the caller adds the meshes.

  Hulls are stored in each PART'S OWN FRAME, matching the STL they came from, so
  they need no pos/quat when attached to the body that already carries that
  part's visual mesh.
  """
  import json

  man = json.loads((COLLISION_DIR / tag / "manifest.json").read_text())
  return {part: entry["files"] for part, entry in man.items()}


def _add_finger_joint(body: mujoco.MjsBody, name: str, cad) -> None:
  """Hinge for one finger link.

  Axis and travel come from the CAD mate connectors (``cad`` is a
  ``hand_urdf.Joint``). Damping, armature, frictionloss and the actuator force
  range stay local: those are sim tuning rather than geometry, and the URDF's
  effort limit is applied as the force range only.
  """
  j = body.add_joint()
  j.name = name
  j.type = mujoco.mjtJoint.mjJNT_HINGE
  j.axis = np.array(cad.axis)
  j.range = np.array([cad.lower, cad.upper])
  j.armature = _FINGER_ARMATURE
  j.damping = np.array([_FINGER_DAMPING, 0.0, 0.0])  # per-DOF vec in mj>=3.8
  j.frictionloss = _FINGER_FRICTIONLOSS
  j.actfrclimited = 1
  j.actfrcrange = np.array(_FINGER_FORCERANGE)


def _add_finger_actuator(spec: mujoco.MjSpec, joint: str) -> None:
  a = spec.add_actuator()
  a.name = joint
  a.trntype = mujoco.mjtTrn.mjTRN_JOINT
  a.target = joint
  a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  gp = np.zeros(mujoco.mjNGAIN)
  gp[0] = _FINGER_KP
  a.gainprm = gp
  bp = np.zeros(mujoco.mjNBIAS)
  bp[1] = -_FINGER_KP
  bp[2] = -_FINGER_KV
  a.biasprm = bp
  a.ctrlrange = np.array(_FINGER_RANGE)
  a.ctrllimited = 1
  a.forcerange = np.array(_FINGER_FORCERANGE)
  a.forcelimited = 1


def _union_box(boxes) -> tuple:
  """Smallest box containing all of ``boxes`` (halfsize, pos) pairs."""
  lo = np.min([np.array(p) - np.array(h) for h, p in boxes], axis=0)
  hi = np.max([np.array(p) + np.array(h) for h, p in boxes], axis=0)
  return (tuple((hi - lo) / 2.0), tuple((hi + lo) / 2.0))


def _add_pad_marker_at(body: mujoco.MjsBody, name: str, body_pos, site_pos) -> None:
  """Marker body at the GRIPPING FACE, site at the FINGERTIP.

  THE TWO ARE 24 mm APART AND THAT IS DELIBERATE. They are references for two
  reward sets that were each calibrated against a different point, and making
  one stand in for the other silently mis-measures the grasp:

    * the BODY sits at ``pad_marker_pos`` -- the centroid of the inner
      gripping face, where a 50 mm cube actually bears. This repo's own
      rewards read it via ``SceneEntityCfg(body_names=PAD_BODIES)``, and the
      aperture law ``sep(q) = 50.1 + 134.0 q`` was FITTED to these markers, so
      moving them invalidates that fit and CONTACT_DIST along with it.
    * the SITE sits at ``tip_marker_pos`` -- the fingertip, matching the 56 mm
      claw's ``left_pad``/``right_pad`` at link-frame z = -0.0440. The mjlab
      manipulation rewards (``antipodal_pinch``, ``pad_site_touch``,
      ``jaw_face_alignment``, the reaching half of ``lift``) read sites, and
      every one of their constants -- ``PAD_SEP_AT_GRASP`` above all -- was
      tuned against a point at that height.

  Carrying that height across is justified rather than assumed: the two claws'
  distal links are the same part. Both meshes measure x -10..+10, y -3.3..29.7,
  z -55..+8, both hold the flat pad at inner x = 10.00 over z -20..-32, and
  both taper the same way below it (inner x at z -40 / -44: 4.62 / 1.33 on the
  56 mm claw, 4.52 / 0.04 on this one). The 70 mm knuckle spacing is in
  base_link, not in the finger.

  The site is NOT a contact point and nothing should treat it as one -- it
  sits behind the gripping face, so the pair straddles the cube with clearance
  on both sides. That is the property the mjlab constants encode.
  """
  pad = body.add_body()
  pad.name = name
  pad.pos = np.array(body_pos)
  s = body.add_site()
  s.name = name
  s.pos = np.array(site_pos)
  s.size = np.array([0.005, 0.0, 0.0])
  s.rgba = np.array([0.10, 0.95, 0.35, 1.0])
  s.group = SITE_GROUP_REF


# Site groups. 4 holds the touch-sensor VOLUMES, which are boxes spanning a
# whole distal link -- correct as sensor geometry, useless in a render, where
# they are opaque slabs that bury the hand. 5 holds the measured REFERENCE
# POINTS (pad patches, grasp centre), so a viewer can switch on the points
# without switching on the volumes. Both are off in MuJoCo's default MjvOption
# (sitegroup defaults to [1,1,1,0,0,0]), so neither shows unless asked for.
SITE_GROUP_SENSOR = 4
SITE_GROUP_REF = 5

# --- grasp centre, on the palm ----------------------------------------------- #
# A fixed point on the PALM standing in for "where the grasp will happen". The
# mjlab manipulation rewards reach toward it, so it is the target the whole
# approach is shaped around. Both components are MEASURED, but from two
# different requirements, and mixing them up is how it goes wrong.
#
# y = +0.00435, from geometry.  The pads are not symmetric about the palm's
# y = 0 plane. Sweeping the proximal joint over its whole working range
# (q = -0.0754 .. +0.2746, aperture 40.0 .. 86.9 mm) and taking the midpoint of
# the two FINGERTIP SITES in the palm frame gives y = +4.350 mm, constant to
# under a micron across the sweep, with x = 0 exactly.
#
# This agrees with the 56 mm claw's +0.005 measured the same way, which is a
# check rather than a coincidence: the two distal links are the same part, so
# the tip references land in the same place relative to the finger. An earlier
# cut of this constant read -0.0030 and claimed the sign was OPPOSITE the old
# claw's -- that came from measuring the GRIPPING-FACE markers instead of the
# tip sites, which are 24 mm apart along the finger and sit on opposite sides of
# its y midline. The pads and the tips genuinely do have opposite y offsets;
# only the tip one is comparable to the mjlab constant.
#
# z = -0.1070, from TIP CLEARANCE, not from the site midpoint.  The measured
# site midpoint is at -0.1221 (stable to 2.96 mm over the sweep), but the site
# is not what has to clear the table -- the fingertips are, and the lowest
# fingertip geom sits at -0.1340 in the palm frame, 11.9 mm below the site
# midpoint, while the cube centre is only 25 mm above the work surface.
#
# So z is solved for the clearance the task's reward assumes -- "the moment
# grasp_site reaches the cube centre the tips clear the table" -- keeping the
# ~2 mm margin the 56 mm claw was tuned to:
#
#     z = tip_z + cube_half + margin = -0.1340 + 0.025 + 0.002 = -0.1070
#
# NOTE THE SIGN, it is counter-intuitive: moving the site DOWN (more negative,
# toward the tips) makes the hand sit HIGHER for the same site target.
#
# Against the 56 mm claw's -0.099 this is 8 mm lower, which is geometry and not
# a retune: this claw hangs its tips further below the palm.
#
# MEASURED alongside, because it is the constant that does NOT carry across and
# it is the one a wired-up task will read next:
#
#     fingertip-site separation, fingers flush on the 50 mm cube
#         this claw    67.21 mm   (each site 8.60 mm clear of the cube face)
#         56 mm claw   64.40 mm   (7.2 mm clear per side)
#
# Neither is 50 mm and neither should be: the sites sit BEHIND the gripping
# faces, so the pair straddles the cube rather than touching it. Anything
# comparing this separation against an object width must use a measured value,
# never 2 * cube_half.
_GRASP_SITE_POS = (0.0, 0.00435, -0.1070)


def _add_grasp_site(base: mujoco.MjsBody) -> None:
  """Grasp centre on the palm; see _GRASP_SITE_POS for both derivations."""
  s = base.add_site()
  s.name = "grasp_site"
  s.pos = np.array(_GRASP_SITE_POS)
  s.size = np.array([0.008, 0.0, 0.0])
  s.rgba = np.array([1.0, 0.15, 0.10, 1.0])
  s.group = SITE_GROUP_REF


def _add_touch_sensor_box(
  spec: mujoco.MjSpec, body: mujoco.MjsBody, name: str, box
) -> None:
  """Touch sensor sized to this link's own collision box, + 3 mm skin."""
  half, pos = box
  site = body.add_site()
  site.name = f"{name}_site"
  site.type = mujoco.mjtGeom.mjGEOM_BOX
  site.pos = np.array(pos)
  site.size = np.array(half) + 0.003
  site.group = SITE_GROUP_SENSOR
  s = spec.add_sensor()
  s.name = name
  s.type = mujoco.mjtSensor.mjSENS_TOUCH
  s.objtype = mujoco.mjtObj.mjOBJ_SITE
  s.objname = site.name


def _set_inertial_full(body: mujoco.MjsBody, link) -> None:
  """Mass, COM and the FULL inertia tensor straight from the CAD link.

  The CAD tensors carry real off-diagonal terms -- order 1e-7 against 1e-5
  diagonals. Small, but free to carry, and the whole point of reading the CAD
  is not to launder it through a simplification on the way in.
  """
  body.mass = link.mass
  body.ipos = np.array(link.com)
  body.fullinertia = np.array(link.fullinertia)
  body.explicitinertial = True


def _hand_spec(collision: str = "boxes") -> mujoco.MjSpec:
  """The 70 mm-knuckle claw, built from the vendored CAD URDF.

  Same shape of spec as ``_hand_spec`` -- root body ``hand_base``, one hinge
  per finger link, box collisions, pad markers, touch sensors -- but every
  geometric and inertial number is read rather than transcribed. See
  ``hand_urdf.py`` for what is and is not taken from the URDF.
  """
  cad = load_hand_cad()
  boxes = fit_collision_boxes(cad)

  spec = mujoco.MjSpec()
  spec.modelname = "two_finger_hand_wide"
  spec.compiler.degree = False  # every angle below is RADIANS
  for n in ("base_link", "adapter", "shim", "left_1", "left_2", "right_1", "right_2"):
    m = spec.add_mesh()
    m.name = n
    m.file = str(WIDE_MESH_DIR / f"{n}.stl")

  # The alternate collision set, if one is selected. Registered here so the
  # geoms below can just reference names; `hulls` stays empty for "boxes" and
  # every branch below falls back to the fitted boxes unchanged.
  hulls: dict[str, list[str]] = {}
  if collision not in HAND_COLLISION_SETS:
    raise ValueError(
      f"collision={collision!r}; expected one of {list(HAND_COLLISION_SETS)}"
    )
  if collision != "boxes":
    for part, files in _coacd_parts(collision).items():
      names = []
      for k, rel in enumerate(files):
        nm = f"col_{part}_{k}"
        mh = spec.add_mesh()
        mh.name = nm
        mh.file = str(COLLISION_DIR / collision / rel)
        names.append(nm)
      hulls[part] = names

  base = spec.worldbody.add_body()
  base.name = "hand_base"
  # Visual stack, with each part's pose taken from its own fixed joint rather
  # than from a table of hand-copied offsets.
  _add_visual(base, "base_link", (0, 0, 0), (1, 0, 0, 0), _CLAW_BLUE)
  for part in ("adapter", "shim"):
    j = cad.joints[f"{part}_fixed"]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array(j.rpy), "xyz")
    _add_visual(base, part, j.origin, tuple(quat), _CLAW_BLUE)

  # Palm and wrist collision. The wide base_link is symmetric in y, so unlike
  # the narrow hand there is no offset stack to box separately -- base_link's
  # own fitted box covers the palm, and the adapter/shim box covers the wrist
  # stack that sweeps low on an aggressive descent.
  _add_grasp_site(base)
  if hulls:
    # base_link's hulls sit in its own frame, which IS this body's frame.
    for k, nm in enumerate(hulls["base_link"]):
      _add_collision_mesh(base, f"palm_col{k}", nm)
    # The adapter and shim are welded to base_link rather than being separate
    # bodies, so their hulls are attached here too -- that is the wrist stack
    # the box version covered with one `wrist_col`.
    for part in ("adapter", "shim"):
      for k, nm in enumerate(hulls.get(part, [])):
        _add_collision_mesh(base, f"{part}_col{k}", nm)
  else:
    _add_collision_box(base, "palm_col", *boxes["base_link"][0])
    wrist_half, wrist_pos = boxes["adapter"][0]
    j_ad = cad.joints["adapter_fixed"]
    _add_collision_box(
      base,
      "wrist_col",
      wrist_half,
      tuple(np.array(j_ad.origin) + np.array([0, 0, -wrist_pos[2]])),
    )
  _set_inertial_full(base, cad.links["base_link"])
  # The adapter and shim are welded to base_link and are not separate bodies
  # here, so their mass has to be added or the hand comes out 56 g light.
  base.mass = (
    cad.links["base_link"].mass + cad.links["adapter"].mass + cad.links["shim"].mass
  )

  for side in ("left", "right"):
    prox_j, dist_j = cad.joints[f"{side}_1"], cad.joints[f"{side}_2"]
    prox = base.add_body()
    prox.name = f"{side}_1"
    prox.pos = np.array(prox_j.origin)
    _add_finger_joint(prox, f"{side}_1", prox_j)
    _add_visual(prox, f"{side}_1", (0, 0, 0), (1, 0, 0, 0), _CLAW_BLUE)
    if hulls:
      for k, nm in enumerate(hulls[f"{side}_1"]):
        _add_collision_mesh(prox, f"{side}_1_col{k}", nm)
    else:
      for k, box in enumerate(boxes[f"{side}_1"]):
        _add_collision_box(prox, f"{side}_1_col{k}", *box)
    _set_inertial_full(prox, cad.links[f"{side}_1"])

    dist = prox.add_body()
    dist.name = f"{side}_2"
    dist.pos = np.array(dist_j.origin)
    _add_finger_joint(dist, f"{side}_2", dist_j)
    _add_visual(dist, f"{side}_2", (0, 0, 0), (1, 0, 0, 0), _CLAW_BLUE)
    if hulls:
      for k, nm in enumerate(hulls[f"{side}_2"]):
        _add_collision_mesh(dist, f"{side}_2_col{k}", nm)
    else:
      for k, box in enumerate(boxes[f"{side}_2"]):
        _add_collision_box(dist, f"{side}_2_col{k}", *box)
    _set_inertial_full(dist, cad.links[f"{side}_2"])
    _add_pad_marker_at(dist, f"{side}_pad", pad_marker_pos(side), tip_marker_pos(side))
    # Touch site spans the WHOLE distal link (union of its bands), so the
    # sensor still reports one force for the finger however it is boxed.
    _add_touch_sensor_box(spec, dist, f"{side}_touch", _union_box(boxes[f"{side}_2"]))

  for j in ("left_1", "left_2", "right_1", "right_2"):
    _add_finger_actuator(spec, j)
  return spec


def _strip_leading_slashes(spec: mujoco.MjSpec) -> None:
  """Strip the leading '/' MjSpec.attach() prepends so mjlab 'robot/<name>'
  lookups resolve (same as rizon_jaw_cfg / allegro_arm_cfg)."""

  def s(x: str) -> str:
    return x.lstrip("/")

  for coll in (
    spec.bodies,
    spec.joints,
    spec.actuators,
    spec.sites,
    spec.geoms,
    spec.tendons,
    spec.sensors,
    spec.meshes,
  ):
    for e in coll:
      if e.name.startswith("/"):
        e.name = s(e.name)
  for g in spec.geoms:
    if isinstance(g.meshname, str) and g.meshname.startswith("/"):
      g.meshname = s(g.meshname)
  for a in spec.actuators:
    if isinstance(a.target, str) and a.target.startswith("/"):
      a.target = s(a.target)
  for sn in spec.sensors:
    if isinstance(sn.objname, str) and sn.objname.startswith("/"):
      sn.objname = s(sn.objname)


def _add_baseplate(arm: mujoco.MjSpec) -> None:
  """Bolt the 225 x 225 x 15 mm baseplate under the arm base body.

  Visual = the exported Onshape mesh (mount holes and all); collision = a
  plain box on the same footprint (cheap + stable, the plate IS a box). Bits
  mirror the table (contype 5 / conaffinity 3) so the plate is a real surface
  for the fingers and the cube. Plate-vs-arm-base needs no exclude: both are
  welded to the world, so MuJoCo filters that pair automatically.
  """
  base = next(b for b in arm.bodies if b.name == "base")

  mesh = arm.add_mesh()
  mesh.name = BASE_PLATE_MESH
  mesh.file = str(_ASSET_DIR / f"{BASE_PLATE_MESH}.stl")

  vis = base.add_geom()
  vis.name = "baseplate_vis"
  vis.type = mujoco.mjtGeom.mjGEOM_MESH
  vis.meshname = BASE_PLATE_MESH
  vis.rgba = np.array([0.58, 0.60, 0.63, 1.0])
  vis.mass = 0.0
  vis.contype = 0
  vis.conaffinity = 0
  vis.group = 1

  col = base.add_geom()
  col.name = "baseplate_col"
  col.type = mujoco.mjtGeom.mjGEOM_BOX
  col.size = np.array(BASE_PLATE_HALF)
  col.pos = np.array([0.0, 0.0, -BASE_PLATE_HALF[2]])  # top face at z = 0
  col.mass = BASE_PLATE_MASS
  col.contype = 5  # 1|4: arm/world + object bits
  col.conaffinity = 3  # 1|2: arm/world + finger bits
  col.friction = np.array([1.0, 0.05, 0.001])
  col.rgba = np.array([0.3, 0.3, 0.3, 0.4])
  col.group = 3


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
  """wxyz quat for a MuJoCo camera at ``eye`` looking at ``target``.

  MuJoCo cameras look down their own -z with +y up, so the camera frame is
  [x | y | z] with z = -forward.
  """
  z = np.asarray(eye, float) - np.asarray(target, float)
  z /= np.linalg.norm(z)
  y = np.asarray(up, float) - np.dot(up, z) * z
  y /= np.linalg.norm(y)
  x = np.cross(y, z)
  q = np.empty(4)
  mujoco.mju_mat2Quat(q, np.column_stack((x, y, z)).flatten())
  return q


def _add_camera_mount(arm: mujoco.MjSpec) -> None:
  """Bolt the static camera mount to the plate and hang a camera off it.

  Added as a JOINTLESS CHILD of ``base``, which is exactly the collision
  behaviour we want: child-of-base means MuJoCo filters mount<->base (they
  are tangent by design, so that pair must not generate contacts), while
  link1..link7 sit in a different weld body and DO collide with it -- the
  mast is a real obstacle the arm has to respect, not scenery.
  """
  base = next(b for b in arm.bodies if b.name == "base")

  mesh = arm.add_mesh()
  mesh.name = CAMERA_MOUNT_MESH
  mesh.file = str(_ASSET_DIR / f"{CAMERA_MOUNT_MESH}.stl")

  body = base.add_body()
  body.name = "camera_mount"
  body.pos = np.array([CAMERA_MOUNT_OFFSET, 0.0, 0.0])
  body.quat = np.array(
    [np.cos(CAMERA_MOUNT_YAW / 2), 0.0, 0.0, np.sin(CAMERA_MOUNT_YAW / 2)]
  )

  vis = body.add_geom()
  vis.name = "camera_mount_vis"
  vis.type = mujoco.mjtGeom.mjGEOM_MESH
  vis.meshname = CAMERA_MOUNT_MESH
  vis.rgba = np.array([0.82, 0.45, 0.20, 1.0])
  vis.mass = 0.0
  vis.contype = 0
  vis.conaffinity = 0
  vis.group = 1

  for name, half, pos in _CAM_BOXES:
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = np.array(half)
    g.pos = np.array(pos)
    g.mass = 0.0
    g.contype = 5  # 1|4: arm/world + object bits (like table)
    g.conaffinity = 3  # 1|2: arm/world + finger bits
    g.rgba = np.array([0.3, 0.3, 0.3, 0.4])
    g.group = 3

  # --- bolt the RealSense D435i onto the 27.45 deg shelf ------------------- #
  # Body frame: +z = optical axis, x = lens row, back face at z = -25.1 mm.
  # Seat it with the BACK PLATE FLAT ON THE SHELF and +z along the shelf
  # normal, so the part's own 27.45 deg geometry sets the camera angle. x is
  # put along mount-local +x, which the +pi/2 yaw maps to world +y -- i.e. the
  # lens row ends up horizontal, across the scene (landscape).
  face_n = np.asarray(_CAM_FACE_NORMAL, float)
  z_b = face_n
  x_b = np.array([1.0, 0.0, 0.0])
  y_b = np.cross(z_b, x_b)
  rot_b = np.column_stack((x_b, y_b, z_b))
  origin_b = np.asarray(_CAM_FACE_CENTRE, float) + _D435I_DEPTH * face_n

  site = body.add_site()
  site.name = "d435i_mount"
  site.pos = origin_b
  q = np.empty(4)
  mujoco.mju_mat2Quat(q, rot_b.flatten())
  site.quat = q
  site.group = 4

  cam_spec = mujoco.MjSpec.from_file(str(_D435I_XML))
  for light in list(cam_spec.lights):  # drop the model's own light
    cam_spec.delete(light)
  for g in cam_spec.geoms:  # match the mount's bits
    if g.contype or g.conaffinity:
      g.contype, g.conaffinity = 5, 3
      # ... and re-fit its ONE collider as a box rather than the capsule
      # Menagerie fits to the metal casing.
      #
      # THIS IS NO LONGER REQUIRED and is retained only because the trained
      # policies were fitted against it. It was added on 2026-08-06 as a
      # workaround for an NVRTC "unable to obtain mapped memory" failure, on the
      # theory that the extra CAPSULE-BOX primitive collision pair pushed the
      # JIT-generated `primitive_narrowphase` past what the compiler could take.
      # That theory was WRONG: the real cause was NVRTC's precompiled headers
      # (see mjlab/__init__.py), and with those off the capsule compiles fine --
      # verified by scripts/wide_diag_workarounds.py, which builds this scene
      # with the capsule restored AND njmax at 1100.
      #
      # Kept for now because reverting mid-pipeline would change the model the
      # current teacher and student were trained in. It is a static collider at
      # the arm base that no policy here ever touches, so the divergence from
      # the reference model is inert -- but it IS a divergence. Revert at the
      # next retrain.
      if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        g.type = mujoco.mjtGeom.mjGEOM_BOX
  arm.attach(cam_spec, prefix="d435i_", site=site.name)

  # Camera at the D435i's IR (depth) lens, looking along the body +z, which
  # the seating above has aligned with the shelf normal.
  lens_b = np.asarray(_D435I_IR_LENS, float)
  eye_l = origin_b + rot_b @ lens_b

  # Then the measured correction (see CALIB_CAM_DROT). Kept as a separate,
  # clearly-marked step rather than folded back into _CAM_FACE_NORMAL so the
  # CAD chain above still reads as CAD and stays diffable against a mount
  # revision, while the measurement stays diffable against a recalibration.
  R_cad = mujoco_quat_to_mat(_look_at_quat(eye_l, eye_l + face_n, up=(0.0, 0.0, 1.0)))
  R_cal = R_cad @ CALIB_CAM_DROT
  q_cal = np.empty(4)
  mujoco.mju_mat2Quat(q_cal, R_cal.flatten())

  cam = body.add_camera()
  cam.name = "scene_cam"
  cam.pos = eye_l + np.asarray(CALIB_CAM_OFFSET, float)
  cam.quat = q_cal
  cam.resolution = np.array(CAMERA_RESOLUTION)
  cam.sensor_size = np.array(CAMERA_SENSORSIZE)
  cam.focal_pixel = np.array(CAMERA_FOCALPIXEL)
  cam.principal_pixel = np.array(CAMERA_PRINCIPALPIXEL)


def _apply_sysid(arm: mujoco.MjSpec) -> None:
  """Override Menagerie passive joint params with the sys-id values."""
  joints = {j.name: j for j in arm.joints}
  for name, (damping, frictionloss, armature, frc) in _SYSID_JOINTS.items():
    j = joints[name]
    j.damping = np.array([damping, 0.0, 0.0])  # per-DOF vec in mj>=3.8
    j.frictionloss = frictionloss
    j.armature = armature
    j.actfrclimited = 1
    j.actfrcrange = np.array([-frc, frc])


# --- gravity compensation ---------------------------------------------------- #
# The real Rizon's controller compensates gravity, and the tool payload too once
# it is configured. Menagerie's position servos are a bare PD against it, so the
# same command that holds the arm level on hardware sagged ~7 cm here. Every arm
# pose in this file used to be a droop-compensated TARGET, solved by inverting
# that sag -- which taught the policy to pre-compensate a bias the real arm does
# not have, and would have shown up on hardware as a systematic overshoot of
# roughly the sag itself.
#
# TWO SETTINGS, and they are not the same:
#   body gravcomp  -- the compensating force lands in qfrc_passive, where it is
#                     FREE and no actuator limit can clip it.
#   joint actgravcomp -- it is routed through the actuator instead, so
#                     actfrcrange clips it.
# The second is the faithful one: on hardware, holding against gravity IS motor
# torque and it spends the same budget _SYSID_JOINTS measured. Both are set, so
# the compensation exists and is charged to the right account.
#
# THE HAND LINKS ARE COMPENSATED TOO, and that is an approximation in one
# direction. It is right for the arm joints (a real Flexiv compensates a
# configured payload) and wrong for the finger joints, whose hobby servos
# compensate nothing -- qfrc_gravcomp projects each body's weight onto EVERY dof
# that moves it, so there is no way to give the arm the hand's weight without
# also giving it to the fingers. The error is one finger's own gravity torque on
# its own hinge: MEASURED at 0.0044 N.m on a proximal and 0.0004 on a distal,
# 0.22% and 0.02% of the 2 N.m servo. Small enough to carry.
#
# MEASURED end to end, settling the _HOME_TARGETS in full physics:
#     without   max |joint error| 2.735 deg, pad midpoint 38 mm short in x
#     with      max |joint error| 0.000 deg, pad lands on the commanded pose
# Which is also why every pose below is now WRONG BY THAT MUCH IN THE OTHER
# DIRECTION -- they were solved to overshoot by exactly the sag that no longer
# happens. Re-solve before trusting anything downstream.
#
# Scenery is skipped: the base plate and the camera mast are welded to the world
# with no dof between them and it, so compensating them would be a no-op with a
# misleading name on it.
_NO_GRAVCOMP = frozenset({"world", "base", "camera_mount"})


def _apply_gravcomp(arm: mujoco.MjSpec) -> None:
  """Compensate gravity on the arm + hand, the way the real controller does.

  Call AFTER ``attach``: the hand bodies do not exist before it, and they are
  the payload the arm most needs compensating for.
  """
  for b in arm.bodies:
    if b.name in _NO_GRAVCOMP or b.name.startswith("d435i_"):
      continue
    b.gravcomp = 1.0
  for j in arm.joints:
    if j.name in ARM_JOINTS:
      j.actgravcomp = 1


def twofinger_arm_spec(collision: str = "boxes") -> mujoco.MjSpec:
  """UNCOMPILED MjSpec: sys-id'd Rizon 4S + two-finger hand.

  ``collision`` picks the hand's collision representation; see
  HAND_COLLISION_SETS and the collision directory's README.
  """
  arm = mujoco.MjSpec.from_file(str(_ARM_XML))
  _apply_sysid(arm)
  _add_baseplate(arm)
  _add_camera_mount(arm)

  for b in arm.bodies:
    if b.name == "link7":
      site = b.add_site()
      site.name = TOOL_MOUNT_SITE
      site.pos = np.array(_mount_pos())
      site.quat = np.array(_MOUNT_QUAT)
      break

  # Drop the arm 'home' keyframe (nu changes on attach); mjlab rebuilds it
  # from InitialStateCfg.
  for k in list(arm.keys):
    arm.delete(k)

  arm.attach(_hand_spec(collision), suffix=_SUFFIX, site=TOOL_MOUNT_SITE)
  _strip_leading_slashes(arm)
  _apply_gravcomp(arm)

  # Elliptic cone + Newton for stable pinch friction (softgrasp discipline).
  arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  arm.option.impratio = 10.0
  arm.option.solver = mujoco.mjtSolver.mjSOL_NEWTON

  arm.visual.global_.offwidth = 1280
  arm.visual.global_.offheight = 960
  return arm


# Home: gripper hovering above the table grasp spot, fingers straight down and
# slightly open, pad midpoint at (TABLE_X, 0.00, RESTING_Z + 67.7 mm) = ~4 cm
# above the cube's top face, finger axis = world -z, grip axis = world y. Named
# as constants rather than written out, because literal figures here have gone
# stale twice already. joint1 is PINNED to 0 so the pose
# stays mirror-symmetric about the table centreline (cube-yaw DR is symmetric,
# so the home pose should be too).
#
# NO LONGER DROOP-COMPENSATED, and that is the headline. These used to be CTRL
# TARGETS deliberately offset from the pose they wanted, because the PD servos
# had no gravity compensation and the arm sagged below whatever it was
# commanded; the targets were solved to overshoot by exactly that sag. See
# _apply_gravcomp: the sim now compensates the way the real controller does, the
# measured sag went 2.735 deg -> 0.000, and _HOME_TARGETS and _SETTLED have
# collapsed to the same numbers (they differ by 0.0007 rad on joint2, which is
# solver residual, not physics).
#
# Kept as two dicts anyway. They are separately solved quantities and the day
# something reintroduces a steady-state offset -- a payload the controller is
# not told about, friction compensation that does not fully cancel -- the two
# will separate again, and it should show up as a diff rather than as a pose
# that quietly stops landing where it says.
#
# For the record, since it explains the shape of everything downstream: on the
# floor mount the sag was nearly pure -z and happened to drop the tool onto the
# cube, so an uncompensated home still worked. Plate-mounted, the arm reaches
# out roughly LEVEL and the same sag was mostly -x, pulling the tool AWAY from
# the cube -- which ate the whole action budget and made the grasp literally
# unreachable (job 39502: grasp reward identically 0.0 for 4000 iterations).
#
# RE-SOLVED with gravity compensation on, over the 50 mm foam (cube centre at
# 425 mm). Emitted by `cubegrasp-solve-poses 0.46`, which preserves the hover as
# a CLEARANCE above the cube (+67.7 mm) rather than an absolute height -- the two
# are only the same while the work surface stays put.
#
# TABLE_X STAYS AT 0.46, re-checked rather than assumed: aim_camera puts the
# nearest cube corner at 205.3 mm against the D435's 200 mm gate and the worst
# frame coordinate over the whole jitter box at 0.646 of half-frame, so 0.00% of
# the 22869-pose envelope is blind or out of frame. That 5 mm of range margin is
# the tightest constraint in the scene and it is what pins TABLE_X -- the axis
# would centre the cube way in at 0.3777, and going there costs the margin.
#
# Settled error 0.00 mm (hover) / 0.17 mm (pre-grasp); tightest ctrlrange margin
# +0.409 rad on joint4. No joint limit needed widening -- reaching FURTHER OUT
# relaxes joint4, because it is the elbow fold that runs out of ctrlrange, and
# reaching closer in folds it harder, so the reach constraint binds at the NEAR
# edge of the spawn box rather than the far one.
#
# joint1 is PINNED to exactly 0 rather than the solver's +0.0026 (hover) /
# +0.0013 (pre-grasp), same as before: 1.3 mm of lateral offset, and keeping it
# at zero preserves the mirror symmetry the cube-yaw DR assumes.
# --- arm poses --------------------------------------------------------------
# Solved through the PHYSICS by cubegrasp-solve-poses: kinematic IK for the pose,
# then the static equilibrium map inverted by fixed-point iteration. Not by hand
# and not by kinematics alone. With gravcomp on, that inversion now converges in
# essentially one step -- which is the point, not a shortcut to remove.
#
# These are specific to this claw. The pad frame moves with the gripper, so a
# hand change invalidates all three: re-run the solver and paste its output.
# It REJECTS a configuration whose settled error or joint-limit margin is too
# small rather than reporting a kinematic residual over a silently clipped
# target.
_HOME_TARGETS = {
  "joint1": 0.0,
  "joint2": -0.3685,
  "joint3": 0.2170,
  "joint4": 2.3382,
  "joint5": -0.1801,
  "joint6": 1.1223,
  "joint7": 1.1535,
}
_SETTLED = {
  "joint1": 0.0,
  "joint2": -0.3691,
  "joint3": 0.2170,
  "joint4": 2.3382,
  "joint5": -0.1801,
  "joint6": 1.1223,
  "joint7": 1.1535,
}
_PREGRASP = {
  "joint1": 0.0,
  "joint2": -0.5358,
  "joint3": 0.1696,
  "joint4": 2.3659,
  "joint5": -0.3405,
  "joint6": 1.3099,
  "joint7": 1.2628,
}


# Where the arm actually COMES TO REST under those targets. Episodes reset to
# this so the arm starts at equilibrium instead of starting 7 cm high and
# sagging through the first ~0.5 s of every episode.
TWOFINGER_ARM_SETTLED: dict[str, float] = {
  **_SETTLED,
  **FINGER_OPEN_POS,
}

# PRE-GRASP pose: pads straddling the cube at its MID-HEIGHT (settled pad mid =
# (TABLE_X, 0.00, RESTING_Z) exactly), fingers vertical, grip axis world y. The
# figures here used to be written out as (0.55, 0.00, 0.375), which was stale
# text from before TABLE_X moved to 0.41 -- the POSE tracked both moves
# correctly, only the comment did not. Naming the constants instead so it cannot
# drift again.
# Solved through the physics like the home pose, so these are resting angles.
#
# Used by the demonstration reset (events.reset_robot_home easy_fraction): a
# fraction of episodes START here instead of at the hover, so the policy
# actually experiences grasp/lift/hold. Those three rewards were never once
# sampled in ~20k iterations of exploration from the hover pose -- the scripted
# check proves the grasp is physically achievable (17.2 cm lift at 11-21 N), so
# what was missing was any experience of it, not any further reward reshaping.
TWOFINGER_ARM_PREGRASP: dict[str, float] = {
  **_PREGRASP,
  **FINGER_OPEN_POS,
}

TWOFINGER_ARM_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, ARM_BASE_Z),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={**_HOME_TARGETS, **FINGER_OPEN_POS},
  joint_vel={".*": 0.0},
)

# 7 Menagerie arm position servos + 4 finger position servos, as TWO groups
# rather than one. Physically identical -- an XmlActuatorCfg only says which
# already-existing XML actuators mjlab drives, and the action term selects by
# actuator NAME, so nothing about the model, the control ordering or the action
# dimension changes. The split exists so `dr.pd_gains` can address them
# separately: it selects whole actuator GROUPS (`asset.actuators[i]`), so with a
# single group the arm and the fingers are forced to share one gain band.
#
# They should not share one. The arm's position servos come from Menagerie and
# are at least a published fit; `_FINGER_KP`/`_FINGER_KV` are estimates for a
# DC15-A01-class servo, and kv alone is the difference between holding 61% and
# 100% of forced-closure rollouts (see the note above `_FINGER_KV`). The most
# uncertain gain in the model deserves the widest band, and that is only
# expressible once the groups are separate.
TWOFINGER_ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(target_names_expr=(r"joint[1-7]",)),
    XmlActuatorCfg(target_names_expr=(r"(left|right)_[12]_tf",)),
  ),
  soft_joint_pos_limit_factor=0.9,
)
# Indices into TWOFINGER_ARM_ARTICULATION.actuators, for SceneEntityCfg's
# `actuator_ids` (which pd_gains uses to pick groups, NOT to pick joints).
ARM_ACTUATOR_GROUP = 0
FINGER_ACTUATOR_GROUP = 1


def get_robot_cfg(collision: str = "boxes") -> EntityCfg:
  """Sys-id'd Rizon 4S + two-finger hand as one loadable mjlab EntityCfg."""
  return EntityCfg(
    init_state=TWOFINGER_ARM_HOME,
    spec_fn=lambda: twofinger_arm_spec(collision),
    articulation=TWOFINGER_ARM_ARTICULATION,
  )


if __name__ == "__main__":
  m = twofinger_arm_spec().compile()
  print(
    f"compiled OK: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
    f"ngeom={m.ngeom} nsensor={m.nsensor}"
  )
  print(
    "actuators:",
    [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)],
  )
  print(
    "sensors:",
    [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)],
  )
