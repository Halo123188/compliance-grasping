"""
Rizon 4S arm + mygripper_H100_R ("elephant") hand — combined viewer with
interactive mount tuning.

The gripper's `base_step` root link is attached to link7 (flange) of the Rizon 4S
via MjSpec.attach() (same combination-script pattern as allegro_arm_cfg.py /
arm_hand_viewer.py). 22 DOF total: 7 arm joints + 15 gripper joints.

## Mount Adjustment
Open the **Actuation** tab → scroll to the bottom → **"Mount Adjustment"** folder.
Six sliders translate (pos x/y/z, metres) and rotate (rot x/y/z, degrees) the
gripper base in link7's local frame in real time. Click "Print mount params" to
get body_pos/body_quat values to paste into build_model() to make the pose
permanent (then mirror them into a get_mygripper_arm_cfg() like allegro_arm_cfg).

## Variants (flags)
- (none)    : 15 independent position servos — free posing / mount tuning
- --coupled : four-bar loops closed via joint-mimic (cubic mjEQ_JOINT polycoef)
- --weld    : four-bar loops closed via rigid mjEQ_WELD (physically correct)
- --port N  : bind the viser server to a specific port

## Notes
- The hand URDF (onshape export) ships no actuators; hand joints get position servos.
  The viewer starts paused — adjust the mount, then unpause to test.
- finger1_step.stl was decimated 719k->108k faces (MuJoCo's 200k-face limit).
- Needs the Rizon 4S arm from MuJoCo Menagerie (set MJ_MENAGERIE; see README).

Run (web viewer, no display needed; tunnel the printed port):
    python mygripper_arm_viewer.py --weld
Quick headless check (build + DOF, no server):
    python mygripper_arm_viewer.py --check --weld
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np

# The Rizon 4S arm comes from MuJoCo Menagerie. Override with the MJ_MENAGERIE
# env var; defaults to ~/mujoco_menagerie.
MENAGERIE = Path(os.environ.get("MJ_MENAGERIE", str(Path.home() / "mujoco_menagerie")))
ARM_XML = MENAGERIE / "flexiv_rizon4s" / "flexiv_rizon4s.xml"
HAND_URDF = (
    Path(__file__).resolve().parent
    / "assets/robots/mygripper_H100_R/robot.urdf"
)

# Name of the attached gripper root body after MjSpec.attach(suffix="_h").
MOUNT_BODY = "/base_step_h"

# Tuned gripper mount pose on the arm flange (body_pos/quat of MOUNT_BODY),
# recovered from interactive Mount-Adjustment tuning (euler deltas rx=-55 ry=0 rz=-85).
# The viewer's Mount-Adjustment sliders start from this pose, so it stays fine-tunable.
MOUNT_POS = np.array([-0.01000, -0.00010, 0.12100])
MOUNT_QUAT = np.array([0.65397, -0.34044, -0.31195, -0.59926])  # wxyz, normalized


def _bake_mount(mjm: mujoco.MjModel) -> None:
    """Apply the tuned gripper mount pose onto the attached root body, in-place."""
    mid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    if mid >= 0:
        mjm.body_pos[mid] = MOUNT_POS
        mjm.body_quat[mid] = MOUNT_QUAT

# ---------------------------------------------------------------------------
# Four-bar LOOP CLOSURE (coupled gripper) -- see build_model_coupled() below.
#
# The mygripper_H100_R is 3 four-bar linkages cut open into a URDF tree (each
# finger's "coupler" link is duplicated = the loop cut). Physically each finger
# is a single 1-DOF mechanism, but the open URDF exposes 15 free joints.
#
# We close each loop in JOINT space with mjEQ_JOINT polycoef couplings (NOT a
# 6-DOF weld: the planar 4-bar has parallel revolute axes, so a weld is
# redundant and the solver blows up -- confirmed empirically). Each finger has
# one INPUT loop joint (nRevolute 5/8/11) and 3 PASSIVE loop joints; each passive
# joint follows a cubic of the input: passive = c0 + c1*u + c2*u^2 + c3*u^3.
# Coefficients were fit by sweeping the input joint and solving the loop so the
# duplicate coupler bodies coincide (kinematic closure error <0.5 mm). The base
# joint (Revolute 1/2/3) is an independent proximal swing DOF (not in the loop).
#
# Per finger: (base joint, input loop joint, {passive joint: [c0,c1,c2,c3]})
# Names are the hand-alone URDF names; build_model_coupled adds these to the hand
# spec BEFORE attach(), so attach()'s "_h" suffix retargets them automatically.
COUPLED_FINGERS = [
    {
        "base": "Revolute 1", "input": "nRevolute 5",
        "passive": {
            "Revolute 4":     [-4e-05,    0.59914, -0.01565,  0.02358],
            "nCylindrical 1": [-9e-05,    0.44785, -0.25694, -0.01842],
            "nRevolute 6":    [-0.00013,  0.04698, -0.27258,  0.00517],
        },
    },
    {
        "base": "Revolute 2", "input": "nRevolute 8",
        "passive": {
            "Revolute 7":     [-0.00326,  0.62842,  0.02157, -0.21583],
            "nCylindrical 2": [ 0.0072,   0.82657,  0.21622,  0.35956],
            "nRevolute 9":    [ 0.00393,  0.45499,  0.2378,   0.14373],
        },
    },
    {
        "base": "Revolute 3", "input": "nRevolute 11",
        "passive": {
            "Revolute 10":    [-0.0001,   0.55013, -0.04187,  0.005],
            "nCylindrical 3": [ 0.0,     -0.62893,  0.29248,  0.00264],
            "nRevolute 12":   [-0.00011,  0.17906, -0.33435,  0.00235],
        },
    },
]
# The 6 driven DOFs of the coupled gripper (2 per finger): proximal swing + curl.
COUPLED_DRIVEN = [j for f in COUPLED_FINGERS for j in (f["base"], f["input"])]
# Passive loop joints are constraint-slaved; they get no actuator.
COUPLED_PASSIVE = [j for f in COUPLED_FINGERS for j in f["passive"]]


# ---------------------------------------------------------------------------
# Four-bar loop closure via rigid WELD (build_model_welded(), --weld flag).
#
# The MIMIC approach (build_model_coupled) constrains the loop in JOINT space, but
# it never physically joins the two duplicate coupler bodies -> they visibly
# SEPARATE under motion ("two mounts appear"; relpose drift up to ~1.95 mm). The
# PHYSICALLY-CORRECT closure fuses each finger's duplicate coupler body into one
# with a 6-DOF WELD equality constraint. The earlier weld attempt blew up to NaN
# because the linkage links are 1-30 g with near-singular inertia tensors -> an
# ill-conditioned mass matrix under a stiff weld.
#
# The fix (validated headless, scripts/proto_weld_*.py): ARMATURE on the gripper
# joints (reflected inertia) is the robust regularizer -- it stabilises the weld
# with ZERO body mass/inertia inflation (fully physical bodies). A stiff weld
# solref/solimp + Newton solver + seeding qpos0 at the assembly pose holds the two
# coupler bodies aligned to <0.2 mm anchor distance across the full joint range
# (vs ~1.5-1.95 mm drift for mimic). See WELD_CLOSURE_PROGRESS.md.
#
# Weld pairs (hand-alone names; body1 <- body2) + relpose (body2 origin in body1):
WELD_FINGERS = [
    {"name": "F1", "b1": "s_link2_step", "b2": "s_link2_step_2",
     "base": "Revolute 1", "input": "nRevolute 5", "tip": "s_link2_step_2",
     "passive": ["Revolute 4", "nCylindrical 1", "nRevolute 6"],
     "relpos": [0.035707, 0.012457, 0.009],
     "relquat": [0.924902, 0.0, 0.0, 0.380206]},
    {"name": "F2", "b1": "finger3_step", "b2": "finger3_step_2",
     "base": "Revolute 2", "input": "nRevolute 8", "tip": "finger3_step_2",
     "passive": ["Revolute 7", "nCylindrical 2", "nRevolute 9"],
     "relpos": [0.03668, -0.011137, 0.0088],
     "relquat": [0.947288, 0.0, 0.0, -0.320384]},
    {"name": "F3", "b1": "finger4_step", "b2": "finger4_step_2",
     "base": "Revolute 3", "input": "nRevolute 11", "tip": "finger4_step_2",
     "passive": ["Revolute 10", "nCylindrical 3", "nRevolute 12"],
     "relpos": [0.03668, 0.011137, 0.0089],
     "relquat": [0.972724, 0.0, 0.0, 0.231967]},
]
# 6 driven DOFs (base swing + input curl per finger); the loop is closed by the weld
# so the other 3 loop joints per finger are determined by physics (no actuator).
WELD_DRIVEN = [j for f in WELD_FINGERS for j in (f["base"], f["input"])]

# Assembly loop-joint angles (rad) that close each loop to ~0 mm; qpos0 is seeded
# with these so the welds start satisfied (avoids the ~39 mm reset snap -> NaN).
WELD_ASSEMBLY = {
    "Revolute 4": 0.11850651675287728, "nRevolute 5": 0.19694636697490253,
    "nCylindrical 1": 0.0774500849888125, "nRevolute 6": -0.0009871188288751393,
    "Revolute 7": 0.1831730332809386, "nRevolute 8": 0.32215584872455566,
    "nCylindrical 2": 0.3312176032729121, "nRevolute 9": 0.19223044142564735,
    "Revolute 10": -0.0019317300325050431, "nRevolute 11": -0.0034647938206841134,
    "nCylindrical 3": 0.0021822067379099544, "nRevolute 12": -0.0006524893470445766,
}

# Stability config (smallest physical regularization that holds the weld; see progress).
WELD_ARMATURE = 0.01       # reflected inertia at the gripper joints -- the real fix
WELD_DAMPING = 0.3
WELD_MASS_FLOOR = 0.0      # no mass inflation (fully physical)
WELD_INERTIA_FLOOR = 0.0   # no inertia inflation (armature alone stabilises)
WELD_SOLREF = [0.002, 1.0]   # stiff weld; tames transient stretch under fast slew
WELD_SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]
WELD_TIMESTEP = 0.001
WELD_ITERATIONS = 150
WELD_LS_ITERATIONS = 50

# Linkage bodies that get optional inertia/mass flooring (hand-alone names).
WELD_LINKAGE_BODIES = [
    "m_link_step", "s_link2_step", "s_link1_step", "finger1_step", "s_link2_step_2",
    "dae_mhl_step", "finger3_step", "finger2_step", "finger1_step_2", "finger3_step_2",
    "dae_mhr_step", "finger4_step", "finger2_step_2", "finger1_step_3", "finger4_step_2",
]


# ---------------------------------------------------------------------------
# PER-JOINT RANGE-OF-MOTION LIMITS (mygripper_H100_R "elephant" hand).
#
# The URDF ships placeholder +/-pi limits; no datasheet publishes per-joint angles.
# These ranges were DERIVED MECHANICALLY in the WELDED sim (the physically-correct
# variant) and anchored to the only 2 REAL specs: gripping aperture 0-130 mm and
# per-joint velocity 60 deg/s (= 1.047 rad/s). See JOINT_LIMITS_PROGRESS.md and
# scripts/derive_joint_limits.py for the full sim evidence.
#
# MECHANISM (sim finding, contradicts the naive "curl = aperture" guess):
#   * The BASE SWING joint (Revolute 1/2/3) is the dominant APERTURE DOF: swinging the
#     3 proximal joints together opens/closes the 3-jaw. The weld stays dead-rigid
#     (0.001 mm) across the whole base sweep (base is outside the four-bar loop).
#       - aperture proxy = circumscribed-circle diameter through the 3 fingertip pad
#         CENTERS, minus the closed-pose baseline (87.3 mm, where the caging circle is
#         smallest = an enclosed object of ~0 mm). Documented + used consistently.
#       - base delta = -0.70 rad (-40 deg): aperture peaks at ~129 mm == fully OPEN (130 mm).
#       - base delta = +0.90 rad (+52 deg): aperture == 0 mm == fully CLOSED (pads meet).
#       - bounded by APERTURE GEOMETRY (curve turns over at both ends), NOT self-collision
#         (the 3 fingers sit ~120 deg apart and never touch each other in this range).
#   * The CURL joint (nRevolute 5/8/11) is the secondary WRAP/conform DOF. Its range is
#     bounded by the FOUR-BAR SINGULARITY (dead-center: weld alignment spikes >1 mm /
#     qpos goes non-finite past these angles), with a ~0.05 rad safety margin inside.
#       - F1 nRevolute 5 : feasible [-1.592,+1.934] -> set [-1.54,+1.88]
#       - F2 nRevolute 8 : feasible [-1.658,+1.247] -> set [-1.61,+1.20] (F2 locks earliest)
#       - F3 nRevolute 11: feasible [-1.593,+1.932] -> set [-1.54,+1.88]
#   * The 9 PASSIVE joints are weld-slaved; their ranges are the min/max they REACH over
#     the full feasible (base x curl) envelope + a 0.1 rad margin so they never fight the
#     weld. Setting them does NOT destabilise the weld (validated).
#
# VELOCITY: 60 deg/s = 1.0472 rad/s per joint (the real spec). MuJoCo has no hard jnt
# velocity limit; it's enforced via actuator/damping. Recorded here as JOINT_VEL_LIMIT;
# build_model_welded/coupled keep the existing damped position servos (which already
# rate-limit slew), and the value is available for a future velocity-limited actuator
# or mjlab ActuatorCfg.velocity_limit.
JOINT_VEL_LIMIT = 1.0472  # rad/s (= 60 deg/s)

JOINT_LIMITS = {
    # --- 6 DRIVEN joints (ENFORCED: joint range + actuator ctrlrange) ---
    # base swing (aperture: open @ -0.70, closed @ +0.90; symmetric per finger)
    "Revolute 1":     (-0.70, 0.90),   # F1 base  -- aperture
    "Revolute 2":     (-0.70, 0.90),   # F2 base  -- aperture
    "Revolute 3":     (-0.70, 0.90),   # F3 base  -- aperture
    # input curl (wrap; bounded by four-bar singularity, per finger)
    "nRevolute 5":    (-1.54, 1.88),   # F1 curl  -- singularity
    "nRevolute 8":    (-1.61, 1.20),   # F2 curl  -- singularity (locks earliest)
    "nRevolute 11":   (-1.54, 1.88),   # F3 curl  -- singularity
    # --- 9 PASSIVE joints (weld-slaved; swept ROM + 0.1 rad margin) ---
    "Revolute 4":     (-0.85, 1.18),   # F1 passive (swept [-0.75,+1.08])
    "nCylindrical 1": (-1.06, 0.28),   # F1 passive (swept [-0.96,+0.18])
    "nRevolute 6":    (-0.73, 0.10),   # F1 passive (swept [-0.63,+0.00])
    "Revolute 7":     (-0.83, 0.54),   # F2 passive (swept [-0.73,+0.44])
    "nCylindrical 2": (-0.83, 1.25),   # F2 passive (swept [-0.73,+1.15])
    "nRevolute 9":    (-0.30, 0.79),   # F2 passive (swept [-0.20,+0.69])
    "Revolute 10":    (-0.76, 0.96),   # F3 passive (swept [-0.66,+0.86])
    "nCylindrical 3": (-0.43, 1.27),   # F3 passive (swept [-0.33,+1.17])
    "nRevolute 12":   (-0.74, 0.12),   # F3 passive (swept [-0.64,+0.02])
}


def _apply_joint_limits(mjm, driven_joints):
    """Set the DERIVED per-joint ROM limits post-compile (additive). For every joint in
    JOINT_LIMITS, set jnt_range + jnt_limited; for the driven joints, also clamp the
    matching actuator ctrlrange to the same range. Names carry the attach '_h' suffix.
    Returns the count applied (for validation)."""
    n_jnt, n_act = 0, 0
    driven = set(driven_joints)
    for jn, (lo, hi) in JOINT_LIMITS.items():
        ji = _attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if ji >= 0:
            mjm.jnt_range[ji] = [lo, hi]
            mjm.jnt_limited[ji] = 1
            n_jnt += 1
        if jn in driven:
            ai = _attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            if ai >= 0:
                mjm.actuator_ctrlrange[ai] = [lo, hi]
                mjm.actuator_ctrllimited[ai] = 1
                n_act += 1
    return n_jnt, n_act


def _attached_id(mjm, objtype, hand_name):
    """Resolve a hand-alone name to its id in the combined model. MjSpec.attach()
    renames the hand's objects to '/<name>_h'; fall back to the bare name so the
    same helpers work on a hand-alone spec too."""
    for cand in (f"/{hand_name}_h", hand_name):
        i = mujoco.mj_name2id(mjm, objtype, cand)
        if i >= 0:
            return i
    return -1


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    wbody = arm.worldbody

    floor = wbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.pos = [0, 0, 0]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]

    light = wbody.add_light()
    light.pos = [0, 0, 2]
    light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    obj = wbody.add_body()
    obj.name = "object"
    obj.pos = [0.5, 0, 0.05]
    fj = obj.add_freejoint()
    fj.name = "object_joint"
    g = obj.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.03, 0, 0]
    g.rgba = [0.8, 0.4, 0.1, 1.0]
    g.condim = 6
    g.friction = [0.7, 0.002, 0.002]

    # Mount site at the link7 flange tip.
    for b in arm.bodies:
        if b.name == "link7":
            site = b.add_site()
            site.name = "hand_mount"
            site.pos = [0.0, 0.0, 0.095]
            site.quat = [1.0, 0.0, 0.0, 0.0]
            break

    hand = mujoco.MjSpec.from_file(str(HAND_URDF))

    # The onshape URDF ships no actuators. Add a position servo for every gripper
    # joint so all 15 hand DOFs are individually controllable in the viewer's
    # Actuation tab. attach() suffixes the names + retargets the transmissions to
    # the "_h" joints automatically.
    KP, KV = 1.0, 0.1
    for j in hand.joints:
        a = hand.add_actuator()
        a.name = j.name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = j.name
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        gp = np.zeros(mujoco.mjNGAIN)
        gp[0] = KP
        a.gainprm = gp
        bp = np.zeros(mujoco.mjNBIAS)
        bp[1] = -KP
        bp[2] = -KV
        a.biasprm = bp
        a.ctrlrange = np.array([-np.pi, np.pi])

    arm.attach(hand, suffix="_h", site="hand_mount")

    # Drop the arm's 'home' keyframe (7 ctrl values) — stale now that the model
    # has 22 DOF; mjlab/this viewer set the pose explicitly instead.
    for k in list(arm.keys):
        arm.delete(k)

    mjm = arm.compile()
    _bake_mount(mjm)

    # The URDF ships no joint damping; give the gripper joints (suffix "_h")
    # light damping/armature post-compile so the fingers don't collapse under
    # gravity when the sim is unpaused.
    for i in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.endswith("_h"):
            adr = mjm.jnt_dofadr[i]
            mjm.dof_damping[adr] = 0.1
            mjm.dof_armature[adr] = 0.001

    mjd = mujoco.MjData(mjm)

    # Arm pre-grasp pose (gripper hovering forward); mount pose baked by
    # _bake_mount() above — the Mount-Adjustment sliders fine-tune from there.
    mjd.qpos[:7] = [0, 0, 0, 1.57, 0, 0, 0]
    mjd.ctrl[:7] = [0, 0, 0, 1.57, 0, 0, 0]
    mujoco.mj_forward(mjm, mjd)
    return mjm, mjd


def build_model_coupled() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Like build_model() but with the 3 four-bar loops CLOSED via mjEQ_JOINT
    couplings, so the gripper has 6 driven DOFs (2 per finger) instead of 15
    free joints. Additive / non-breaking: build_model() is untouched.

    Differences vs build_model():
      * actuators only on the 6 driven joints (base swing + input curl per finger);
        passive loop joints are slaved by equality constraints.
      * 9 mjEQ_JOINT polycoef equality constraints (3 per finger) close the loops.
      * linkage geoms get contype/conaffinity=0 (the 4-bar links overlap as a
        mechanism -> self-collision is non-physical and destabilises the solver).
      * extra solver iterations + per-joint damping/armature for a stable loop.
    """
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    wbody = arm.worldbody

    floor = wbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.pos = [0, 0, 0]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]

    light = wbody.add_light()
    light.pos = [0, 0, 2]
    light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    obj = wbody.add_body()
    obj.name = "object"
    obj.pos = [0.5, 0, 0.05]
    fj = obj.add_freejoint()
    fj.name = "object_joint"
    g = obj.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.03, 0, 0]
    g.rgba = [0.8, 0.4, 0.1, 1.0]
    g.condim = 6
    g.friction = [0.7, 0.002, 0.002]

    for b in arm.bodies:
        if b.name == "link7":
            site = b.add_site()
            site.name = "hand_mount"
            site.pos = [0.0, 0.0, 0.095]
            site.quat = [1.0, 0.0, 0.0, 0.0]
            break

    hand = mujoco.MjSpec.from_file(str(HAND_URDF))

    # Position servos ONLY on the 6 driven joints (base swing + input curl).
    KP, KV = 8.0, 0.4
    for jn in COUPLED_DRIVEN:
        a = hand.add_actuator()
        a.name = jn
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        gp = np.zeros(mujoco.mjNGAIN)
        gp[0] = KP
        a.gainprm = gp
        bp = np.zeros(mujoco.mjNBIAS)
        bp[1] = -KP
        bp[2] = -KV
        a.biasprm = bp
        a.ctrlrange = np.array([-np.pi, np.pi])

    # mjEQ_JOINT loop closures: passive = cubic(input).
    # name1 = DEPENDENT (passive) joint, name2 = INDEPENDENT (input) joint.
    for f in COUPLED_FINGERS:
        for jn, coef in f["passive"].items():
            eq = hand.add_equality()
            eq.name = f"loop_{jn}".replace(" ", "_")
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
            eq.name1 = jn
            eq.name2 = f["input"]
            data = np.zeros(11)
            data[:4] = coef          # [c0, c1, c2, c3]; c4 = 0
            data[10] = 1.0
            eq.data = data
            eq.solref = np.array([0.01, 1.0])
            eq.solimp = np.array([0.95, 0.99, 0.001, 0.5, 2.0])

    # The four-bar links physically overlap as a mechanism: disable self-collision.
    # contype=0 prevents any geom from initiating contact (self-collision disabled)
    # while keeping conaffinity intact so MuJoCo's compiler retains the geoms for
    # rendering. Setting both to 0 causes the compiler to silently drop all geoms.
    for geom in hand.geoms:
        geom.contype = 0

    arm.attach(hand, suffix="_h", site="hand_mount")

    for k in list(arm.keys):
        arm.delete(k)

    arm.option.iterations = 100
    arm.option.ls_iterations = 50

    mjm = arm.compile()
    _bake_mount(mjm)

    # Damping + armature on the gripper joints (suffix "_h"): the URDF ships none,
    # and the massless slaved loop links need it for a well-conditioned solver.
    for i in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.endswith("_h"):
            adr = mjm.jnt_dofadr[i]
            mjm.dof_damping[adr] = 0.2
            mjm.dof_armature[adr] = 0.002

    # Derived per-joint ROM limits (joint range + driven actuator ctrlrange). Additive.
    _apply_joint_limits(mjm, COUPLED_DRIVEN)

    mjd = mujoco.MjData(mjm)

    mjd.qpos[:7] = [0, 0, 0, 1.57, 0, 0, 0]
    mjd.ctrl[:7] = [0, 0, 0, 1.57, 0, 0, 0]
    mujoco.mj_forward(mjm, mjd)
    return mjm, mjd


def build_model_welded() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Like build_model() but with the 3 four-bar loops closed by PHYSICALLY-CORRECT
    rigid WELD constraints (each finger's duplicate coupler body fused into one),
    made numerically stable via armature regularization. Additive / non-breaking:
    build_model() and build_model_coupled() are untouched.

    Differences vs build_model():
      * actuators only on the 6 driven joints (base swing + input curl per finger);
        the weld closes each loop so the other 3 loop joints are determined by physics.
      * 3 mjEQ_WELD equality constraints (one per finger) fuse the duplicate couplers.
      * armature (WELD_ARMATURE) + damping (WELD_DAMPING) on the gripper joints --
        the real regularizer that stabilises the weld over the tiny linkage inertias.
      * optional mass/inertia floors (WELD_MASS_FLOOR/WELD_INERTIA_FLOOR; default 0,
        i.e. fully physical -- armature alone is sufficient).
      * Newton solver, stiff weld solref/solimp, dt=WELD_TIMESTEP.
      * linkage geoms get contype/conaffinity=0 (the 4-bar links overlap as a mechanism).
      * qpos0 seeded at the assembly pose so the welds start satisfied (no reset snap).
    """
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    wbody = arm.worldbody

    floor = wbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.pos = [0, 0, 0]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]

    light = wbody.add_light()
    light.pos = [0, 0, 2]
    light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    obj = wbody.add_body()
    obj.name = "object"
    obj.pos = [0.5, 0, 0.05]
    fj = obj.add_freejoint()
    fj.name = "object_joint"
    g = obj.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.03, 0, 0]
    g.rgba = [0.8, 0.4, 0.1, 1.0]
    g.condim = 6
    g.friction = [0.7, 0.002, 0.002]

    for b in arm.bodies:
        if b.name == "link7":
            site = b.add_site()
            site.name = "hand_mount"
            site.pos = [0.0, 0.0, 0.095]
            site.quat = [1.0, 0.0, 0.0, 0.0]
            break

    hand = mujoco.MjSpec.from_file(str(HAND_URDF))

    # Position servos ONLY on the 6 driven joints (base swing + input curl).
    KP, KV = 8.0, 0.4
    for jn in WELD_DRIVEN:
        a = hand.add_actuator()
        a.name = jn
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        gp = np.zeros(mujoco.mjNGAIN)
        gp[0] = KP
        a.gainprm = gp
        bp = np.zeros(mujoco.mjNBIAS)
        bp[1] = -KP
        bp[2] = -KV
        a.biasprm = bp
        a.ctrlrange = np.array([-np.pi, np.pi])

    # 3 mjEQ_WELD constraints: fuse body2 (distal coupler) onto body1 (input coupler)
    # at the solved assembly relpose. Added to the HAND spec BEFORE attach() so the
    # "_h" suffix retargets the body refs automatically. Anchor = body2 origin (data
    # [:3]=0); data[3:6]=relpos (body2 origin in body1), data[6:10]=relquat (wxyz),
    # data[10]=torquescale.
    for f in WELD_FINGERS:
        eq = hand.add_equality()
        eq.name = f"weld_{f['name']}"
        eq.name1 = f["b1"]
        eq.name2 = f["b2"]
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.type = mujoco.mjtEq.mjEQ_WELD
        data = np.zeros(11)
        data[:3] = 0.0
        data[3:6] = f["relpos"]
        data[6:10] = f["relquat"]
        data[10] = 1.0
        eq.data = data
        eq.solref = np.array(WELD_SOLREF)
        eq.solimp = np.array(WELD_SOLIMP)

    # The four-bar links physically overlap as a mechanism: disable self-collision.
    # Keep conaffinity intact — setting both to 0 makes the compiler drop all geoms.
    for geom in hand.geoms:
        geom.contype = 0

    arm.attach(hand, suffix="_h", site="hand_mount")

    for k in list(arm.keys):
        arm.delete(k)

    arm.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    arm.option.iterations = WELD_ITERATIONS
    arm.option.ls_iterations = WELD_LS_ITERATIONS
    arm.option.timestep = WELD_TIMESTEP

    mjm = arm.compile()
    _bake_mount(mjm)

    # --- Stability regularization (post-compile; M is recomputed each step) ---
    # Armature + damping on every gripper joint (suffix "_h"): the URDF ships none,
    # and the tiny-inertia loop links need reflected inertia for a well-conditioned
    # mass matrix under the weld. This is the key fix.
    for i in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.endswith("_h"):
            adr = mjm.jnt_dofadr[i]
            mjm.dof_armature[adr] = WELD_ARMATURE
            mjm.dof_damping[adr] = WELD_DAMPING

    # Derived per-joint ROM limits (joint range + driven actuator ctrlrange). Additive;
    # the assembly qpos0 / 6 driven joints all start well inside these ranges. The 9
    # passive joints get swept-ROM + margin ranges that do NOT fight the weld (validated).
    _apply_joint_limits(mjm, WELD_DRIVEN)

    # Optional inertia/mass floors on the linkage bodies (default 0 = fully physical;
    # armature alone is sufficient, but the knobs are kept for robustness tuning).
    if WELD_MASS_FLOOR > 0 or WELD_INERTIA_FLOOR > 0:
        for bn in WELD_LINKAGE_BODIES:
            bi = _attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bi < 0:
                continue
            if WELD_MASS_FLOOR > 0 and mjm.body_mass[bi] < WELD_MASS_FLOOR:
                mjm.body_mass[bi] = WELD_MASS_FLOOR
            if WELD_INERTIA_FLOOR > 0:
                mjm.body_inertia[bi] = np.maximum(mjm.body_inertia[bi], WELD_INERTIA_FLOOR)

    mjd = mujoco.MjData(mjm)

    # Seed qpos0: arm pre-grasp + the assembly loop angles so the welds start satisfied.
    mjd.qpos[:7] = [0, 0, 0, 1.57, 0, 0, 0]
    mjd.ctrl[:7] = [0, 0, 0, 1.57, 0, 0, 0]
    for jn, val in WELD_ASSEMBLY.items():
        ji = _attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if ji >= 0:
            mjd.qpos[mjm.jnt_qposadr[ji]] = val
    # Hold the 6 driven joints at their assembly values so they don't snap on unpause.
    for jn in WELD_DRIVEN:
        ji = _attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        ai = _attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        if ji >= 0 and ai >= 0:
            mjd.ctrl[ai] = mjd.qpos[mjm.jnt_qposadr[ji]]
    mujoco.mj_forward(mjm, mjd)
    return mjm, mjd


def weld_alignment_mm(mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, float]:
    """Live weld alignment error (mm) per finger = world distance between the shared
    weld anchor point computed from body1 vs from body2. ~0 means the two coupler
    bodies are fused; large means they have separated. Used for the viewer readout
    and the diagnostic. Body names carry the attach "_h" suffix if present."""
    out = {}
    for f in WELD_FINGERS:
        b1 = _attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["b1"])
        b2 = _attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["b2"])
        if b1 < 0 or b2 < 0:
            continue
        R1 = mjd.xmat[b1].reshape(3, 3)
        p_from_b1 = mjd.xpos[b1] + R1 @ np.array(f["relpos"])
        p_from_b2 = mjd.xpos[b2]
        out[f["name"]] = float(np.linalg.norm(p_from_b1 - p_from_b2) * 1000.0)
    return out


def build_hand_only(closure: str = "weld") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Standalone gripper — the mygripper_H100_R URDF on its own (NO arm, NO
    MuJoCo Menagerie dependency), fixed at the world origin, with the 3 four-bar
    loops closed and position servos + joint limits on the 6 driven joints. Lets
    the hand asset be loaded/posed independently of the Rizon mount.

    closure: "weld" (rigid mjEQ_WELD, physically correct) or "coupled" (joint-mimic).
    Run:  python mygripper_arm_viewer.py --hand-only            (weld)
          python mygripper_arm_viewer.py --hand-only --coupled  (mimic)
          python mygripper_arm_viewer.py --check --hand-only
    """
    hand = mujoco.MjSpec.from_file(str(HAND_URDF))
    wbody = hand.worldbody

    floor = wbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.pos = [0, 0, -0.25]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]
    light = wbody.add_light()
    light.pos = [0, 0, 1.5]
    light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    # Position servos on the 6 driven joints (base swing + input curl per finger).
    KP, KV = 8.0, 0.4
    for jn in WELD_DRIVEN:
        a = hand.add_actuator()
        a.name = jn
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        gp = np.zeros(mujoco.mjNGAIN)
        gp[0] = KP
        a.gainprm = gp
        bp = np.zeros(mujoco.mjNBIAS)
        bp[1] = -KP
        bp[2] = -KV
        a.biasprm = bp
        a.ctrlrange = np.array([-np.pi, np.pi])

    if closure == "coupled":
        for f in COUPLED_FINGERS:
            for jn, coef in f["passive"].items():
                eq = hand.add_equality()
                eq.name = f"loop_{jn}".replace(" ", "_")
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
                eq.name1 = jn
                eq.name2 = f["input"]
                data = np.zeros(11)
                data[:4] = coef
                data[10] = 1.0
                eq.data = data
                eq.solref = np.array([0.01, 1.0])
                eq.solimp = np.array([0.95, 0.99, 0.001, 0.5, 2.0])
    else:  # weld
        for f in WELD_FINGERS:
            eq = hand.add_equality()
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
            eq.solref = np.array(WELD_SOLREF)
            eq.solimp = np.array(WELD_SOLIMP)
        for geom in hand.geoms:
            geom.contype = 0  # disable self-collision; keep conaffinity so compiler retains geoms
        hand.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        hand.option.iterations = WELD_ITERATIONS
        hand.option.ls_iterations = WELD_LS_ITERATIONS
        hand.option.timestep = WELD_TIMESTEP

    for k in list(hand.keys):
        hand.delete(k)

    mjm = hand.compile()

    # Damping + armature on every gripper joint (regularizes the loop closure).
    for i in range(mjm.njnt):
        adr = mjm.jnt_dofadr[i]
        mjm.dof_damping[adr] = WELD_DAMPING
        mjm.dof_armature[adr] = WELD_ARMATURE

    _apply_joint_limits(mjm, WELD_DRIVEN)

    mjd = mujoco.MjData(mjm)
    for jn, val in WELD_ASSEMBLY.items():
        ji = _attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if ji >= 0:
            mjd.qpos[mjm.jnt_qposadr[ji]] = val
    for jn in WELD_DRIVEN:
        ji = _attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        ai = _attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        if ji >= 0 and ai >= 0:
            mjd.ctrl[ai] = mjd.qpos[mjm.jnt_qposadr[ji]]
    mujoco.mj_forward(mjm, mjd)
    return mjm, mjd


def _print_stats(mjm: mujoco.MjModel, coupled: bool = False, welded: bool = False,
                 hand_only: bool = False) -> None:
    print("Rizon 4S + mygripper_H100_R (elephant) hand")
    if hand_only:
        print(f"  DOF       : {mjm.nq}  (standalone gripper — 15 joints, no arm)")
        print(f"  Actuators : {mjm.nu}  (6 driven gripper DOFs)")
        print(f"  Equalities: {mjm.neq}  (four-bar loop closures)")
        print(f"  Bodies    : {mjm.nbody}")
        return
    if welded:
        print(f"  DOF       : {mjm.nq}  (7 arm + 15 gripper joints + 6 object freejoint)")
        print(f"  Actuators : {mjm.nu}  (7 arm + 6 driven gripper DOFs; loop closed by weld)")
        print(f"  Equalities: {mjm.neq}  (3 mjEQ_WELD four-bar loop closures)")
        print(f"  Stability : armature={WELD_ARMATURE} damping={WELD_DAMPING} "
              f"mass_floor={WELD_MASS_FLOOR} inertia_floor={WELD_INERTIA_FLOOR}")
        print(f"  Solver    : Newton iters={WELD_ITERATIONS} ls={WELD_LS_ITERATIONS} "
              f"dt={WELD_TIMESTEP}  weld solref={WELD_SOLREF} solimp={WELD_SOLIMP}")
    elif coupled:
        print(f"  DOF       : {mjm.nq}  (7 arm + 15 gripper joints + 6 object freejoint)")
        print(f"  Actuators : {mjm.nu}  (7 arm + 6 driven gripper DOFs; 9 loop joints slaved)")
        print(f"  Equalities: {mjm.neq}  (9 mjEQ_JOINT four-bar loop closures)")
    else:
        print(f"  DOF       : {mjm.nq}  (7 arm + 15 gripper + 6 object freejoint)")
        print(f"  Actuators : {mjm.nu}  (7 arm + 15 gripper position servos)")
    print(f"  Bodies    : {mjm.nbody}")
    mid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    print(f"  Mount body: {MOUNT_BODY!r} (id={mid})")


def main() -> None:
    welded = "--weld" in sys.argv
    coupled = "--coupled" in sys.argv
    hand_only = "--hand-only" in sys.argv
    if hand_only:
        closure = "coupled" if coupled else "weld"
        builder = lambda: build_hand_only(closure)
        welded = (closure == "weld")  # drives the live weld-alignment panel + stats
    elif welded:
        builder = build_model_welded
    elif coupled:
        builder = build_model_coupled
    else:
        builder = build_model
    port_arg = None
    if "--port" in sys.argv:
        port_arg = int(sys.argv[sys.argv.index("--port") + 1])

    if "--check" in sys.argv:
        mjm, _ = builder()
        _print_stats(mjm, coupled=coupled, welded=welded, hand_only=hand_only)
        print("OK (build/compile only)")
        return

    import mjviser  # heavy import; only needed for the live server

    mjm, mjd = builder()
    _print_stats(mjm, coupled=coupled, welded=welded, hand_only=hand_only)

    class MyGripperViewer(mjviser.Viewer):
        def _setup_gui(self) -> None:
            super()._setup_gui()
            self._setup_mount_panel()
            self._weld_texts = None
            if welded:
                self._setup_weld_panel()

        def _setup_weld_panel(self) -> None:
            """Live weld alignment readout: world distance (mm) between each finger's
            two coupler bodies' shared weld anchor. Should stay ~0 while dragging the
            driven joints -- proof the weld physically holds the loop closed."""
            gui = self._server.gui
            with gui.add_folder("Weld Alignment (live) — should stay ~0 mm",
                                expand_by_default=True):
                gui.add_markdown(
                    "_World distance between each finger's two fused coupler bodies. "
                    "Drag the driven joints in the Actuation tab; these should stay "
                    "sub-0.2 mm (mimic drifts ~1.5-4 mm)._"
                )
                texts = {
                    f["name"]: gui.add_text(f["name"], "0.000 mm", disabled=True)
                    for f in WELD_FINGERS
                }
            self._weld_texts = texts

        def _tick(self) -> None:
            super()._tick()
            # Live alignment readout (throttled to render cadence to stay cheap).
            if self._weld_texts is not None:
                al = weld_alignment_mm(self.model, self.data)
                for name, t in self._weld_texts.items():
                    if name in al:
                        t.value = f"{al[name]:.3f} mm"

        def _setup_mount_panel(self) -> None:
            mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
            if mid < 0:
                return
            base_pos = self.model.body_pos[mid].copy()
            base_quat = self.model.body_quat[mid].copy()
            gui = self._server.gui
            with gui.add_folder("Mount Adjustment — gripper on flange", expand_by_default=True):
                gui.add_markdown(
                    "_Translate / rotate the gripper base in link7's local frame. "
                    "Sliders update live. Click **Print** to bake the values into "
                    "build\\_model()._"
                )
                s_px = gui.add_slider("pos x (m)", -0.15, 0.15, 0.001, 0.0)
                s_py = gui.add_slider("pos y (m)", -0.15, 0.15, 0.001, 0.0)
                s_pz = gui.add_slider("pos z (m)", -0.05, 0.25, 0.001, 0.0)
                s_rx = gui.add_slider("rot x (deg)", -180, 180, 1, 0)
                s_ry = gui.add_slider("rot y (deg)", -180, 180, 1, 0)
                s_rz = gui.add_slider("rot z (deg)", -180, 180, 1, 0)
                btn = gui.add_button("Print mount params")

            def _apply(_=None) -> None:
                dp = np.array([s_px.value, s_py.value, s_pz.value])
                dq = np.zeros(4)
                mujoco.mju_euler2Quat(dq, np.deg2rad([s_rx.value, s_ry.value, s_rz.value]), "xyz")
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, base_quat, dq)
                with self._lock:
                    self.model.body_pos[mid] = base_pos + dp
                    self.model.body_quat[mid] = new_quat
                    mujoco.mj_forward(self.model, self.data)
                self._dirty = True

            for s in [s_px, s_py, s_pz, s_rx, s_ry, s_rz]:
                s.on_update(_apply)

            @btn.on_click
            def _print(_event) -> None:
                with self._lock:
                    fp = self.model.body_pos[mid].copy()
                    fq = self.model.body_quat[mid].copy()
                pstr = "[" + ", ".join(f"{v:.5f}" for v in fp) + "]"
                qstr = "[" + ", ".join(f"{v:.5f}" for v in fq) + "]"
                print("\n-- Mount params (paste into build_model after arm.compile()) --")
                print(f"  mid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, '{MOUNT_BODY}')")
                print(f"  mjm.body_pos [mid] = np.array({pstr})")
                print(f"  mjm.body_quat[mid] = np.array({qstr})")
                print(f"  # euler deltas: rx={s_rx.value:.1f} ry={s_ry.value:.1f} rz={s_rz.value:.1f}\n")

    if port_arg is not None:
        import viser
        viewer = MyGripperViewer(model=mjm, data=mjd, server=viser.ViserServer(port=port_arg))
    else:
        viewer = MyGripperViewer(model=mjm, data=mjd)
    viewer._paused = True
    port = viewer._server.get_port()
    print(f"\nOpen http://localhost:{port}")
    print(f"  SSH tunnel : ssh -L {port}:localhost:{port} <user>@<remote>")
    print("Actuation tab -> scroll down -> 'Mount Adjustment' folder.")
    print("Drag sliders to orient the gripper, click Print to bake the values. Ctrl-C to stop.\n")
    viewer.run()


if __name__ == "__main__":
    main()
