"""Step 3-4: add welds, set assembly qpos0 keyframe, drive driven joints, validate.

Builds the hand-alone MjSpec, adds 3 WELD equality constraints closing each
four-bar loop at the solved assembly relpose, adds position actuators on the 6
driven joints, sets the assembly qpos as a keyframe, then sweeps each finger's
two driven joints and reports:
  (a) qpos finite / no blow-up
  (b) fingertip travel (distal body world displacement)
  (c) weld constraint residual (efc) over the sweep

Run: /home/alok/.conda/envs/g313/bin/python scripts/proto_loop_validate.py [weld|connect]
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "src/my_mjlab_project/assets/robots/mygripper_H100_R/robot.urdf"

MESH_ORIGINS = {
    "s_link2_step":   (( 0.0005,    0.0005,   0.001 ), (0, 0, 0)),
    "s_link2_step_2": ((-0.0334378, 0.016261, -0.008), (0, 0, -0.780037)),
    "finger3_step":   (( 0.0,       0.0,      0.0009), (-3.14159, 0, 0)),
    "finger3_step_2": ((-0.0359102, -0.0134137, -0.0079), (-3.14159, 0, 0.652269)),
    "finger4_step":   (( 0.0,       0.0,      0.0009), (0, 0, 0)),
    "finger4_step_2": ((-0.0377587, 0.00661434, -0.008), (0, 0, -0.468199)),
}

# distal "fingertip" body per finger (the distal coupler is the fingertip phalanx)
FINGERS = [
    {"name": "F1", "b1": "s_link2_step", "b2": "s_link2_step_2",
     "driven": ["Revolute 1", "nRevolute 5"], "tip": "s_link2_step_2",
     "loop": ["Revolute 4", "nRevolute 5", "nCylindrical 1", "nRevolute 6"]},
    {"name": "F2", "b1": "finger3_step", "b2": "finger3_step_2",
     "driven": ["Revolute 2", "nRevolute 8"], "tip": "finger3_step_2",
     "loop": ["Revolute 7", "nRevolute 8", "nCylindrical 2", "nRevolute 9"]},
    {"name": "F3", "b1": "finger4_step", "b2": "finger4_step_2",
     "driven": ["Revolute 3", "nRevolute 11"], "tip": "finger4_step_2",
     "loop": ["Revolute 10", "nRevolute 11", "nCylindrical 3", "nRevolute 12"]},
]

# Solved assembly loop-joint angles (from proto_loop_closure.py).
ASSEMBLY = {
    "Revolute 4": 0.11850651675287728, "nRevolute 5": 0.19694636697490253,
    "nCylindrical 1": 0.0774500849888125, "nRevolute 6": -0.0009871188288751393,
    "Revolute 7": 0.1831730332809386, "nRevolute 8": 0.32215584872455566,
    "nCylindrical 2": 0.3312176032729121, "nRevolute 9": 0.19223044142564735,
    "Revolute 10": -0.0019317300325050431, "nRevolute 11": -0.0034647938206841134,
    "nCylindrical 3": 0.0021822067379099544, "nRevolute 12": -0.0006524893470445766,
}


def mesh_T(name):
    xyz, rpy = MESH_ORIGINS[name]
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def relpose(b1, b2):
    rel = mesh_T(b1) @ np.linalg.inv(mesh_T(b2))
    q = R.from_matrix(rel[:3, :3]).as_quat()  # xyzw
    return rel[:3, 3].copy(), np.array([q[3], q[0], q[1], q[2]])


def build(mode="weld"):
    spec = mujoco.MjSpec.from_file(str(HAND_URDF))

    # position actuators on the 6 driven joints
    KP, KV = 5.0, 0.3
    for f in FINGERS:
        for jn in f["driven"]:
            a = spec.add_actuator()
            a.name = f"act_{jn}".replace(" ", "_")
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.target = jn
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gp = np.zeros(mujoco.mjNGAIN); gp[0] = KP; a.gainprm = gp
            bp = np.zeros(mujoco.mjNBIAS); bp[1] = -KP; bp[2] = -KV; a.biasprm = bp
            a.ctrlrange = np.array([-np.pi, np.pi])

    # equality constraints
    for f in FINGERS:
        pos, quat = relpose(f["b1"], f["b2"])
        eq = spec.add_equality()
        eq.name = f"loop_{f['name']}"
        eq.name1 = f["b1"]
        eq.name2 = f["b2"]
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        if mode == "weld":
            eq.type = mujoco.mjtEq.mjEQ_WELD
            data = np.zeros(11)
            data[:3] = 0.0          # anchor point in body2 frame (origins coincide)
            data[3:6] = pos         # relpose translation: body2 origin in body1 frame
            data[6:10] = quat       # relpose quat wxyz
            data[10] = 1.0          # torquescale
            eq.data = data
            # softer constraint to absorb planar redundancy
            eq.solref = np.array([0.02, 1.0])
            eq.solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
        elif mode == "connect":
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            data = np.zeros(11)
            # connect anchor: world pivot expressed... use M1 origin in body1 frame
            data[:3] = mesh_T(f["b1"])[:3, 3]
            eq.data = data
            eq.solref = np.array([0.02, 1.0])
            eq.solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0])

    # Disable self-collision of the linkage geoms: the four-bar links physically
    # overlap as a mechanism (proven by 31 self-contacts at the assembly pose).
    # For kinematic loop-closure validation we turn collisions off entirely.
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0

    m = spec.compile()
    return m


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "weld"
    print(f"=== MODE: {mode} ===", flush=True)
    m = build(mode)
    d = mujoco.MjData(m)

    def jadr(n):
        return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
    def bid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    print(f"nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} nbody={m.nbody}", flush=True)

    # gravity off for kinematic validation (no mass to fight; isolate the linkage)
    m.opt.gravity[:] = 0.0
    m.opt.timestep = 0.002

    # build assembly qpos
    q0 = np.zeros(m.nq)
    for jn, v in ASSEMBLY.items():
        q0[jadr(jn)] = v

    def reset_to(q):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)

    reset_to(q0)
    # initial weld residual
    print(f"initial efc (after forward at assembly): nefc={d.nefc} "
          f"max|efc_pos|={np.max(np.abs(d.efc_pos[:d.nefc])) if d.nefc else 0:.2e}",
          flush=True)

    # settle at assembly pose with ctrl holding driven joints at their assembly values
    ctrl0 = np.zeros(m.nu)
    for i in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jn = an.replace("act_", "").replace("_", " ")
        # map back to joint name (handles "Revolute 1" etc.)
    # simpler: set ctrl to current driven qpos
    for f in FINGERS:
        for jn in f["driven"]:
            ai = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   f"act_{jn}".replace(" ", "_"))
            ctrl0[ai] = d.qpos[jadr(jn)]
    d.ctrl[:] = ctrl0
    for _ in range(200):
        mujoco.mj_step(m, d)
    finite_settle = bool(np.all(np.isfinite(d.qpos)))
    settle_res = np.max(np.abs(d.efc_pos[:d.nefc])) if d.nefc else 0.0
    print(f"after 200 settle steps: finite={finite_settle} "
          f"max|efc_pos|={settle_res:.2e} m", flush=True)

    # ---- drive each finger across range, measure tip travel + residual ----
    print("\n--- sweep driven joints, measure fingertip travel + weld residual ---", flush=True)
    for f in FINGERS:
        reset_to(q0)
        d.ctrl[:] = ctrl0
        for _ in range(100):
            mujoco.mj_step(m, d)
        tip = bid(f["tip"])
        # primary driven joint = the base joint (Revolute 1/2/3). Sweep it open->close.
        base_jn = f["driven"][0]
        ai = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                               f"act_{base_jn}".replace(" ", "_"))
        base0 = d.qpos[jadr(base_jn)]
        sweep = np.linspace(base0, base0 + 1.2, 25)  # ~70deg of drive
        tip_positions = []
        max_res = 0.0
        finite = True
        loop_jn_record = []
        for target in sweep:
            d.ctrl[ai] = target
            for _ in range(60):
                mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                finite = False
                break
            tip_positions.append(d.xpos[tip].copy())
            if d.nefc:
                max_res = max(max_res, np.max(np.abs(d.efc_pos[:d.nefc])))
        tip_positions = np.array(tip_positions)
        if len(tip_positions) >= 2:
            travel = np.linalg.norm(tip_positions[-1] - tip_positions[0]) * 1000
            path = np.sum(np.linalg.norm(np.diff(tip_positions, axis=0), axis=1)) * 1000
        else:
            travel = path = 0.0
        # how much did each non-driven loop joint move (proves coupling)?
        passive = [j for j in f["loop"] if j != base_jn]
        print(f"{f['name']}: finite={finite}  tip travel(net)={travel:.1f} mm  "
              f"path len={path:.1f} mm  max|efc_pos|={max_res*1000:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
