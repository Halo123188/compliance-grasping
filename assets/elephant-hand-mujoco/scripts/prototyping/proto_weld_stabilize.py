"""Stabilize the 3 four-bar WELD loop closures (hand-alone) via inertia regularization.

The earlier WELD attempt blew up to NaN: tiny linkage links (2-30 g, inertia 1e-6..1e-8)
make the mass matrix near-singular under a 6-DOF weld. Fix: floor body mass + diagonal
inertia, add armature/damping, use Newton solver + stiff weld solref/solimp, seed qpos0
at the assembly pose so the welds start satisfied.

This is a sweep harness: it tries a grid of (mass_floor, inertia_floor, armature, damping,
solref, impratio, timestep) and reports which configs settle finite with small weld
residual, then which survive an actuated sweep of the input loop joints.

Run: /home/alok/.conda/envs/g313/bin/python scripts/proto_weld_stabilize.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "src/my_mjlab_project/assets/robots/mygripper_H100_R/robot.urdf"

MESH_ORIGINS = {
  "s_link2_step": ((0.0005, 0.0005, 0.001), (0, 0, 0)),
  "s_link2_step_2": ((-0.0334378, 0.016261, -0.008), (0, 0, -0.780037)),
  "finger3_step": ((0.0, 0.0, 0.0009), (-3.14159, 0, 0)),
  "finger3_step_2": ((-0.0359102, -0.0134137, -0.0079), (-3.14159, 0, 0.652269)),
  "finger4_step": ((0.0, 0.0, 0.0009), (0, 0, 0)),
  "finger4_step_2": ((-0.0377587, 0.00661434, -0.008), (0, 0, -0.468199)),
}

FINGERS = [
  {
    "name": "F1",
    "b1": "s_link2_step",
    "b2": "s_link2_step_2",
    "input": "nRevolute 5",
    "base": "Revolute 1",
    "tip": "s_link2_step_2",
    "loop": ["Revolute 4", "nRevolute 5", "nCylindrical 1", "nRevolute 6"],
  },
  {
    "name": "F2",
    "b1": "finger3_step",
    "b2": "finger3_step_2",
    "input": "nRevolute 8",
    "base": "Revolute 2",
    "tip": "finger3_step_2",
    "loop": ["Revolute 7", "nRevolute 8", "nCylindrical 2", "nRevolute 9"],
  },
  {
    "name": "F3",
    "b1": "finger4_step",
    "b2": "finger4_step_2",
    "input": "nRevolute 11",
    "base": "Revolute 3",
    "tip": "finger4_step_2",
    "loop": ["Revolute 10", "nRevolute 11", "nCylindrical 3", "nRevolute 12"],
  },
]

ASSEMBLY = {
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

# Bodies that form the linkage mechanism (everything except base_step root).
LINKAGE_BODIES = [
  "m_link_step",
  "s_link2_step",
  "s_link1_step",
  "finger1_step",
  "s_link2_step_2",
  "dae_mhl_step",
  "finger3_step",
  "finger2_step",
  "finger1_step_2",
  "finger3_step_2",
  "dae_mhr_step",
  "finger4_step",
  "finger2_step_2",
  "finger1_step_3",
  "finger4_step_2",
]


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


def build(cfg):
  """cfg = dict(mass_floor, inertia_floor, armature, damping, solref, solimp,
                impratio, timestep, iterations, ls_iterations, solver).

  Inertia regularization is done POST-COMPILE on mjm.body_mass / mjm.body_inertia.
  The onshape URDF stores inertia as fullinertia (off-diagonal terms), so the
  MjSpec.inertia principal-moment vector is unpopulated until compile; editing the
  compiled diagonal inertia is the robust path (M is recomputed each step)."""
  spec = mujoco.MjSpec.from_file(str(HAND_URDF))

  # position actuators on driven joints (base swing + input loop joint)
  KP, KV = cfg.get("kp", 5.0), cfg.get("kv", 0.5)
  driven = [j for f in FINGERS for j in (f["base"], f["input"])]
  for jn in driven:
    a = spec.add_actuator()
    a.name = f"act_{jn}".replace(" ", "_")
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

  # weld constraints
  for f in FINGERS:
    pos, quat = relpose(f["b1"], f["b2"])
    eq = spec.add_equality()
    eq.name = f"weld_{f['name']}"
    eq.name1 = f["b1"]
    eq.name2 = f["b2"]
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.type = mujoco.mjtEq.mjEQ_WELD
    data = np.zeros(11)
    data[:3] = 0.0  # anchor in body2 frame (origins coincide at assembly)
    data[3:6] = pos
    data[6:10] = quat
    data[10] = cfg.get("torquescale", 1.0)
    eq.data = data
    eq.solref = np.array(cfg["solref"])
    eq.solimp = np.array(cfg["solimp"])

  # disable self-collision (the 4-bar links overlap as a mechanism)
  for g in spec.geoms:
    g.contype = 0
    g.conaffinity = 0

  spec.option.solver = cfg["solver"]
  spec.option.iterations = cfg["iterations"]
  spec.option.ls_iterations = cfg["ls_iterations"]
  spec.option.impratio = cfg["impratio"]
  spec.option.timestep = cfg["timestep"]

  m = spec.compile()

  # --- INERTIA REGULARIZATION (post-compile; M recomputed each step) ---
  mf = cfg["mass_floor"]
  inf = cfg["inertia_floor"]
  for bn in LINKAGE_BODIES:
    i = bid(m, bn)
    if i < 0:
      continue
    if m.body_mass[i] < mf:
      m.body_mass[i] = mf
    m.body_inertia[i] = np.maximum(m.body_inertia[i], inf)
  # body_subtreemass etc. are recomputed by mj_step from body_mass; no need to touch.

  # armature + damping on the gripper joints (all joints here are gripper joints)
  for i in range(m.njnt):
    adr = m.jnt_dofadr[i]
    m.dof_armature[adr] = cfg["armature"]
    m.dof_damping[adr] = cfg["damping"]
  return m


def jadr(m, n):
  return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])


def bid(m, n):
  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def q_assembly(m):
  q = np.zeros(m.nq)
  for jn, v in ASSEMBLY.items():
    q[jadr(m, jn)] = v
  return q


def settle_test(m, gravity=False, nsteps=300):
  """Reset to assembly, hold driven ctrl, step. Returns (finite, max_weld_res_mm, q)."""
  d = mujoco.MjData(m)
  if not gravity:
    m.opt.gravity[:] = 0.0
  q0 = q_assembly(m)
  d.qpos[:] = q0
  mujoco.mj_forward(m, d)
  # hold driven joints at assembly value
  for f in FINGERS:
    for jn in (f["base"], f["input"]):
      ai = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jn}".replace(" ", "_")
      )
      d.ctrl[ai] = d.qpos[jadr(m, jn)]
  maxres = 0.0
  for k in range(nsteps):
    mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
      return False, float("inf"), d, k
    if d.nefc:
      maxres = max(maxres, np.max(np.abs(d.efc_pos[: d.nefc])))
  return True, maxres * 1000, d, nsteps


def main():
  base = dict(
    mass_floor=0.04,
    inertia_floor=5e-5,
    armature=0.01,
    damping=0.3,
    solref=[0.01, 1.0],
    solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
    impratio=10.0,
    timestep=0.001,
    iterations=150,
    ls_iterations=50,
    solver=mujoco.mjtSolver.mjSOL_NEWTON,
    kp=5.0,
    kv=0.5,
    torquescale=1.0,
  )
  print("=== baseline config settle test (gravity OFF) ===", flush=True)
  m = build(base)
  print(f"nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq}", flush=True)
  ok, res, d, k = settle_test(m, gravity=False)
  print(
    f"baseline: finite={ok} max_weld_res={res:.4f} mm  (stopped at step {k})",
    flush=True,
  )

  ok, res, d, k = settle_test(m, gravity=True)
  print(
    f"baseline w/GRAVITY: finite={ok} max_weld_res={res:.4f} mm (step {k})", flush=True
  )

  # sweep the key knobs
  print(
    "\n=== sweep mass_floor x inertia_floor x armature (gravity ON, 300 steps) ===",
    flush=True,
  )
  results = []
  for mf, inf, arm in itertools.product(
    [0.0, 0.02, 0.04], [0.0, 1e-5, 5e-5, 1e-4], [0.0, 0.005, 0.02, 0.05]
  ):
    cfg = dict(base, mass_floor=mf, inertia_floor=inf, armature=arm)
    try:
      m = build(cfg)
      ok, res, d, k = settle_test(m, gravity=True, nsteps=300)
    except Exception:
      ok, res, k = False, float("inf"), -1
    flag = "OK " if ok and res < 1.0 else ("fin" if ok else "NaN")
    results.append((mf, inf, arm, ok, res, flag))
    print(
      f"  mass={mf:.2f} inertia={inf:.0e} arm={arm:.3f} -> {flag} res={res:.4f}mm step={k}",
      flush=True,
    )

  stable = [r for r in results if r[3] and r[4] < 1.0]
  print(f"\n{len(stable)}/{len(results)} configs stable (finite, res<1mm)", flush=True)
  if stable:
    # smallest regularization that works
    stable.sort(key=lambda r: (r[0], r[1], r[2]))
    print("smallest stable:", stable[0][:3], "res=%.4fmm" % stable[0][4], flush=True)

  # --- alignment under MOVEMENT for the chosen physical config ---
  print("\n=== ALIGNMENT UNDER MOVEMENT (input loop joint swept) ===", flush=True)
  chosen = dict(base, mass_floor=0.0, inertia_floor=5e-5, armature=0.01)
  m = build(chosen)
  print(
    "config: mass_floor=0 inertia_floor=5e-5 armature=0.01 damping=0.3 "
    "solref=[0.01,1] solimp=[0.99,0.999,..] impratio=10 dt=0.001 Newton/150",
    flush=True,
  )
  for f in FINGERS:
    worlds, residuals, inputs, finite = sweep_input(m, f, gravity=True)
    if not worlds:
      print(f"  {f['name']}: DIVERGED", flush=True)
      continue
    align = np.array(worlds) * 1000  # mm anchor-distance between the two coupler bodies
    res = np.array(residuals) * 1000
    print(
      f"  {f['name']}: input[{inputs[0]:+.2f}..{inputs[-1]:+.2f}] "
      f"anchor-align max={align.max():.4f} mean={align.mean():.4f} mm | "
      f"efc-res max={res.max():.4f} mean={res.mean():.4f} mm | finite={finite}",
      flush=True,
    )


def sweep_input(m, f, gravity=True, lo=-0.6, hi=1.2, n=25, settle=60):
  """Sweep the finger's INPUT loop joint across [lo,hi] (rad, relative to assembly),
  settle, and measure the WELD ANCHOR ALIGNMENT error = world distance between the
  weld anchor point computed from body1 vs body2 (should be ~0 for a true weld)."""
  d = mujoco.MjData(m)
  if not gravity:
    m.opt.gravity[:] = 0.0
  q0 = q_assembly(m)
  d.qpos[:] = q0
  mujoco.mj_forward(m, d)
  inp_jn = f["input"]
  ai_inp = mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{inp_jn}".replace(" ", "_")
  )
  # hold all driven at assembly
  for ff in FINGERS:
    for jn in (ff["base"], ff["input"]):
      ai = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jn}".replace(" ", "_")
      )
      d.ctrl[ai] = d.qpos[jadr(m, jn)]
  for _ in range(100):
    mujoco.mj_step(m, d)

  b1, b2 = bid(m, f["b1"]), bid(m, f["b2"])
  pos_rel, _ = relpose(f["b1"], f["b2"])  # body2 origin in body1 frame
  inp0 = d.qpos[jadr(m, inp_jn)]
  sweep = np.linspace(inp0 + lo, inp0 + hi, n)
  worlds, residuals, inputs = [], [], []
  for tgt in sweep:
    d.ctrl[ai_inp] = tgt
    for _ in range(settle):
      mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
      return worlds, residuals, inputs, False
    # weld anchor point in world from BOTH bodies (anchor = body2 origin):
    # from body2: just xpos[b2]. from body1: xpos[b1] + R1 @ pos_rel.
    p_from_b2 = d.xpos[b2].copy()
    R1 = d.xmat[b1].reshape(3, 3)
    p_from_b1 = d.xpos[b1] + R1 @ pos_rel
    worlds.append(float(np.linalg.norm(p_from_b1 - p_from_b2)))
    residuals.append(float(np.max(np.abs(d.efc_pos[: d.nefc]))) if d.nefc else 0.0)
    inputs.append(float(d.qpos[jadr(m, inp_jn)]))
  return worlds, residuals, inputs, True


if __name__ == "__main__":
  main()
