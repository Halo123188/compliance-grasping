# WELD Loop Closure — mygripper_H100_R ("elephant")

Goal: PHYSICALLY-CORRECT four-bar loop closure via rigid WELD constraints (fuse each
finger's duplicate coupler body into one), made STABLE via inertia regularization.
The MIMIC approach lets the two coupler bodies visibly separate ("two mounts"); only a
weld physically joins them. Prove the weld holds alignment ~0 under movement.

## Env
- python: /home/alok/.conda/envs/g313/bin/python  (mujoco 3.8.1, numpy 2.4.6, scipy 1.17.1)
- COUPLED viewer RUNNING on port 8083 — DO NOT bind it. Headless only.

## Weld pairs (hand-alone names; body1 <- body2) + relpose (body2 in body1)
- F1 s_link2_step <- s_link2_step_2: pos=[0.035707,0.012457,0.009]  quat=[0.924902,0,0,0.380206]
- F2 finger3_step <- finger3_step_2: pos=[0.03668,-0.011137,0.0088] quat=[0.947288,0,0,-0.320384]
- F3 finger4_step <- finger4_step_2: pos=[0.03668,0.011137,0.0089]  quat=[0.972724,0,0,0.231967]

## Assembly qpos0 (rad) — close loops to 0.000mm
- F1: Revolute 4=0.1185, nRevolute 5=0.1969, nCylindrical 1=0.0775, nRevolute 6=-0.0010
- F2: Revolute 7=0.1832, nRevolute 8=0.3222, nCylindrical 2=0.3312, nRevolute 9=0.1922
- F3: Revolute 10=-0.0019, nRevolute 11=-0.0035, nCylindrical 3=0.0022, nRevolute 12=-0.0007

## Body masses/inertias (hand-alone, compiled) — the ill-conditioning source
links are 2-30 g, inertia diag 1e-6..1e-8 kg*m^2. Weld of two such bodies => near-singular M.

## Plan / fixes to apply
1. INERTIA REGULARIZATION (key): floor body mass (~0.03-0.05) + diag inertia (~1e-5..1e-4) on linkage bodies.
2. ARMATURE (~0.005-0.05) + DAMPING (~0.1-0.5) on gripper joints.
3. Newton solver, iterations 150-200, ls 50, impratio 5-50, timestep 0.002->0.001/0.0005, weld solref/solimp stiff.
4. contype=conaffinity=0 on hand linkage geoms.
5. Seed qpos0 with assembly angles.

## Log

### WELD STABILIZED (hand-alone) — armature is the real regularizer
- Earlier blow-up reproduced: armature=0 AND inertia_floor=0 -> finite but weld DRIFTS 414mm
  (= the "two mounts separate" symptom). Adding EITHER armature>=0.005 OR inertia_floor>=5e-5
  stops it. 42/48 of the (mass,inertia,armature) grid stable.
- KEY: armature alone (inertia_floor=0, armature=0.01) gives IDENTICAL alignment (0.1785mm)
  to a heavy 5e-5 inertia floor. The 5e-5 floor inflates ALL 15 links 3-700x (non-physical);
  armature (reflected inertia at the joint) is the robust, minimally-invasive fix. -> use it.
- INERTIA FLOOR NOT NEEDED for stability with armature present (data: inf 0..5e-5 -> align unchanged).
  Keep mass_floor=0, inertia_floor=0 (fully physical bodies).
- Weld stiffness: solref TIME CONSTANT is the dominant alignment knob.
  solref=[0.02,1]->1.78mm, [0.01,1]->0.56mm, [0.005,1]->0.18mm, [0.002,1]->0.13mm.
  Chose solref=[0.005,1] (sub-0.2mm, robust). solimp=[0.99,0.999,0.001,0.5,2.0].
  impratio has NO effect (bilateral constraint, no friction cone). dt=0.001.
- armature sweet spot = 0.01 (0.005 & 0.02 both slightly worse). damping=0.3.

### FINAL WELD CONFIG (hand-alone validated)
  mass_floor=0, inertia_floor=0, armature=0.01, damping=0.3 (on 15 gripper joints)
  solver=Newton, iterations=150, ls_iterations=50, impratio=1, timestep=0.001
  weld solref=[0.005,1.0] solimp=[0.99,0.999,0.001,0.5,2.0] torquescale=1
  contype=conaffinity=0 on hand geoms; qpos0 seeded at assembly angles.

### ALIGNMENT UNDER MOVEMENT (hand-alone, input swept [-0.6,1.2] rad, anchor world-dist)
  F1: max 0.179 mean 0.030 mm | F2: max 0.147 mean 0.059 mm | F3: max 0.088 mean 0.031 mm
  -> vs MIMIC loop relpose err up to 1.48/1.95/1.52 mm. WELD is ~10x tighter.
### CLEAN 1-DOF (input drive): passive joints move 13-110deg (loop genuinely coupled).
  tip net 13-31mm (net/path 0.52-0.79 = real curved coupler arc, not a flop).
### BASE-swing drive (independent DOF): align stays 0.002-0.006mm (loop unchanged, swings rigidly).

### TODO -> ALL DONE
- [x] port to build_model_welded() + --weld flag (additive; build_model/build_model_coupled untouched)
- [x] scripts/weld_alignment_diag.py (weld vs mimic table on COMBINED model)
- [x] scripts/weld_validate_combined.py (drive all 6, arm pre-grasp) -> PASS
- [x] live alignment readout in viewer (overrides _tick, updates GUI text panel per finger)

### NAMING GOTCHA (caught + fixed)
- MjSpec.attach(suffix="_h") renames hand objects to "/<name>_h" (LEADING SLASH).
  Initial qpos seeding used "<name>_h" (no slash) -> silently no-op'd -> welds did NOT
  start satisfied. Fixed with _attached_id() helper that tries "/<name>_h" then "<name>".
  Now qpos0 seeds correctly: weld alignment at qpos0 = 0.0003 mm, initial efc 0.0023 mm.

### FINAL CONFIG (combined arm+hand; in build_model_welded)
  WELD: 3 mjEQ_WELD, anchor=body2 origin, relpose body2-in-body1 (precomputed),
        solref=[0.002,1.0] solimp=[0.99,0.999,0.001,0.5,2.0] torquescale=1.
  STABILITY: armature=0.01 + damping=0.3 on the 15 gripper joints. << the real fix.
  mass_floor=0, inertia_floor=0 (FULLY PHYSICAL bodies -- no inflation needed).
  SOLVER: Newton, iterations=150, ls_iterations=50, timestep=0.001.
  contype=conaffinity=0 on hand geoms. qpos0 seeded at assembly angles.
  -> stiffened solref from 0.005 to 0.002 after combined test: cuts the transient
     stretch under fast simultaneous 6-joint slew (F2 peak 0.81->0.35 mm) with no
     loss of stability and identical settled alignment.

### ALIGNMENT-UNDER-MOVEMENT TABLE (combined model, per-target settled, input swept assembly+[-0.6,1.2])
| Finger | WELD max | WELD mean | MIMIC max | MIMIC mean | improvement |
|--------|----------|-----------|-----------|------------|-------------|
| F1     | 0.031 mm | 0.006 mm  | 0.981 mm  | 0.184 mm   | 31.7x       |
| F2     | 0.119 mm | 0.039 mm  | 3.920 mm  | 0.934 mm   | 32.9x       |
| F3     | 0.022 mm | 0.007 mm  | 1.261 mm  | 0.193 mm   | 57.4x       |
(align = world distance between the two duplicate coupler bodies' shared anchor.
 MIMIC drifts up to 3.9 mm = the "two mounts appear" symptom; WELD holds sub-0.12 mm.)

### COMBINED VALIDATION (drive all 6 joints, 4 aggressive ramp phases, 3200 steps) -> PASS
  (a) qpos finite throughout: True
  (b) coupler separation: peak 0.14/0.35/0.18 mm (F1/F2/F3) during fastest simultaneous
      slew; settled 0.009/0.16/0.010 mm. efc residual peaks 3.3 mm = force-weighted
      constraint STRESS during slew, NOT geometric distance (which is the sep numbers).
  (c) clean 1-DOF: F1 tip net 35.5 ~= path 36.0 mm (net/path 0.985), passive joints move
      Revolute4 +34.8deg, nCyl1 +3.5deg, nRev6 -18.0deg -> loop genuinely articulates.
  (d) config reported above.

### HAND-ALONE cross-check (proto_weld_articulate, solref=0.002)
  input drive align_max: F1 0.045 F2 0.103 F3 0.051 mm; passive joints move 13-110deg.
  base-swing drive (independent DOF): align stays 0.0004-0.001 mm (loop rigidly swings).

### FILES
- src/my_mjlab_project/mygripper_arm_viewer.py: ADDED WELD_* consts, _attached_id(),
  build_model_welded(), weld_alignment_mm(), --weld flag, live weld GUI panel.
  build_model() and build_model_coupled() UNCHANGED.
- scripts/proto_weld_stabilize.py : inertia/armature/solver stability sweep + movement test.
- scripts/proto_weld_tune.py      : weld-stiffness tuning (solref/solimp/impratio/dt; minimal regularization).
- scripts/proto_weld_articulate.py: clean-1-DOF + base/input robustness (hand-alone).
- scripts/weld_alignment_diag.py  : weld-vs-mimic alignment table (combined model).
- scripts/weld_validate_combined.py: must-pass combined validation (a)-(d).

### HEADLESS COMMANDS
  cd /home/alok/hand_mjlab/my_mjlab_project
  PYTHONPATH=src /home/alok/.conda/envs/g313/bin/python -m my_mjlab_project.mygripper_arm_viewer --check --weld
  PYTHONPATH=src /home/alok/.conda/envs/g313/bin/python scripts/weld_alignment_diag.py
  PYTHONPATH=src /home/alok/.conda/envs/g313/bin/python scripts/weld_validate_combined.py
  # (parent can run the live viewer: ... --weld --port <N>; do NOT bind 8083)

### CONCLUSION
  WELD STABILIZED = YES. The blow-up was tiny-link inertia ill-conditioning; ARMATURE
  (reflected inertia at the joints) is the robust fix and needs ZERO mass/inertia
  inflation. A stiff weld (solref=0.002) + Newton + qpos0 seeding holds the two coupler
  bodies fused to sub-0.12 mm under movement, vs up to 3.9 mm drift for the mimic --
  the weld physically joins what the mimic only correlates in joint space.
