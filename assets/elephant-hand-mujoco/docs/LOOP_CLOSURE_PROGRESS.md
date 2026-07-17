# Four-bar Loop Closure Prototype — mygripper_H100_R ("elephant")

Goal: close the 3 four-bar linkages (loop-cut as duplicate coupler links) so driving
the 6 proximal joints articulates each finger as a ~1-DOF linkage.

## Confirmed structure (hand-alone names)
| Finger | body1 (input coupler) | body2 (distal coupler) | driven joints | loop joints |
|---|---|---|---|---|
| F1 | s_link2_step  | s_link2_step_2  | Revolute 1, nRevolute 5  | Revolute 4, nRevolute 5, nCylindrical 1, nRevolute 6 |
| F2 | finger3_step  | finger3_step_2  | Revolute 2, nRevolute 8  | Revolute 7, nRevolute 8, nCylindrical 2, nRevolute 9 |
| F3 | finger4_step  | finger4_step_2  | Revolute 3, nRevolute 11 | Revolute 10, nRevolute 11, nCylindrical 3, nRevolute 12 |

Gap at rest qpos0 (probe): F1 38.9mm, F2 39.3mm, F3 39.4mm.

## Env
- python: /home/alok/.conda/envs/g313/bin/python  (mujoco 3.8.1, scipy 1.17.1, numpy 2.4.6)
- viewer RUNNING on port 8083 — DO NOT launch viser. Headless only.

## Status
- [x] Read URDF + viewer + helper scripts
- [x] Step 1: compute M1*inv(M2) relpose per finger  (EXACT)
- [x] Step 2: solve assembly config per finger -> qpos0  (residual 0.000mm/0.000deg all 3)
- [x] Step 3: add 3 welds  (WELD + CONNECT both BLOW UP — planar 4-bar redundancy)
- [x] Step 4: validate headless
- [x] Step 5: FALLBACK -> Option B JOINT-MIMIC works (clean 1-DOF, cubic fit)
- [ ] Step 6: wire into build_model_coupled() + --coupled flag

## KEY FINDING — mechanism identification
- Revolute 1/2/3 (base) are NOT the 4-bar input: they swing the whole finger; sweeping
  them barely moves the loop (F3 passive joints moved <0.005 rad).
- The 4-bar's single internal DOF input is the LOOP joint nRevolute 5/8/11.
- Driving that loop joint, the loop stays closeable over [-0.6,1.2] rad and the fingertip
  (distal coupler body) travels: F1 39.8mm, F2 26.5mm, F3 35.2mm  -> sensible 1-DOF.
- Passive joints follow a CUBIC of the input to <0.15deg (F1/F3); F2 ~6deg (more nonlinear).

## Step 3/4 — WELD/CONNECT result (FAILED as guardrails predicted)
- WELD: nefc=18 (3x6 correct), assembly residual 2.3e-6 m (great), BUT settles unstable
  -> NaN at DOF 6. CONNECT: nefc=9, residual 2.2e-7, also NaN at DOF 7.
- Root cause = planar 4-bar redundancy (parallel revolute axes already lock out-of-plane
  DOFs; 6-DOF weld is redundant) + tiny link inertias (1-3 g) -> ill-conditioned solver.
- Also had to disable self-collision: the 4-bar links physically overlap as a mechanism
  (31 self-contacts at assembly) -> must set contype/conaffinity=0 on linkage geoms.

## DECISION: use Option B joint-mimic (mjEQ_JOINT polycoef), input = nRevolute 5/8/11.

## WORKING MIMIC RESULT (hand-alone, headless, scripts/proto_mimic_validate.py)
- 9 mjEQ_JOINT constraints (3 per finger): name1=DEPENDENT passive joint, name2=INDEPENDENT input.
  polycoef [c0,c1,c2,c3,0], passive = c0 + c1*inp + c2*inp^2 + c3*inp^3 (input-relative, qpos0=0).
- mjEQ_JOINT semantics confirmed: joint1=dependent, joint2=driver, polycoef lowest-order first.
- solref=[0.01,1.0] solimp=[0.95,0.99,0.001,0.5,2.0]; solver iterations=100 ls_iterations=50.
- dof_damping=0.2 dof_armature=0.002 on ALL joints (massless free DOFs need it).
- contype/conaffinity=0 on all linkage geoms (they overlap as a mechanism).
- KINEMATIC loop-closure error (passive=cubic(input)): F1 max0.048mm, F2 max0.513mm, F3 max0.055mm.
- DYNAMIC validation: stable (finite), settle efc 0.341mm.
  Tip travel: F1 27.7mm F2 23.0mm F3 24.1mm (path~=net -> clean monotonic 1-DOF, no flop).
  Loop relpose error across full sweep: F1 max1.48 F2 max1.95 F3 max1.52 mm (within few-mm target).
- Cubic coeffs [c0,c1,c2,c3]:
  F1 Revolute4=[−4e-5,0.59914,−0.01565,0.02358] nCyl1=[−9e-5,0.44785,−0.25694,−0.01842] nRev6=[−0.00013,0.04698,−0.27258,0.00517]
  F2 Revolute7=[−0.00326,0.62842,0.02157,−0.21583] nCyl2=[0.0072,0.82657,0.21622,0.35956] nRev9=[0.00393,0.45499,0.2378,0.14373]
  F3 Revolute10=[−0.0001,0.55013,−0.04187,0.005] nCyl3=[0.0,−0.62893,0.29248,0.00264] nRev12=[−0.00011,0.17906,−0.33435,0.00235]

## Step 6: wire into build_model_coupled() + --coupled flag  (DONE)
- Added to src/my_mjlab_project/mygripper_arm_viewer.py (build_model() UNTOUCHED):
  * module consts COUPLED_FINGERS / COUPLED_DRIVEN / COUPLED_PASSIVE (cubic coeffs baked in)
  * build_model_coupled(): arm+hand attach, 6 driven actuators, 9 mjEQ_JOINT loop closures,
    contype/conaffinity=0 on hand geoms, iterations=100/ls_iterations=50, damping0.2/armature0.002
  * --coupled CLI flag in main(); _print_stats() shows coupled stats
- Equalities added to the HAND spec BEFORE attach() so the "_h" suffix retargets joint refs
  automatically (verified attach remaps mjEQ_JOINT joint1/joint2 + keeps polycoef).

## VALIDATION (headless, both models, no viser launched)
- `--check`     : build_model() unchanged -> nq29 nu22 (7 arm + 15 gripper servos)
- `--check --coupled`: nq29 nu13 (7 arm + 6 driven) neq9
- HAND-ALONE dynamic sweep (scripts/proto_mimic_validate.py): stable, tip travel F1 27.7/F2 23.0/F3 24.1 mm,
  loop relpose err max F1 1.48/F2 1.95/F3 1.52 mm.
- COMBINED arm+hand coupled dynamic sweep (driving each /nRevolute*_h actuator, arm at pre-grasp):
  F1 tip 28.5mm err max0.83/mean0.19 | F2 tip 23.4mm err max1.32/mean0.65 | F3 tip 24.8mm err max0.92/mean0.21 mm
  all finite, settle efc 1.34mm.

## FINAL: WORKING APPROACH = JOINT-MIMIC (mjEQ_JOINT cubic). WELD/CONNECT both blow up (planar redundancy).

## Test commands
  cd /home/alok/hand_mjlab/my_mjlab_project
  PYTHONPATH=src /home/alok/.conda/envs/g313/bin/python -m my_mjlab_project.mygripper_arm_viewer --check --coupled
  /home/alok/.conda/envs/g313/bin/python scripts/proto_mimic_validate.py   # hand-alone dynamic sweep
  /home/alok/.conda/envs/g313/bin/python scripts/proto_loop_mimic2.py      # re-derive cubic coeffs

## Files added
- src/my_mjlab_project/mygripper_arm_viewer.py  (ADD build_model_coupled + --coupled; build_model untouched)
- scripts/proto_loop_closure.py    (relpose + assembly solve)
- scripts/proto_loop_validate.py   (weld/connect attempt -> shows the blowup)
- scripts/proto_loop_mimic.py      (proved base joints are NOT the 4-bar input)
- scripts/proto_loop_mimic2.py     (identify input loop joint + fit cubic mimic)
- scripts/proto_mimic_validate.py  (final hand-alone dynamic validation)

## Manual help still useful (optional)
- Viewer is live on 8083; user can run `--coupled` in a SEPARATE port to eyeball the
  fingers curling, and drag /nRevolute 5_h etc. to confirm the assembly pose looks right.
- F2 mimic is the least linear (cubic res ~6deg, 2nd cut at finger3 has the most nonlinear
  4-bar). If F2 grasp looks off, refit with degree-4 polycoef (slot data[4] is free).

## Step 1 results — weld relpose (body2 in body1, pos[xyz] + quat[wxyz])
- F1 s_link2_step<-s_link2_step_2: pos=[0.035707,0.012457,0.009]    quat=[0.924902,0,0,0.380206]
- F2 finger3_step<-finger3_step_2: pos=[0.03668,-0.011137,0.0088]   quat=[0.947288,0,0,-0.320384]
- F3 finger4_step<-finger4_step_2: pos=[0.03668,0.011137,0.0089]    quat=[0.972724,0,0,0.231967]

## Step 2 results — assembly loop-joint angles (rad), all close loop to 0.000mm
- F1: Revolute 4=0.1185, nRevolute 5=0.1969, nCylindrical 1=0.0775, nRevolute 6=-0.0010
- F2: Revolute 7=0.1832, nRevolute 8=0.3222, nCylindrical 2=0.3312, nRevolute 9=0.1922
- F3: Revolute 10=-0.0019, nRevolute 11=-0.0035, nCylindrical 3=0.0022, nRevolute 12=-0.0007
NOTE: driven joints nRevolute 5/8/11 are ALSO loop joints -> their assembly value is the
baseline; driving them changes the loop config (weld keeps it closed).

## Log
(updates below)
