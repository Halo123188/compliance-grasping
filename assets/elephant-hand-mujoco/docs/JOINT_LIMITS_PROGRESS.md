# Joint Limits Derivation — mygripper_H100_R ("elephant" hand)

Goal: derive PHYSICALLY-PLAUSIBLE per-joint ROM limits for the Elephant Robotics
myGripper H100 (no datasheet publishes per-joint angles). Derive mechanically in the
WELDED sim, anchor to the only 2 REAL specs:
  - gripping aperture 0–130 mm (diameter the 3 fingers span)
  - per-joint velocity 60 deg/s = 1.047 rad/s
Then set joint range + actuator ctrlrange + velocity limit and validate headless.

## Env / guardrails
- python: /home/alok/.conda/envs/g313/bin/python (mujoco 3.8.1)
- WELD viewer RUNNING on port 8083 — DO NOT bind a port. Headless only.
- Additive edits to src/my_mjlab_project/mygripper_arm_viewer.py only. No git add/commit.

## Driven DOFs (2 per finger, 6 total)
- base swing: Revolute 1 (F1), Revolute 2 (F2), Revolute 3 (F3)  — proximal spread
- input curl: nRevolute 5 (F1), nRevolute 8 (F2), nRevolute 11 (F3) — main grasp DOF
## Passive (slaved by weld, 9 total)
- F1: Revolute 4, nCylindrical 1, nRevolute 6
- F2: Revolute 7, nCylindrical 2, nRevolute 9
- F3: Revolute 10, nCylindrical 3, nRevolute 12
## Fingertips: s_link2_step_2 (F1), finger3_step_2 (F2), finger4_step_2 (F3)

## Aperture proxy (documented choice)
The 3 fingers are arranged ~120deg apart (a centering 3-jaw). Aperture = diameter of
the largest object the 3 fingertips can enclose. Chosen proxy:
**circumscribed-circle diameter through the 3 fingertip CONTACT points**, where the
contact point = the fingertip geom center in world (body xpos + R @ geom_pos), the
physical pad surface (not the link origin). This is the diameter of the circle the 3
pads sit on = the largest sphere/cylinder they can cage. Documented + used consistently.

## KEY MECHANICAL FINDING (sim evidence overrides the naive method)
The prompt's METHOD assumed CURL (nRevolute 5/8/11) is the aperture DOF. The sim shows
the opposite: **BASE SWING (Revolute 1/2/3) is the dominant aperture DOF.** Sweeping the
3 base joints together opens/closes the 3-jaw; curl barely moves the fingertip radius
(F1 tip-palm dist changes only ~119->124 mm over the whole curl range). So:
  - BASE SWING bounded by APERTURE CALIBRATION (0-130 mm anchor).
  - CURL bounded by the FOUR-BAR SINGULARITY (weld dead-center).
The weld stays dead-rigid (0.001 mm) across the full base sweep (base is outside the
four-bar loop). Curl drives the loop, so its limit is the linkage lock.

## APERTURE CALIBRATION (base swing -> aperture, weld model, gravity on, arm pre-grasp)
Proxy = circumscribed-circle diameter through the 3 fingertip pad CENTERS (geom_xpos),
minus the closed-pose baseline 87.3 mm (the caging-circle minimum = enclosed object 0 mm).
Base delta is relative to the assembly base angle (~0 rad). Weld stayed 0.001 mm at all:

| base delta rad | base deg | aperture mm | note |
|----------------|----------|-------------|------|
| -0.70 | -40.1 | 129.1 | aperture PEAK == fully OPEN (130 mm spec) |
| -0.60 | -34.4 | 128.0 | |
| -0.40 | -22.9 | 116.7 | |
| -0.20 | -11.5 |  95.2 | |
|  0.00 |   0.0 |  68.4 | assembly pose |
| +0.20 | +11.5 |  42.0 | |
| +0.40 | +22.9 |  20.7 | |
| +0.60 | +34.4 |   6.9 | |
| +0.80 | +45.8 |   0.6 | |
| +0.90 | +51.6 |   0.0 | aperture MIN == fully CLOSED (0 mm spec) |
Beyond -0.70 the aperture turns over (re-closes), beyond +0.90 it re-opens slightly ->
both ends are bounded by APERTURE GEOMETRY, not self-collision. Selective collision check
(tips contype/conaffinity enabled, 20 mm margin) found NO tip-tip contact anywhere in the
range: the 3 fingers sit ~120 deg apart and converge on a common center, never on each
other. So the 0 mm "closed" = caging circle collapsed onto an enclosed-object-of-0, and
no self-collision occurs within the base range (documented; no collision bound needed).
=> BASE range set to [-0.70, +0.90] rad (open 130 mm -> closed 0 mm), per finger symmetric.

## CURL feasibility (four-bar singularity; weld>1mm or non-finite = dead-center)
Fine bisection per finger, abs angles (assembly value in parens):
  F1 nRevolute 5  (asm +0.198): feasible [-1.592,+1.934] rad -> set [-1.54,+1.88] (0.05 margin)
  F2 nRevolute 8  (asm +0.322): feasible [-1.658,+1.247] rad -> set [-1.61,+1.20] (locks earliest)
  F3 nRevolute 11 (asm -0.004): feasible [-1.593,+1.932] rad -> set [-1.54,+1.88]
F2 hits its dead-center first on the close side (weld alignment spikes 0.08->0.20->0.40 mm
at +1.18..+1.24 rad). Curl secondary effect on aperture is small (<10 mm) so curl is the
wrap/conform DOF, not the aperture DOF.

## PASSIVE ROM (weld-slaved; swept over full base[-0.7,+0.9] x curl-feasible envelope)
Recorded min/max each passive joint REACHES; set range = swept +/- 0.10 rad margin so they
never fight the weld (validated: weld stays <0.04 mm at the driven extremes WITH these set).
| joint | swept rad | set rad |
|-------|-----------|---------|
| Revolute 4      | [-0.75,+1.08] | [-0.85,+1.18] |
| nCylindrical 1  | [-0.96,+0.18] | [-1.06,+0.28] |
| nRevolute 6     | [-0.63,+0.00] | [-0.73,+0.10] |
| Revolute 7      | [-0.73,+0.44] | [-0.83,+0.54] |
| nCylindrical 2  | [-0.73,+1.15] | [-0.83,+1.25] |
| nRevolute 9     | [-0.20,+0.69] | [-0.30,+0.79] |
| Revolute 10     | [-0.66,+0.86] | [-0.76,+0.96] |
| nCylindrical 3  | [-0.33,+1.17] | [-0.43,+1.27] |
| nRevolute 12    | [-0.64,+0.02] | [-0.74,+0.12] |

## VELOCITY
60 deg/s = 1.0472 rad/s (real spec). Stored as JOINT_VEL_LIMIT. MuJoCo has no hard jnt
velocity limit (enforced via actuator/damping); the existing damped position servos
(KP=8, KV=0.4) already rate-limit slew. Value is available for a future
velocity-limited actuator / mjlab ActuatorCfg.velocity_limit. NOT a hard cap yet.

## CODE CHANGED (additive, non-breaking)
src/my_mjlab_project/mygripper_arm_viewer.py:
  + JOINT_VEL_LIMIT = 1.0472
  + JOINT_LIMITS dict (15 joints: 6 driven enforced + 9 passive swept)
  + _apply_joint_limits(mjm, driven) helper (sets jnt_range/jnt_limited + driven
    actuator_ctrlrange/ctrllimited, post-compile, attach "_h" aware)
  + wired into build_model_welded() (after armature loop) and build_model_coupled()
    (after damping loop). build_model() LEFT UNCHANGED (keeps URDF +/-pi placeholders).
scripts/derive_joint_limits.py  : the derivation sweeps (curl/aperture/base/passive).
scripts/validate_joint_limits.py: must-pass validation (limits applied, drive to both
    limits, aperture 0-130 mm, assembly inside limits, coupled compiles, base untouched).

## VALIDATION RESULTS (headless)
- --check --weld / --coupled / (base): all compile OK.
- validate_joint_limits.py: OVERALL PASS (exit 0)
    (1) all 15 jnt_range/jnt_limited set; 6 driven ctrlrange set -> all OK
    (2) drive 6 driven to MIN: finite, weld 0.020 mm; to MAX: finite, weld 0.034 mm
    (3) aperture: base->open 128.1 mm (~130), base->closed -0.0 mm (~0) -> PASS
    (4) assembly qpos0 inside all limits -> True (no reset snap)
    (5) coupled compiles, 15/15 limited, 100 steps finite
    base build_model() still at URDF +/-pi (derived limits NOT applied) -> True
- weld_validate_combined.py (pre-existing): still PASS, coupler separation UNCHANGED
    (F1 0.14/0.009, F2 0.32/0.11, F3 0.18/0.010 mm peak/settled) -> limits non-breaking.

## ESTIMATES / UNCERTAINTY (no invented precision)
- Aperture proxy choice (circumdiam-through-pad-centers minus 87.3 baseline) is a
  geometric model of "diameter the 3 jaws span"; the 87.3 mm baseline is the measured
  closed-pose value, so 0 and 130 are anchored to sim, but the exact mm at intermediate
  angles depends on the proxy. Open peaks at 128-129 mm (within 1-2 mm of 130).
- BASE open/closed limits are bounded by aperture geometry (turn-over), accurate to the
  sweep step (~0.05 rad). No self-collision occurs in range (verified), so no tighter
  collision bound exists; the +0.90 close could be pushed to the geometric minimum but
  +0.90 already gives 0 mm aperture so it's the natural close.
- CURL limits are the four-bar singularity (hard mechanical bound), bisected to ~0.05 rad,
  with a 0.05 rad safety margin so the servo never commands the dead-center.
- Velocity is recorded, not hard-enforced (MuJoCo limitation); see VELOCITY above.
