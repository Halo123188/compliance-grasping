# Hands

One directory per gripper. `CG_HAND=<dirname>` selects which one the task builds.

```
hands/
  wide/            70 mm knuckle -- the claw on the robot
    hand.urdf        CAD: masses, COMs, full inertia tensors, joint frames
    meshes/*.stl     visual geometry; collision boxes are fitted to these
    PROVENANCE       which robot-actuator-model commit this came from
```

One gripper ships. A 56 mm claw preceded it and was removed on 2026-08-04 once
the wide claw was verified on hardware; what re-introducing a second hand takes
is at the bottom of this file.

The base plate and the static camera mount are **not** here — they are not part
of any hand and live in `../assets/`. They used to sit in this directory, which
made `_MESH_DIR` mean three different things at once.

## Dropping in a new or revised hand

The wide claw is read from its URDF at build time; nothing about its geometry or
inertia is transcribed into Python. That is the whole point — see
[docs/hand.md](../docs/hand.md) — and it is what makes a hand revision
a file copy rather than an editing session.

**To revise the wide hand in place** (a new CAD export of the same claw):

```bash
SRC=/path/to/robot-actuator-model
cp $SRC/urdf/two_finger_hand_wide.urdf src/cubegrasp_tf/assets/hands/wide/hand.urdf
cp $SRC/urdf/meshes/wide/*.stl         src/cubegrasp_tf/assets/hands/wide/meshes/
cp $SRC/urdf/meshes/{adapter,shim}.stl src/cubegrasp_tf/assets/hands/wide/meshes/
( cd $SRC && echo "source: $(git remote get-url origin)"; echo "commit: $(git rev-parse HEAD)" ) \
    > src/cubegrasp_tf/assets/hands/wide/PROVENANCE
```

**Always re-run these three after any hand change**, in order — each one
catches something the next assumes:

```bash
export PYTHONPATH=src MUJOCO_GL=egl CG_HAND=wide

# 1. does it still build, and what moved?
python scripts/check/check_cubegrasp_env.py

# 2. re-solve the arm poses. The pad frame moves with the claw, so the hover
#    and pre-grasp poses are NOT portable across a geometry change.
python scripts/calib/solve_home_pose.py 0.46
#    -> paste the emitted dicts into _*_BY_HAND in twofinger_arm_cfg.py

# 3. can the new hand actually grasp, lift and hold?
python scripts/expert/scripted_expert.py
```

Step 3 is the one that matters. A hand that cannot close on the cube produces a
reward curve identical to a policy that has not learned yet, and this project has
lost runs to exactly that.

## What a new hand needs from the code

`src/cubegrasp_tf/robots/hand_urdf.py` reads the URDF generically, but
three things are **claw-specific** and are worth checking against a new design
rather than inherited:

| | where | why it is not automatic |
|---|---|---|
| aperture law | `APERTURE_A`, `APERTURE_B` in `twofinger_arm_cfg.py` | FK-measured `sep(q) = a + b·q`; sets the open pose so full close still squeezes ~10 mm on the cube |
| horn cutoff | `fit_collision_boxes(horn_z=...)` | the linkage horn reaches further inward than the gripping face and must not inflate the collision box |
| pad marker | `pad_marker_pos(z_max=...)` | the contact patch is on the working face, not at the link's inner extremum |

All three defaulted correctly for the fitted claw. **All three were wrong on the
first attempt**, when carried over from the previous gripper unexamined, and none
of them failed loudly — the symptoms were "the claw stops 26 mm short", "the
aperture does not respond to the joint", and "the fingers stall against the
foam". `docs/hand.md` records each one.

## Adding a second hand

Create `hands/<name>/` with the same layout. Nothing hardcodes a hand name —
`CG_HAND` is validated by checking `hands/<name>/hand.urdf` exists — but four
constants are currently **scalars for the single gripper** and would become
per-hand tables keyed on `CG_HAND`:

| constant | what it is |
|---|---|
| `APERTURE_A`, `APERTURE_B` | the FK-measured `sep(q) = a + b·q` line |
| `FINGER_OPEN_ANGLE` | derived from the above; keeps a ~10 mm squeeze at full close |
| `_HOME_TARGETS`, `_SETTLED`, `_PREGRASP` | arm poses, re-solved per hand |
| `_MOUNT_OFFSET_HAND` | only if the flange interface differs — see the derivation in [docs/hand.md](../docs/hand.md), the one number CAD did not supply |

This was a per-hand table until 2026-08-04 and was collapsed when the second
gripper was retired; re-splitting it is mechanical.
