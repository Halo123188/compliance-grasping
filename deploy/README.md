# Deploying the camera-only grasp policy

Runs the trained student on the real Flexiv arm + the two-finger XC330 gripper
(`/home/yiboc/gripper`). The robot host needs `numpy`, `onnxruntime`, `pyserial`
and `pyrealsense2` — **not** mjlab, mujoco or torch. Training exports ONNX on
every checkpoint save, so the deployed weights are the evaluated weights.

```
deploy/
  calib.py            every BENCH constant, grouped by PROVENANCE
  profiles.py         what changes with the checkpoint, keyed by run name
  policy.py           ONNX session + the observation + action -> targets
  perception.py       D435 depth -> the exact (1,120,160) tensor trained on
  hand.py             sim radians <-> encoder counts, over the binary protocol
  arm.py              the Flexiv contract, FlexivArm (RDK) and ReplayArm
  run.py              the 50 Hz loop
  home_arm.py         drives the arm to the policy's home, before run.py
  calibrate_hand.py   measures the one map that cannot be derived
  check_camera.py     is the camera aimed where the policy thinks it is?
  live_view.py        the live D435 next to the sim's render, on localhost
  capture_sweep.py    a depth frame at each of several arm poses, for the
                      camera fit (screens every pose before it moves)
  kinematics.py       joint angles -> where the gripper is, numpy only
```

The robot host wants its own venv, without the simulator:

```sh
uv venv --python 3.12 .venv-deploy
uv pip install --python .venv-deploy/bin/python \
  "flexivrdk==1.9.0" numpy onnxruntime pyserial pyrealsense2
```

One exporter lives outside this tree, because it needs torch and mjlab and the
robot host has neither: `scripts/export_student_onnx.py` turns a bare
`model_N.pt` into a deployable ONNX. See "The August 18 arms".

`hand.py` looks for the gripper checkout at `/home/yiboc/gripper/firmware/host`;
set `GRIPPER_HOST_DIR` if yours is elsewhere.

## What the policy actually is

**Output: 11 joint POSITION targets**, not torques, not velocities, not deltas.

```
q_target[j] = default_joint_pos[j] + action_scale[j] * a[j]
```

| idx | joint | scale | default (rad) |
|---|---|---|---|
| 0–5 | joint1–6 (arm) | 0.40 | 0, −0.368, 0.217, 2.338, −0.180, 1.122 |
| 6 | joint7 (wrist roll) | 1.10 | 1.153 |
| 7, 9 | left_1, right_1 (proximal) | 0.35 (**0.45** on `R2-*`) | +0.2746, −0.2746 |
| 8, 10 | left_2, right_2 (distal) | 0.35 (**0.45** on `R2-*`) | 0.0, 0.0 |

There is no action clipping anywhere in the trained pipeline. Control rate is
50 Hz (sim timestep 5 ms × decimation 4).

**Input: two tensors.**

```
obs[0:11]    joint_pos − default_joint_pos     rad
obs[11:22]   joint_vel                         rad/s
obs[22:33]   the PREVIOUS step's raw action    network units, not the target
obs[33]      goal_height                       m above the FLOOR, 0.50–0.60
camera       (1,120,160) depth, clamp(m, 0.01, 3.0) / 3.0
```

The `R2-*` checkpoints read one more term — the object's half-edge — before
`goal_height`, which makes their frame 35 wide; see "Round 2" below. Which terms
this checkpoint reads is `profiles.Profile.obs_terms`, and `policy.py` assembles
the vector from that list rather than from a fixed four.

`obs[22:33]` is the raw network output, not the joint target it became. Feeding
the target back is dimensionally plausible and completely wrong.

## Which checkpoint, and what changes with it

Thirteen are deployable and `run.py` takes any of them. Five things differ
between checkpoints, all five fail silently when wrong, and none of them is an
operator setting any more: they live in `deploy/profiles.py` keyed by the run
that produced the weights, and the profile is **selected from the ONNX filename
and the graph** at load.

| checkpoint | profile | obs | fingers | travel | slew | cnn |
|---|---|---|---|---|---|---|
| `2026-08-06_17-29-41_s1_dr` | `v11-unlimited` | 34 | 0.35 | −0.0754 | **off** | 2/2/2 |
| `2026-08-06_20-53-44_s3_dr_fine` | `v11-unlimited` | 34 | 0.35 | −0.0754 | **off** | 2/2/2 |
| `2026-08-11_06-12-17_s_facelevel` | `v11` | 34 | 0.35 | −0.0754 | on | 2/2/2 |
| `2026-08-10_20-21-25_s_g1_relabel` | `v11` | 34 | 0.35 | −0.0754 | on | 2/2/2 |
| `2026-08-11_00-52-56_s_g1_hist` | `v11` | **136** | 0.35 | −0.0754 | on | 2/2/2 |
| `2026-08-12_18-20-34_d-R1-SatPen-All` | `r1` | 34 | 0.35 | −0.0754 | on | 2/2/2 |
| `2026-08-12_18-20-31_d-R1-FaceLevel-All` | `r1` | 34 | 0.35 | −0.0754 | on | 2/2/2 |
| `2026-08-12_18-20-48_d-R2-Reach` | `r2-reach` | **35** | **0.45** | **−0.32** | on | 2/2/2 |
| `2026-08-12_18-20-19_d-R2-Small` | `r2-small` | **35** | **0.45** | **−0.32** | on | 2/2/2 |
| `square_fixH_model_2999` | `square-fixh` | 34 | **0.45** | **−0.32** | on | 2/2/2 |
| `square_variableH_model_2999` | `square-varh` | 34 | **0.45** | **−0.32** | on | 2/2/2 |
| `cube_model_2999` | `cube` | 34 | **0.45** | **−0.32** | on | **1/2/2** |
| `everyShape_model_2800` | `everyshape` | **136** | **0.45** | **−0.32** | on | **1/2/2** |

Every one of them reads a **120×160** depth frame.

The five are the observation layout, the depth frame size, the finger action
scale, the finger travel limits and the command slew limit. Three of them are
corroborated against the file — the observation width and the camera shape come
off the graph, the action scale off the metadata, and any disagreement raises at
load — and the other two rest on the profile having named the right run, which
is why the run name is what selects it. `run.py` prints the
whole profile before anything moves; read that line rather than trusting the
filename.

**Round 3 broke the pattern the first two rounds set.** Through round 2, the
finger settings could be read off the observation width: 34-d meant 0.35 and
−0.0754, 35-d meant 0.45 and −0.32. The four round-3 arms are 34-d with 0.45 and
−0.32 — round 2's action term on a v11 frame, size-blind. So the layout does not
date a checkpoint, and a profile inferred from it would clamp these at −0.0754,
where the closure a 15 mm object needs cannot be commanded and nothing anywhere
reports a fault.

**Keep the exported name.** `<run>.onnx` inside `<run>/` is what identifies the
arm. A file renamed to `policy.onnx` falls back to shape alone, which cannot
tell `v11` from `v11-unlimited` (both 34-d at 0.35), so it silently gets the
*limited* profile — the safe direction, since limiting an unlimited policy makes
it lag while publishing a limited policy's raw request is a full-speed move into
the bench. Pass `--profile` to say which it really is.

`relabel` and `hist` are otherwise the same deployment as `facelevel`.
`relabel_achievable` changes the distillation loss (the student regresses onto
the teacher's request *after* its rate limiter clipped it) and is invisible from
here.

### Round 1 (`-All`): no code difference, a different bench

`R1-SatPen-All` and `R1-FaceLevel-All` are byte-for-byte the same deployment as
`s_g1_relabel` — same 34-d student, same scales, same limiter, same gains. What
changed is what they trained *against*, and those are claims about your bench:

- **cube 42.5–57.5 mm** (was 45–55), **mass 30–300 g** (was the nominal cube),
- **spawn box x 0.36–0.56, y ±0.12 m** in the base frame (was ±0.08/±0.10 about
  the manipulation centre, i.e. x 0.38–0.54, y ±0.10).

They are also, as far as the run metadata can say, the first checkpoints trained
**with the corrected camera extrinsic** — everything through `2026-08-11` was
trained against the old, 19.9 mm-off `CAM_POS_BASE`, so that fix applied to the
bench but not to the policy. The evidence is indirect and worth restating: the
runs record base commit `634f8db` with the training clone three commits *ahead*
of `origin/antipodal-gripper`, whose tip already carries
`arm_cfg.CALIB_CAM_OFFSET`, and their diff does not touch it. That is an
inference from a commit this checkout does not have, so confirm it against the
run tree before leaning on it.

### Round 2 (`R2-*`): the policy is told how big the object is

`R2-Reach` and `R2-Small` are a different deployment, in three coupled ways.

**The observation is 35 wide**, and the extra number sits *between* the actions
and `goal_height`:

```
obs[0:11]    joint_pos − default_joint_pos     rad
obs[11:22]   joint_vel                         rad/s
obs[22:33]   the PREVIOUS step's raw action    network units
obs[33]      cube_size — the object's HALF EDGE, metres     <- new
obs[34]      goal_height                       m above the FLOOR
```

Both of the last two terms are one number wide, so swapping them produces a
perfectly valid 35-vector and nothing downstream can tell. Two independent
confirmations that the order above is the real one. The baked normalizer's dim
33 has mean **0.0250** on `R2-Reach` and **0.0175** on `R2-Small` — the
midpoints of those two arms' half-extent ranges, in metres — while dim 34
carries the 0.549 / 0.039 that has been `goal_height` on every export. And
feeding the swapped vector on the bench frame takes `max|a|` from **0.39 to
356** on `R2-Reach` and from **0.21 to 42.5** on `R2-Small`, so `--max-action`
does catch it, but only after it has been flown.

**You supply the size.** `--cube-mm` is the cube's EDGE in mm, measured with a
caliper, and it defaults to `calib.CUBE_EDGE_MM` (50.0). It is an observation
through the same baked normalizer that turned an out-of-range `goal_height` into
a `|3641|` action, so it is range-checked hard: `R2-Reach` refuses anything
outside 45–55 mm and `R2-Small` outside 15–57.5 mm. The two mistakes to expect
are mm-for-metres and edge-for-half-edge; `run.py` takes the edge in mm and does
the halving, so the units live in one place.

**The fingers reach further and are commanded harder.** The proximal lower bound
opens from −0.0754 rad (40 mm of pad separation, a 10 mm squeeze on the 50 mm
cube) to **−0.32** (7.2 mm nominal — the fingers *cross* at −0.38), and the
finger action scale rises 0.35 → 0.45 so that closure stays about −1.3σ of the
policy's own exploration noise. That is why the size observation is required
rather than nice to have: the same full-close command that used to bottom out on
the cube becomes ~0.24 rad of over-travel on a 50 mm one, order 350 N.

On hardware there is no joint range to stop it — the sim's limit is a physical
stop, the robot just gets the command — so `policy.act()` clamps to the
checkpoint's own travel, `profiles.Profile.joint_limits`. **This also fixes the
older checkpoints**, which were being clamped at the URDF's ±1.6 rad on all four
finger joints rather than at the travel the training env set: proximal
−0.0754…+1.60 and distal −1.60…+0.10 for the left finger, mirrored as
[−hi, −lo] on the right because closing is left-negative and right-positive.
Expect the first R2 grasp to squeeze harder than the round-1 ones; the current
cap (`--grip-cap`) is still what bounds the force.

### The 136-d observation is not four stacked observations

`-hist` reads four control steps, and mjlab applies a group's `history_length`
to each **term**, so each term carries its own history and the terms are laid
out one after another:

```
obs[  0: 44]  joint_pos    at t−3, t−2, t−1, t
obs[ 44: 88]  joint_vel    at t−3, t−2, t−1, t
obs[ 88:132]  last_action  at t−3, t−2, t−1, t
obs[132:136]  goal_height  at t−3, t−2, t−1, t
```

Four whole 34-d observations end to end is the same 136 numbers in the wrong
order, and nothing downstream can tell: a mis-ordered 136-vector is a valid
136-vector. Fed the frame-major version on a moving sequence, this checkpoint
returns `max|a| = 301` against the correct layout's `0.42` — so `--max-action`
would in fact catch it, but only after it had been flown.

Two independent confirmations that the layout above is the real one. The
baked-in normalizer's last four dimensions share one mean (0.5492) and one std
(0.0391), which is `goal_height` four times over — frame-major would put a
single `goal_height` at dim 135 and three action dims beside it. And
`scripts/wide_onnx_parity.py` compares the assembled vector against the env's
own `student` group, which is what to run before flying it:

```sh
sbatch sbatch/wide_parity.sbatch \
  Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-SatPen-Smooth-Hist \
  model_2999.pt 2026-08-11_00-52-56_s_g1_hist.onnx
```

`relabel`'s task is the same id without the `-Hist`. The round-1 and round-2
arms are `Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-<arm>` for `<arm>` in
`R1-SatPen-All`, `R1-FaceLevel-All`, `R2-Reach`, `R2-Small`, each against its
own `model_2999.pt`. **All of them live in the run's own
`git/compliance-grasping.diff`, not on this branch** — the parity check needs
that tree checked out, since it builds the env the checkpoint trained in.

For the `R2-*` pair the parity check is the only thing that verifies the new
term end to end: it reads each env's cube half-extent the way `object_half_extent`
does and hands it to `deploy/policy.py`'s own `frame()`, so a wrong slot or a
wrong unit shows up as a large obs difference rather than as a policy that
merely grips badly.

The history is **backfilled** at reset — the first observation of a trial is one
pose repeated four times, not one pose behind three zero frames — because that
is what `CircularBuffer` does on its first push after a reset. So `policy.reset()`
between trials matters more for this checkpoint than for the others, and
`observe()` is stateful: one call advances the history by one step.

### Round 3 (August 18): four bare checkpoints

`square_fixH`, `square_variableH`, `cube` and `everyShape` arrived as bare
`model_N.pt` files — `student_state_dict`, an optimizer state and an iteration
count, with **no `params/` directory beside them**. Everything before them was
deployed from an ONNX that training itself exported out of the live env, so the
scene constants in the file were measured rather than asserted. These were not.

Export them with the offline exporter, which rebuilds the student from the
weights and takes the rest from the profile:

```sh
uv run python scripts/export_student_onnx.py \
  --checkpoint ~/yiboc/cube_model_2999.pt --profile cube --check
```

`--check` re-runs the export against the torch model on random inputs; it should
print a worst difference around 1e-6. **Keep the output name** — `profiles.select`
matches `square_fixH`, `square_variableH`, `cube_model` and `everyShape` in the
filename, and a renamed export of one of these is *refused* rather than guessed
at, because (34-d, 0.45) is four profiles and no fallback.

What the weights settle by themselves: the observation layout (34-d, no
`cube_size`), the history depth (`everyShape` alone stacks four control steps,
136-d), and the CNN's **feature grid** — 15×20 on the two `square` arms, 30×40
on `cube` and `everyShape`, read off the spatial-softmax coordinate buffers.

**What the grid does not settle is the stride, and that is the trap.** Same
padding makes each layer `ceil(dim / stride)`, so a 30×40 grid is a 120×160
frame at stride 1/2/2 *or* a 240×320 frame at 2/2/2 — and both rebuild, both
load with `strict=True`, both export cleanly. The first guess here was the
second reading, which would have shipped a network reading the world at half
the scale it was trained on, with nothing raising anywhere. The truth is stride
**1/2/2** on a 120×160 frame: round 3 stopped halving in the *first*
convolution, which doubles the resolution carried into the softmax off an
unchanged camera for about 4× that layer's arithmetic. So `Profile` now declares
`depth_hw` *and* `cnn_stride`, and the exporter checks the pair against the grid
— a real check only because neither half is derived from the other.

Confirm an export against the ONNX itself:

```sh
uv run python -c "
import onnx,sys
m=onnx.load(sys.argv[1])
print({i.name:[d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input})
print([list(a.ints) for n in m.graph.node if n.op_type=='Conv' for a in n.attribute if a.name=='strides'])
" ~/yiboc/cube_model_2999.onnx
```

should print `{'obs': [1, 34], 'camera': [1, 1, 120, 160]}` and
`[[1, 1], [2, 2], [2, 2]]`.

The camera work does not change at all: 848×480 stream, 640×480 centre crop —
that crop is where the trained field of view comes from — box-averaged by 4 to
120×160, exactly as before.

The action term came out of the runs' `env.yaml`, and it is **round 2's on a v11
observation**: fingers at **0.45**, travel out to **−0.32**, `use_default_offset`
with a zero offset, `ema_tau` null, and the slew limit at 1.0 / 1.5 / 3.0 rad/s
that `_LIMITED` already carried. Training scaled that limit down a curriculum
(3.0 → 2.0 → 1.5 → 1.0×) and all four reached the end of it, so the final values
are the ones to deploy.

| profile | objects | reset pose |
|---|---|---|
| `square-fixh` | upright cubes, 15–57.5 mm | fixed, +68 mm above the object |
| `square-varh` | upright cubes, 15–57.5 mm | +68 / +110 / +150 / +200 mm |
| `cube` | boxes, stretched in plan | fixed, +68 mm |
| `everyShape` | box / cylinder / sphere at 50/25/25 | fixed, +68 mm |

All four spawn over the same box (x 0.36–0.56, y ±0.12) at any yaw, and all four
draw size as a 25 mm nominal half-extent times U(0.30, 1.15), clamped to
7.5–32.5 mm — so the **height** is 15–57.5 mm everywhere. `cube` and
`everyShape` then stretch the horizontal plane: x and y each draw U(0.7, 1.6),
normalised so their geometric mean is 1, which leaves the footprint edges at
about 0.66–1.51× the height (plan aspect ratio p95 ≈ 1.76). A cylinder or a
sphere takes that on its radius and stays circular in plan, so what separates
`everyShape` from `cube` is the shape mix rather than the aspect.

`fixH` / `variableH` is the **reset pose**, not the goal-height observation — the
two runs' normalizers agree on `goal_height` to four decimals, so the same
`--goal-height` applies to both. `square-varh` draws its start from a solved
hover family at +68/+110/+150/+200 mm; the claw leaves the depth frame above
+143 mm, so roughly 46% of its training episodes begin with the claw not visible
to itself. **That is the one thing it buys, and the only reason to prefer it:**
it tolerates any start height in that band, where the other three expect +68 mm.
Only the reset pose moves — the action term's default offset stays put, because
lifting that too puts the descent at 1.51σ.

`--cube-mm` is advisory for all four: none of them observes the object's size,
and on `everyShape` the object need not be a box at all.

## What the two controllers are

- **Hand** — 4× DYNAMIXEL XC330-M181-T on a Teensy 4.0, **Current-Based Position
  Control (mode 5)**: goal position in encoder counts plus a current cap that
  bounds torque. The cap *is* the grip force. Synchronous ACTION → OBS over USB
  serial; the controller drops torque after 500 ms without a frame, so a crashed
  policy disarms rather than clamping on a stale goal.
- **Arm** — Flexiv Rizon 4S, 7 joints, through RDK's **NRT_JOINT_IMPEDANCE**:
  `SetJointImpedance(K_q)` with the sim's servo stiffness, then a new absolute
  `SendJointPosition` target every 20 ms. `arm.FlexivArm` is that; `ReplayArm`
  still lets everything else be tested without the arm.

  **Pin flexivrdk 1.9.0.** The version handshake is exact, not a floor. Probed
  against robot software **v3.11**: 1.9.1/1.9.2/1.9.3 all fail with "Version of
  this client is incompatible", 1.8.x fails to connect at all, and 2.1.0 refuses
  with "Product model [Rizon4s] is not supported". If the robot firmware is ever
  updated, re-probe rather than assuming the nearest version works.

  Two further caveats, both structural:
  - **Damping is not matched.** Sim `kd` is absolute (N·m·s/rad), RDK's `Z_q` is
    a ratio, and converting needs the joint-space inertia RDK does not expose.
    Only stiffness is set. If the arm rings at 50 Hz, look here before the policy.
  - **Sim and hardware saturate at the SAME torque.** This used to say the sim
    arm was unlimited, citing `forcerange` 0,0 on all seven joints — the wrong
    field. Each joint's `actfrcrange` is set from the sys-id fit to
    `[123, 123, 64, 64, 39, 39, 39]` N·m, i.e. exactly `info().tau_max`. So
    `max_offset_rad` is not a safety margin over the trained behaviour, it *is*
    the trained behaviour when set to `tau_max / K_q`. See below.

## Order of operations

1. **`--dry-run` first, on a desk.** Catches a metadata mismatch, a wrong
   observation width, a cube outside the trained size range and a missing
   calibration in a second. Read the `[profile]` block it prints: that is the
   checkpoint's own observation layout, action scale, finger travel and slew
   limit, and it is what the rest of the run depends on being right.
   ```sh
   uv run python -m deploy.run --dry-run --onnx <policy>.onnx
   # round 2, with the cube you will actually put on the bench:
   uv run python -m deploy.run --dry-run --onnx <policy>.onnx --cube-mm 50
   ```
2. **Parity check** (cluster, needs mjlab). Confirms the ONNX is numerically the
   checkpoint that was evaluated, and that `policy.py` assembles the same
   observation the env does.
   ```sh
   sbatch sbatch/wide_parity.sbatch <TASK> <model.pt> <policy.onnx>
   ```
3. **Calibrate the hand.** `calib.COUNTS_PER_RAD` / `COUNTS_AT_ZERO_RAD` are
   `None` and `hand.py` refuses to arm until they are filled.
   ```sh
   uv run python -m deploy.calibrate_hand --port /dev/ttyACM0
   # the scale, measured against a caliper rather than an aperture model:
   .venv-deploy/bin/python -m deploy.calibrate_finger_scale \
       --port /dev/ttyACM0 --method caliper
   ```
   Both want the motors parked at 0 counts first, which is the gripper repo's
   own CLI. **Its docs are written for a Mac** — `~/.platformio/penv/bin/python`
   does not exist on a Linux host, and its `find_port()` globs
   `/dev/cu.usbmodem*` and exits with "no Teensy serial port found". Use the
   deploy venv (it has pyserial) and name the port:
   ```sh
   .venv-deploy/bin/python ../gripper/firmware/host/gripper_ctl.py /dev/ttyACM0
   #   h  arm      : cmd -> `pos 1 1 0`   s  status      l  limp      q  quit
   ```
4. **Calibrate the camera along its optical axis** — see below. This is worth
   more than anything else on this list.
5. **Move the arm to the policy's home pose.** The pendant's "home" is Flexiv's
   default posture, not this one, and `run.py --require-home` refuses to start
   until the arm is within 0.10 rad of `calib.DEFAULT_JOINT_POS[0:7]`.
   ```sh
   .venv-deploy/bin/python -m deploy.home_arm --robot-sn Rizon4s-063501
   ```
6. **Arm alone first, slowly.** `--no-hand` runs the seven arm joints and
   discards the finger half of every action, so you can watch whether the arm
   reaches the grasp pose before the gripper is in the loop at all. `--speed`
   scales the three motion limits together and is the one knob to turn down
   when the arm moves faster than you want to stand next to.
   ```sh
   .venv-deploy/bin/python -m deploy.run \
     --onnx <policy>.onnx --robot-sn Rizon4s-063501 --no-hand --speed 0.5
   ```
   The fingers are reported to the policy as parked at home, which is what the
   network sees at the start of every sim episode — so the *reach* is
   representative and the *grasp* is not. The policy will close on the cube,
   nothing will close, and it will keep reaching.

7. **Run, with the hand.** Without `--robot-sn` the arm is the stationary stub
   even outside `--dry-run`, so it cannot move by accident.
   ```sh
   .venv-deploy/bin/python -m deploy.run \
     --onnx <policy>.onnx --robot-sn Rizon4s-063501 --gripper-port /dev/ttyACM0
   ```
   Exactly one of `--gripper-port` / `--no-hand` is required: on Linux the port
   cannot be auto-detected, so a forgotten flag would otherwise mean a silent
   hand-less run.

### Is the camera aimed where the policy thinks it is?

The depth image is the one input nothing else cross-checks. The joint vector is
verified by `--require-home`, the action feedback is the policy's own output,
and `goal_height` is range-checked. A camera mounted 180° out, or mirrored, or
aimed 10° off produces a perfectly plausible frame and a policy that reaches
confidently at the wrong place — which reads on the bench as "the policy is
bad".

```sh
# 1. clear the bench, capture the reference
.venv-deploy/bin/python -m deploy.check_camera --save baseline.npz
# 2. put the cube at a MEASURED spot, capture and compare
.venv-deploy/bin/python -m deploy.check_camera --baseline baseline.npz \
    --cube-xy 0.52 0.10
```

It projects the cube through the sim's own camera model and asks the sensor
where it actually sees it, then scores four hypotheses — as modelled, rotated
180°, mirrored left–right, mirrored up–down. Coordinates are the **robot base
frame** (the frame RDK reports `tcp_pose` in): x forward, y to the arm's left,
z up.

Two design points that are not incidental:

- **Difference against an empty bench**, don't hunt for "the nearest thing".
  The claw and the camera mount are in frame and at the home pose they are
  nearer than the cube.
- **The default cube position is off-centre on purpose.** At the scene's
  nominal (0.46, 0) the cube lands at u = 78.3 in a 160-wide frame, so a
  mirrored camera predicts u = 80.7 — 2.4 px away, which the measurement cannot
  resolve. At (0.52, 0.10) the nearest wrong hypothesis is 33 px away. The
  check says so rather than reporting a confident OK the geometry cannot
  support.

For reference, the sim camera sits at base-frame `(0.127, −0.006, 0.196)`
looking along `+x` and 28.494° down, with **image-right = world −y**. So a cube
moved 100 mm to the arm's left moves ~30 px *left* in the frame; if it moves
right, the y axis is inverted.

### Measuring the camera pose against the arm

`check_camera.py` answers "is it mounted the way the model says". For *how far
off*, use the arm: it is rigid, its joints are known to 3e-5 rad, and forward
kinematics agree with RDK's own tool pose to 1.4 mm.

```sh
.venv-deploy/bin/python -m deploy.capture_sweep --robot-sn Rizon4s-063501 \
    --out sweep.npz          # screens every pose, then moves; --dry-run to look
uv run python scripts/fit_camera_pose.py sweep.npz    # needs mjlab
```

Measured 2026-08-07, nine poses spanning 31.6° of joint1 and 0.37–0.48 m of
range: the camera sat **19.9 mm to the image-LEFT** of where `CAM_POS_BASE` put
it — base-frame y **+0.0142**, not −0.0057 — with rotation under 0.9° on every
axis and a residual of **1.64 mm** over 27 equations. Position alone fits to
1.79 mm, rotation alone to 3.44 mm and no correction at all leaves 11.69 mm, so
it is one lateral offset and not a rotation. (Image-right is world −y, so a
camera displaced toward +y makes everything appear shifted RIGHT in the real
frame, which is what the overlay showed.)

**This correction is applied**, as `arm_cfg.CALIB_CAM_OFFSET` and the
`CAM_POS_BASE` above — the same measurement in two frames, which must move
together. Re-running the fit against the same sweep now reports a residual of
**1.04 mm under the current extrinsic**, with a further six-parameter fit
buying 0.09 mm, i.e. there is nothing left to correct; per-pose lateral error
went from −15.9…−23.6 mm to −2.0…+0.4 mm. On `live_view` the best-fit shift of
real onto sim went from **du +7 px (19 mm sideways) to du +0 px**.

That is the size of error `arm_cfg.py`'s `_D435I_IR_LENS` comment warns about
("close enough for a nominal model … should be replaced by the extrinsics from
calibration") — the depth origin is the left IR imager and the two imagers are
50 mm apart. It is also **6.6× the ±3 mm of camera-position jitter the policy
was trained under** (`dr_cfg._CAM_POS_JITTER`), so the *current checkpoint was
trained against the old, wrong camera* and does not inherit this fix — it
applies from the next retrain. The ablation found this axis benign at ±3 mm
(90.6% vs 92.2%) and never measured further out.

One caveat the fit prints and this run tripped: the claw's silhouette is
**10% narrower in the real frame than in the render** (it was ~18% *wider* on an
earlier run, so this is not a stable property of the bench — the sensor preset
or the foreground-fattening behaviour changed between them). A width mismatch
biases a centroid-based displacement by an amount that depends on where in the
frame the claw sits, so it inflates the per-pose spread. Here it does not touch
the conclusion — the offset is constant to ±4 mm across a claw that traverses
u = 20…124 — but it is why the residual floors out at ~1 mm rather than at the
0.3 mm the joint encoders would allow.

**A joint1 zero-offset would look identical to the sweep.** Substituting
`p_base = R^T p_cam + cam_pos` into `dj*(z × p_base)` splits it exactly into a
camera rotation plus a camera translation, so it lies in the span of the
camera's own six numbers at *every* pose — singular by construction, not for
want of data. Correcting the camera is right for the arm either way, and right
for the table and the cube only if the camera really is the cause.

**Settled with the cube, which does not turn with joint1.** At (0.508, 0.102)
on the bare bench the two hypotheses predict pixels 5.1 px apart:

| hypothesis | predicts | measured error |
| --- | --- | --- |
| camera (this correction) | (57.9, 55.8) | **1.79 px** |
| joint1 zero-offset | (52.8, 56.2) | 4.30 px |
| mirrored up-down | (57.9, 63.2) | 8.92 px |
| mirrored left-right | (101.1, 55.8) | 44.45 px |
| rotated 180° | (101.1, 63.2) | 45.30 px |

The cube came back at (56.7, 54.4), 223 px, 0.407 m; the sim render of the same
scene puts it at (56.8, 54.0), 223 px, 0.412 m. **So the camera is the cause**,
and the correction is right for the whole scene rather than only for the arm.
The same sweep agrees on its own: a joint1-only fit needs `dj = +1.99°` and
still leaves 3.13 mm against camera-position's 1.79 mm, and with both free
joint1 collapses to +0.11° — and 2° is an absurd zero offset on encoders good
to 3e-5 rad.

One trap if you re-run this. `check_camera` locates the cube as the LARGEST
changed blob, so anything else that enters the scene between the baseline and
the cube frame can outvote it — on this run a 328 px object in the top-right
corner did exactly that and the script confidently reported
`THE CAMERA IS MIRRORED LEFT-RIGHT`, i.e. advised unbolting a correctly mounted
camera. Clear the bench, and read the raw `found a N-pixel change at (u, v)`
line rather than the verdict: a 50 mm cube at bench range is ~223 px, so a blob
far off that size is not the cube.

### The camera sets the loop rate

`wait_for_frames()` blocks until the sensor has a new frame, so the streaming
rate is a control-loop parameter, not a sensor setting. Measured on the bench at
the old 30 fps default: **44 ms per step — 23 Hz against the 50 Hz the policy
was trained at, with every single step late.** ONNX inference is not the
bottleneck (0.84 ms mean, 2.1 ms p99 on this checkpoint).

Two changes, both defaults now:

- **90 fps.** The 848×480 depth profile does up to 90. If the stream will not
  start, lower `--camera-fps`; never the resolution, which is what the trained
  field of view comes from.
- **A reader thread.** Even at 90 fps a blocking read couples the loop to the
  sensor's phase — finish 1 ms after a frame boundary and you wait a whole
  frame period — so the loop aliases down to a divisor of the frame rate. The
  thread keeps the latest frame in hand and `read()` returns immediately. A
  frame can be up to one frame period old (11 ms at 90 fps); the run summary
  counts how often the same frame was served twice, so that cost is measured.
  `--camera-blocking` restores the old behaviour for comparison.

Every run ends with a per-stage breakdown — camera / sensors / policy / command
— printed even after an abort, because the run that aborts is the one whose
timing you want. `late` says the loop slipped; this says which call did it.

### The floor guard, and why joint space could not provide one

**On 2026-08-07 the arm was driven into the foam.** With the speed cap removed
the pads dropped 213 mm in 0.52 s (0.41 m/s); `--max-jump` did not reach its
threshold until two steps *after* they were under the surface.

That is not bad tuning, it is the wrong quantity. `--max-jump` watches joint
tracking error, which only grows once the arm is *failing* to follow — an arm
that follows a bad target perfectly never trips it. Nothing in joint space can
express "do not go below the table".

`--floor-z` does. `kinematics.py` gives `deploy/` forward kinematics in numpy
(no mujoco on the robot host, checked against `mujoco.mj_kinematics` to 1e-6 m
on the recorded runs), and the guard compares

```
pad_height − descent_rate × --floor-lookahead   <   --floor-z
```

Height alone is also too late: at 0.4 m/s the pads cross the last 20 mm in one
step. Replayed against the three recorded runs at the 0.15 s default:

| run | fires at | pad height | clearance over foam | descending |
|---|---|---|---|---|
| run3 (the crash) | step 117 of 127 | 123 mm | **+88 mm** | 0.67 m/s |
| run2 | step 106 of 114 | 102 mm | +67 mm | 0.47 m/s |
| run1 (slow) | step 613 of 648 | 45 mm | +10 mm | 0.07 m/s |

The floor defaults to the foam top. In sim this policy never takes the pads
below 0.057 — 22 mm of clearance — so a floor at the surface cannot fire on
behaviour the policy was trained to produce.

It is checked on the **measured** pose, not the commanded target. The target is
an absolute joint goal the arm never reaches (the offset clamp holds it back),
so its forward kinematics sit far below anything the gripper visits: checking it
aborted run1 at step 5 of 648, in a run whose pads never went below the foam.

**The 0.15 s default is tuned for a slow arm, and a fast one trips it on every
approach.** The lookahead extrapolates the current descent rate as if it will
continue, which it does not — the policy decelerates into the grasp. At the
bring-up `--arm-max-vel 0.25` that cost nothing (2 trips in 20 runs). At
`--arm-max-vel 1.2` the approach descends at ~0.31 m/s, the lookahead subtracts
47 mm, and it fires at a pad height of ~79 mm: **13 of 15 runs on 2026-08-11
were stopped this way**, none of them anywhere near the foam (the lowest pad in
that batch was 50.7 mm, against a 35 mm floor and the sim's own 57 mm minimum).

Replaying all 15 at `--floor-lookahead 0.08` fires on none of them, and
replaying the 2026-08-07 crash at 0.08 still stops it **4 steps before the
breach** (0.15 s buys 7 steps, 0.05 s buys 3). So run a fast arm with
`--floor-lookahead 0.08`. The default is left at 0.15 until someone re-derives
it against a deceleration model rather than a constant-velocity one, because the
number that is right for a 0.3 m/s approach is not obviously right for a 0.7 m/s
dive.

### Torque authority is not a deployment choice

`--arm-max-offset` caps `|commanded − measured|`, so it caps the impedance
controller's torque at `K_q × offset`. It is tempting to read that as a safety
knob. **It is not** — the training environment is torque-limited too, and at the
same numbers. The wide-claw model sets each joint's `actfrcrange` from the
sys-id fit (`_SYSID_JOINTS` in `arm_cfg.py`) to `[123, 123, 64, 64, 39, 39, 39]`
N·m, which is exactly the robot's own `info().tau_max`. Sim and hardware
saturate identically.

So the offset that reproduces training is `tau_max / K_q` per joint, and that is
now the default, read off the robot rather than transcribed:

| | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|---|---|---|---|---|---|---|---|
| `tau_max/K_q` (rad) | 0.426 | 0.183 | 0.286 | 0.172 | 0.165 | 0.168 | 0.210 |
| a scalar 0.08 is | 19% | 44% | 28% | 47% | 49% | 48% | 38% |

A scalar below that is a **rate limit inside the policy's closed loop**: the arm
cannot reach the commanded target, the policy sees a stale pose and pushes
harder, then overshoots when it finally arrives. That is a limit cycle, not a
slower version of the trained behaviour, and `FlexivArm` now warns when the
configured offset is under 95% of full authority.

An earlier version of this file claimed the sim arm had no torque ceiling at
all. That read `actuator_forcerange` (0,0 on all seven) — the wrong field.

### Speed, and what it costs

Bring-up defaults are `--arm-max-vel 0.25` and `--arm-max-acc 0.6`, ramped in
over 2 s — roughly a tenth of the |dq| p99 2.3 rad/s this checkpoint runs at in
sim. That is a real change to the closed loop, not a comfort setting: a slower
arm is in a different state by the next inference than the policy expects.
Restore `--arm-max-vel 2.5 --arm-max-acc 6.0` before judging a success rate.
Note that `--speed` scales the offset too, so it throttles torque authority
along with speed — which is exactly the thing to be careful about above.

### Watching the real frame next to the sim's

The policy sees one depth frame and nothing else, so whether the real bench
looks like the trained bench is a question about that image.

```sh
# on the workstation (needs mjlab + a GPU)
uv run python scripts/render_sim_depth.py sim_ref.npz --cube-xy 0.5207 0.1016
# ... or, for the `cube` / `everyShape` checkpoints:
uv run python scripts/render_sim_depth.py sim_ref.npz --depth-hw 240,320 \
  --cube-xy 0.5207 0.1016
# on the robot host
.venv-deploy/bin/python -m deploy.live_view --sim sim_ref.npz   # localhost:8000
```

The **reference decides the resolution** — `live_view` reads it off the npz and
opens the camera to match, so the two panels are always the same picture.
`check_camera.py` stays at 120×160 whatever is being flown: it measures the
extrinsic and the field of view, neither of which depends on the sampling.

Four panels: real (live), sim (still — the scene does not move), a signed
difference, and an overlay. The difference is two colours, blue for real nearer
and red for real further, with a pixel valid in only one frame saturating to
whichever colour it is heading towards. The overlay is the one to read for
*registration*: sim goes on the red channel and real on green+blue, so
agreement is neutral grey and a misplaced gripper is a red ghost beside a cyan
one. A signed difference cannot show that — a shifted object makes a red band
and a blue band with nothing tying them together. The stats line gives the same
thing as a number: the whole-image shift that best fits real onto sim over the
gripper band. Binds to localhost only; it serves the robot's camera with no
authentication.

`--record run.npz` on `deploy/run.py` writes every step's joints, raw action,
commanded target and depth frame — including the aborting step — which is the
only way to see what the policy was looking at when it did something odd.

### Measuring the camera pose against the arm

The extrinsic in `check_camera.py` came from a plane fit, which can only see
three of its six numbers: a plane has no yaw and no position within itself. The
other three have to come from something with structure, and the arm is the only
such thing in frame.

```sh
# stop live_view first -- it holds the camera
.venv-deploy/bin/python -m deploy.capture_sweep --robot-sn Rizon4s-063501 \
    --out sweep.npz            # omit --yes to screen the poses without moving
uv run python scripts/fit_camera_pose.py sweep.npz
```

Two things this deliberately does not do. It does not solve for a joint1
offset: a joint1 error decomposes *exactly* into a camera rotation plus a camera
translation, so no arrangement of arm poses can separate the two — everything
joint1 moves is one rigid body, and a rigid body cannot say what it is rigid
with respect to. Pinning that down needs a feature that stays put when joint1
turns, which is what the cube in `check_camera.py` is for. And it does not
report a correction without first reporting the claw's silhouette **width** in
both frames: measured here the real claw comes back about 18% wider than the
rendered one at every pose, and until that is explained a displacement between
the two is not a pose error.

## Two numbers that decide whether this works

### Camera position along the optical axis: better than ~1.5 mm

Measured by re-scoring one trained checkpoint under narrower environments
(`scripts/wide_dr_ablate.py`), 128 envs, ±2.5 pt sampling noise:

| perturbation | depth bias | success |
|---|---|---|
| ±3 mm ⟂ to the optical axis | 0.006 mm | 90.6% |
| ±3 mm at 28° to it | 1.4 mm | 88.3% |
| ±3 mm along it | 2.6 mm | **67.2%** |
| none | 0 | 92.2% |

Depth is metric, so sliding the camera along its own view direction offsets
every reading, and the policy has no way to know the camera moved — it eats a
constant position error for the whole episode. Error *perpendicular* to the
optical axis is nearly free. It is a **threshold, not a proportionality**: the
knee sits between 1.4 mm and 2.6 mm.

This is a constant you can measure, so measure it and put the residual in
`calib.DEPTH_AXIS_OFFSET_M` rather than asking the policy to be robust to it.

### Grip force: the sim's finger is stronger than the real motor

Sim runs a position servo with `forcerange ±2.0 N·m` per finger joint (the
URDF's effort field). Real is an XC330-M181-T with 1.8 A stall at 5.0 V, shipped
at a 0.8 A cap. **Nothing in either repo relates those two quantities.** Until
pad force vs cap is measured on the bench, start at the firmware default and
expect the real grip to be the weaker of the two.

## Failure modes that do not raise

- **An out-of-distribution observation multiplies the output.** The normalizer
  is baked into the ONNX, so a bad input does not degrade the output gracefully.
  Measured: `goal_height = 0.0` instead of a value in [0.50, 0.60] takes the raw
  action from `|0.95|` to **`|3641|`**, because that term's training std is
  ~0.029 so an out-of-range value lands tens of sigma out. The output is an
  absolute joint target that goes straight to the arm.

  Three guards, all load-bearing. `policy.act()` clamps every target to
  `calib.JOINT_LIMITS`, which turns that blow-up into a saturated but legal
  command. `run.py --max-action` then stops the loop, judging the **raw action**
  rather than how far the target sits from the measured angle — a position servo
  legitimately lags while reaching, so that distance measures intent, not
  malfunction, and an earlier version of this guard aborted healthy runs on it.
  `--max-jump` is kept alongside it, on the arm joints only and at a threshold
  (1.2 rad) set from this checkpoint's own sim behaviour, because the two watch
  different failures: the raw action catches a blown-up *inference*, the jump
  catches an arm that has stopped following a perfectly reasonable one. Under
  `calib.RATE_LIMIT` the raw-action guard is the one that still fires in time —
  the limiter publishes at most 20 mrad of new arm motion per step, so a `|600|`
  action and a `|3|` one produce the same first command.
- **`GripperLink`'s port auto-detect is macOS-only** (`/dev/cu.usbmodem*`). On
  Linux the Teensy is `/dev/ttyACM0` and auto-detect fails with "no Teensy
  serial port found" while the board is plugged in. Pass `--gripper-port`. The
  same applies to the gripper repo's `gripper_ctl.py`, which additionally tells
  you to run it with `~/.platformio/penv/bin/python` — a Mac path. Name the port
  as its first argument and run it with `.venv-deploy/bin/python`.
- **Which physical finger is "left" is a measurement, not a convention.** Get it
  backwards and every asymmetric grasp mirrors, silently.
- **The distal joints are the weak part of the hand calibration.** Aperture is a
  shared observable that pins the proximal joints; nothing pins the distal ones.
  The policy leans on the distal curl heavily (~11× in pull-out resistance), so
  if grasps slip, distrust that map first.
- **The ONNX metadata's `observation_names` is wrong for a distillation export.**
  It lists the teacher's 43-d actor group, not the student's 34-d one, because
  `get_base_metadata` always reads `active_terms["actor"]`. `policy.py` checks
  the input shape instead.

## Hardware readiness — check before trusting any of this

`gripper/firmware/NEXT_STEPS.md` is dated **2026-07-13** and its status table is
at least partly stale (it calls `POS_MIN`/`POS_MAX` placeholders, but Config.cpp
says they were calibrated 2026-07-16). As of that table, still open:

- **5 V power path browns out under load** — called the blocker for the whole
  project. A force-limited grasp at 0.8 A × 4 motors through the measured ~1 Ω
  series resistance drops the rail below the XC330's 3.7 V minimum, and the
  motors reset *the moment you actually grip something*.
- **4th motor not fitted** (slot id 22 empty). `hand.py` refuses to run on a
  partial set.
- **The 200 Hz `Gripper` loop has never run on hardware**, and mode 5
  (current-based position — the mode this policy needs) has never run at all;
  all bench work was velocity mode.

None of that is visible from the code, so confirm the current state on the bench
before wiring the policy to it.
