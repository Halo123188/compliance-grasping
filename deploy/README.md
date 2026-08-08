# Deploying the camera-only grasp policy

Runs the trained student on the real Flexiv arm + the two-finger XC330 gripper
(`/home/yiboc/gripper`). The robot host needs `numpy`, `onnxruntime`, `pyserial`
and `pyrealsense2` — **not** mjlab, mujoco or torch. Training exports ONNX on
every checkpoint save, so the deployed weights are the evaluated weights.

```
deploy/
  calib.py            every constant, grouped by PROVENANCE
  policy.py           ONNX session + the 34-d observation + action -> targets
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
| 7, 9 | left_1, right_1 (proximal) | 0.35 | +0.2746, −0.2746 |
| 8, 10 | left_2, right_2 (distal) | 0.35 | 0.0, 0.0 |

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

`obs[22:33]` is the raw network output, not the joint target it became. Feeding
the target back is dimensionally plausible and completely wrong.

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
   observation width and a missing calibration in a second.
   ```sh
   uv run python -m deploy.run --dry-run --onnx <policy>.onnx
   ```
2. **Parity check** (cluster, needs mjlab). Confirms the ONNX is numerically the
   checkpoint that was evaluated, and that `policy.py` assembles the same 34-d
   observation the env does.
   ```sh
   sbatch sbatch/wide_parity.sbatch <TASK> <model.pt> <policy.onnx>
   ```
3. **Calibrate the hand.** `calib.COUNTS_PER_RAD` / `COUNTS_AT_ZERO_RAD` are
   `None` and `hand.py` refuses to arm until they are filled.
   ```sh
   uv run python -m deploy.calibrate_hand --port /dev/ttyACM0
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

The policy sees one 120×160 depth frame and nothing else, so whether the real
bench looks like the trained bench is a question about that image.

```sh
# on the workstation (needs mjlab + a GPU)
uv run python scripts/render_sim_depth.py sim_ref.npz --cube-xy 0.5207 0.1016
# on the robot host
.venv-deploy/bin/python -m deploy.live_view --sim sim_ref.npz   # localhost:8000
```

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

  Two guards, both load-bearing. `policy.act()` clamps every target to
  `calib.JOINT_LIMITS`, which turns that blow-up into a saturated but legal
  command. `run.py --max-action` then stops the loop, judging the **raw action**
  rather than how far the target sits from the measured angle — a position servo
  legitimately lags while reaching, so that distance measures intent, not
  malfunction, and an earlier version of this guard aborted healthy runs on it.
- **`GripperLink`'s port auto-detect is macOS-only** (`/dev/cu.usbmodem*`). On
  Linux the Teensy is `/dev/ttyACM0` and auto-detect fails with "no Teensy
  serial port found" while the board is plugged in. Pass `--gripper-port`.
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
