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
  arm.py              the Flexiv contract (interface only — RDK is not vendored)
  run.py              the 50 Hz loop
  calibrate_hand.py   measures the one map that cannot be derived
```

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
- **Arm** — Flexiv, 7 joints, through RDK. Not in either repo; `arm.py` pins the
  contract and `ReplayArm` lets everything else be tested without it.

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
5. Move the arm to the home pose, then run.

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
  is baked into the ONNX. Measured on this checkpoint: `goal_height = 0.0`
  instead of a value in [0.50, 0.60] takes `max|target|` from 2.6 rad to
  **668 rad** — because that term's training std is ~0.029, so an out-of-range
  value lands tens of sigma out. The output is an absolute joint target that
  goes straight to the arm. `run.py --max-jump` is the guard; do not remove it.
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
