# RT joint-torque bridge

True `RT_JOINT_TORQUE` at 1 kHz for the route-A compliance policies, on a
Rizon 4S.

## Why a C++ process at all

The RDK **Python bindings deliberately exclude every real-time mode and the
Scheduler**. From the v1.x manual, API Overview:

| API | C++ | Python |
|---|---|---|
| `flexiv::rdk::Robot` | Full | Partial — *exclusion: functions marked "Real-time"* |
| `flexiv::rdk::Mode` | Full | Partial — *exclusion: real-time modes* |
| `flexiv::rdk::Scheduler` | Full | **None** |

Confirmed by probing every published wheel (1.6.0 → 2.1.0) at runtime: only
2.1.0 registers `RT_JOINT_TORQUE`, and 2.1.0 dropped the Rizon line entirely
(its `ProductModel` enum is `Enlight_L`/`Enlight_LL`/`MICO_*` only, and its
`Robot` ctor documents `"Enlight-L-123456"` serial numbers). The two never
overlap, so **no Python-only path to joint torque exists for this arm**.

Note that `strings` on the 1.9.x wheel *does* show `RT_JOINT_TORQUE` — those are
dead symbols from the bundled C++ library, never registered with pybind. An
earlier version of this repo's docstring drew the wrong conclusion from exactly
that check. Verify at runtime (`dir(flexivrdk.Mode)`), not with `strings`.

## Split of work

```
  C++ (this process, root)            Python (deploy_rizon_torque.py, user)
  ────────────────────────            ─────────────────────────────────────
  owns the RDK connection             FK, 37-D observation, ONNX GRU actor,
  SwitchMode(RT_JOINT_TORQUE)         safety guards, goal/episode logic
  Scheduler task @ 1 kHz:
    publish RobotStates       ──shm──▶  read state @ 100 Hz
    StreamJointTorque(tau)    ◀──shm──  write residual torque @ 100 Hz
```

Holding one policy torque across ten 1 kHz ticks **is** the sim's
`decimation=10`, so this split is faithful rather than an approximation. ONNX
inference measures ~0.02 ms (p99 0.018 ms) against a 10 ms budget, so Python has
no trouble keeping up.

## Build

The RDK version must match the robot's firmware. Ours reports `software_ver:
v3.11` → **RDK v1.9** (see the manual's compatibility table; v1.9.1/1.9.2 want
v3.11.1, v1.9.3 wants v3.11.2, and initialization fails on a mismatch).

```bash
git clone -b v1.9 --depth 1 https://github.com/flexivrobotics/flexiv_rdk.git
cd flexiv_rdk
bash thirdparty/build_and_install_dependencies.sh ~/rdk_install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/rdk_install -DCMAKE_PREFIX_PATH=~/rdk_install
cmake --build . --target install --config Release -j

# then this bridge
cd <repo>/src/mjlab/tasks/compliance/deploy/rt_bridge
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=~/rdk_install
cmake --build . -j
```

## Run

`rdk::Scheduler` needs RT thread priority, so the bridge **must run as root**
(flexiv_rdk README: "Root privilege is required if the real-time scheduler API
`flexiv::rdk::Scheduler` is used"). It hands the shared-memory segment back to
`$SUDO_UID` so the unprivileged policy process can attach.

```bash
# terminal 1 — the bridge (leave running)
sudo LD_LIBRARY_PATH=~/rdk_install/lib \
  ./build/rt_torque_bridge Rizon4s-063501 --enable

# terminal 2 — the policy
uv run python src/mjlab/tasks/compliance/deploy/deploy_rizon_torque.py \
  --policy tracking --backend shm
```

Home the arm first — `q_default` differs per policy and `joint_pos_rel` is
measured against it:

```bash
uv run python src/mjlab/tasks/compliance/deploy/home_rizon.py \
  --robot-sn Rizon4s-063501 --policy tracking
```

The bridge returns the robot to IDLE and exits when the policy finishes, so
start a fresh one per run.

### Without a robot

`--sim` runs the whole loop and IPC against a fake state fixed at the `tracking`
home pose (override with `--sim-q q1..q7`), and echoes back the torque it
*would* have streamed:

```bash
sudo ./build/rt_torque_bridge --sim
uv run python .../deploy_rizon_torque.py --policy tracking --backend shm --steps 200 --yes
```

## Safety behaviour

| situation | bridge response |
|---|---|
| before the first policy command | **holds the startup pose** under joint impedance (see below) |
| Python late by <50 ms | keeps streaming the last torque — same as decimation |
| Python silent >50 ms | linear fade to zero over 100 ms (`--cmd-timeout-ms`, `--fade-ms`) |
| Python silent >150 ms | zero residual → pure gravity compensation, arm floats |
| bridge stops ticking | Python raises after 5 stale reads |
| robot faults | scheduler stops, back to IDLE, fault flag published |
| Ctrl-C / stop flag | scheduler stops, `Stop()`, back to IDLE |

### Position hold before handover

`RT_JOINT_TORQUE` with a zero residual is **zero stiffness**. Gravity
compensation cancels gravity but holds no position, so the arm floats and any
imbalance walks it away. Measured on the real robot: the EE drifted 0.19 m while
an operator answered the deploy script's confirmation prompt, and after a few
minutes of floating the arm sat ~180 deg from its home pose — far enough that
`home_rizon.py` refused to drive it back.

So the bridge captures `q` on its first tick and runs a joint impedance about it
until the policy's first command lands, then crossfades to the policy over
`--release-ms` (default 200 ms) so the handover is not a torque step. Gains are
the RDK's own demo values, clamped to `--hold-tau-frac` of `tau_max`
(default 0.5).

### Locking a joint out of the policy's control

`--lock-joints 7` pins joint 7 to its startup angle for the whole run, under the
same impedance controller, and **discards** the policy's torque for it. The other
joints follow the policy normally. Comma or space separated, 1-based to match the
`j1..j7` labels in the deploy script's guard dump:

```bash
sudo ./build/rt_torque_bridge Rizon4s-063501 --enable --lock-joints 7
```

Why this exists: the `tracking` policy saturates joint 7 at its negative torque
limit (`tanh(a7) = -1`, i.e. -39 Nm) from the home pose, identically at every
checkpoint from iter 3000 to 4999. With `armature=0.13` on the wrist -- 25x less
than the shoulder's 3.17 -- that reaches the 50%-of-`dq_max` speed guard in a
fraction of a second. Locking joint 7 lets the rest of the arm be evaluated while
that is investigated separately. Note the policy still *sees* the locked joint's
state and still emits an action for it (`a_prev` stays the raw action, as in
training), so its observation is no longer what it would be in sim -- this is a
diagnostic, not a fix.

### Changing the lock set at runtime (layout v2)

`--lock-joints` is fixed for the life of the process, which is fine for a policy
run but wrong for system identification: that has to test one joint at a time with
the other six held, and has to park each joint away from its stops first. Doing
that with `--lock-joints` alone meant restarting this process seven times with a
different list and a different startup pose each time.

So a client can instead take over the lock set through the command block: a lock
mask and a hold target ride along with every torque, inside the same seqlock, and
override both `--lock-joints` and the startup pose. `identify_rizon.py` uses this
to sweep all seven joints against one bridge instance.

The bridge does not trust what arrives. The target it actually tracks slews toward
the requested one at `--hold-slew` rad/s (default 0.30) and is clamped 0.10 rad
inside the robot's own `q_min`/`q_max`, so the worst a client bug can do is ask for
a legal pose and get there slowly — not step a stiff joint or drive one into a
stop. A client that writes nothing leaves the override off and gets exactly the
old behaviour, which is why `deploy_rizon_torque.py` needed no change.

Bumping the layout to v2 means **an un-rebuilt bridge is rejected** rather than
silently ignoring the new fields and leaving six joints unheld:

```
layout mismatch: bridge speaks v1, this client v2. Rebuild rt_torque_bridge.
```

**The startup hold makes the arm stiff at startup.** If you want to hand-push it to check
the compliance feel, use `--no-hold` (and do not leave it unattended) or soften
with `--hold-scale 0.1`.

Gravity compensation is done **on-robot** (`enable_gravity_comp=true`), so the
faded state is a floating arm, not a falling one. Soft joint limits are on by
default; `--no-soft-limits` exists but the RDK warns it lets the arm drive into
a position fault.

## Caveat: this host is not a real-time kernel

`uname` reports `PREEMPT_DYNAMIC`, not `PREEMPT_RT`. Per Flexiv's guidance that
means *soft* real-time: whole-loop latency 4–10 ms and less deterministic,
versus ~2 ms on an RT kernel. The robot faults once more than
`--timeliness-limit` percent of commands arrive late (default 2%). Watch that
figure. Installing Ubuntu's realtime kernel (needs Ubuntu Pro) is the fix if
1 kHz fidelity matters; it is orthogonal to everything above.
