/**
 * @file rt_torque_bridge.cpp
 * @brief RT joint-torque shim for the compliance policies (route A, "plan B").
 *
 * Why this exists
 * ---------------
 * The Flexiv RDK Python bindings deliberately exclude every real-time mode and
 * the Scheduler (v1.x manual, API Overview: `Mode` is "Partial (exclusion:
 * real-time modes)", `Scheduler` is "None").  So `RT_JOINT_TORQUE` is reachable
 * only from C++.  The policy itself, however, is a 100 Hz GRU whose ONNX
 * inference costs ~0.02 ms -- there is no reason to rewrite its forward
 * kinematics, observation assembly and safety guards in C++ just to reach the
 * RT layer.
 *
 * This process therefore owns *only* the RT half:
 *
 *   - holds the single RDK connection,
 *   - runs a 1 kHz Scheduler task at max priority,
 *   - each tick: publishes `RobotStates` and streams the latest residual torque
 *     via `StreamJointTorque(tau, gravity_comp, soft_limits)`.
 *
 * Python attaches to the same shared memory, reads state at 100 Hz, runs the
 * ONNX actor, and writes back a residual torque.  Holding one torque across ten
 * 1 kHz ticks is exactly the sim's `decimation=10`, so the decoupling is
 * faithful rather than an approximation.
 *
 * Degradation
 * -----------
 * If Python stalls, this loop keeps streaming the last commanded torque -- the
 * same semantics as decimation, just with a staler command -- so no cycle is
 * missed and the robot's timeliness monitor stays happy.  Past
 * `--cmd-timeout-ms` the residual is ramped to zero over `--fade-ms`, leaving
 * the arm on pure gravity compensation (it floats in place) rather than holding
 * a stale torque indefinitely.  A fault on the robot, or the Python side
 * setting the stop flag, stops the scheduler and returns to IDLE.
 *
 * Build: see CMakeLists.txt in this directory.
 * Test without a robot: --sim (runs the full loop and IPC, no RDK connection).
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <flexiv/rdk/robot.hpp>
#include <flexiv/rdk/scheduler.hpp>
#include <spdlog/spdlog.h>

using namespace flexiv;

namespace {

constexpr int kNumJoints = 7;
constexpr size_t kShmSize = 4096;
constexpr const char* kShmName = "/flexiv_rt_bridge";
constexpr uint64_t kMagic = 0x464C58565254'3031ULL;  // "FLXVRT01"
constexpr uint64_t kLayoutVersion = 2;

/**
 * Hard backstop on any runtime hold target Python asks for: never track a target
 * closer than this to a joint stop, whatever the other side says.  Python's
 * identification script keeps a much wider margin of its own; this one exists so
 * that a bug there still cannot drive a joint into its limit.
 */
constexpr double kHoldLimitMargin = 0.10;  // rad

// Offsets -- mirrored from shm_layout.py.  Keep the two in lockstep.
constexpr size_t kOffMagic = 0x000, kOffVersion = 0x008, kOffInfoReady = 0x010,
                 kOffDoF = 0x018, kOffQMin = 0x020, kOffQMax = 0x058, kOffDqMax = 0x090,
                 kOffTauMax = 0x0C8, kOffKqNom = 0x100;
constexpr size_t kOffStateSeq = 0x400, kOffQ = 0x408, kOffDq = 0x440, kOffTau = 0x478,
                 kOffTauExt = 0x4B0, kOffStateTick = 0x4E8, kOffLateCount = 0x4F0,
                 kOffFault = 0x4F8;
constexpr size_t kOffCmdSeq = 0x800, kOffTauCmd = 0x808, kOffCmdEpoch = 0x840,
                 kOffStopFlag = 0x848, kOffLockOverride = 0x850, kOffLockMask = 0x858,
                 kOffQHold = 0x860;

constexpr size_t kVec = sizeof(double) * kNumJoints;
static_assert(kOffQMin + 5 * kVec <= kOffStateSeq, "header block overruns state block");
static_assert(kOffTauExt + kVec <= kOffStateTick, "state block overruns its counters");
static_assert(kOffTauCmd + kVec <= kOffCmdEpoch, "command block overruns its counters");
static_assert(kOffStopFlag + sizeof(uint64_t) <= kOffLockOverride, "stop flag overruns");
static_assert(kOffLockOverride + sizeof(uint64_t) <= kOffLockMask, "override overruns");
static_assert(kOffLockMask + sizeof(uint64_t) <= kOffQHold, "lock mask overruns");
static_assert(kOffQHold + kVec <= kShmSize, "command block overruns the mapping");

uint8_t* g_shm = nullptr;
std::atomic<bool> g_stop_sched{false};
std::atomic<bool> g_signalled{false};

// Accessors. Every field is 8-byte aligned, so these are single atomic
// loads/stores on x86-64; the seqlock protects the grouping.
inline std::atomic<uint64_t>* U64(size_t off)
{
    return reinterpret_cast<std::atomic<uint64_t>*>(g_shm + off);
}
inline double* Vec(size_t off) { return reinterpret_cast<double*>(g_shm + off); }

inline void WriteVec(size_t off, const std::vector<double>& v)
{
    double* dst = Vec(off);
    const size_t n = std::min<size_t>(v.size(), kNumJoints);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = v[i];
    }
    for (size_t i = n; i < kNumJoints; ++i) {
        dst[i] = 0.0;
    }
}

void HandleSignal(int) { g_signalled = true; }

/** Shared state between main() and the periodic task. */
struct BridgeCtx {
    rdk::Robot* robot = nullptr;  // null in --sim
    bool gravity_comp = true;
    bool soft_limits = true;
    double cmd_timeout_s = 0.05;
    double fade_s = 0.10;
    uint64_t tick = 0;
    uint64_t last_cmd_epoch = 0;
    uint64_t last_cmd_tick = 0;
    double tau_hold[kNumJoints] = {0};

    // Position hold, active until the policy's first command.
    //
    // RT_JOINT_TORQUE with a zero residual is *zero stiffness*: gravity
    // compensation cancels gravity but provides no position holding, so the arm
    // floats and any small imbalance walks it away.  Observed on the real robot:
    // the EE drifted 0.19 m in the seconds it took an operator to answer the
    // deploy script's confirmation prompt, and the arm ended up ~180 deg from
    // its home pose after a few minutes of this.  So until the first policy
    // torque arrives we run a joint impedance about the pose captured at
    // startup.  Gains are the RDK's own demo values from
    // intermediate3_realtime_joint_torque_control.cpp.
    bool hold_enabled = true;
    bool have_first_cmd = false;
    bool q_init_set = false;
    double q_init[kNumJoints] = {0};
    double hold_kp[kNumJoints] = {3000.0, 3000.0, 800.0, 800.0, 200.0, 200.0, 200.0};
    double hold_kd[kNumJoints] = {80.0, 80.0, 40.0, 40.0, 8.0, 8.0, 8.0};
    double hold_scale = 1.0;
    /** Fraction of tau_max the hold may ever command. */
    double hold_tau_frac = 0.5;
    double tau_max[kNumJoints] = {123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0};
    /** Crossfade from hold to policy, so the handover is not a torque step. */
    double release_s = 0.20;
    uint64_t first_cmd_tick = 0;
    /** Joints pinned to q_init for the whole run; the policy cannot drive them. */
    bool locked[kNumJoints] = {false};
    bool any_locked = false;

    // Runtime lock override (layout v2).  System identification has to test one
    // joint at a time with the other six held, and has to park each joint away
    // from its stops first -- j4 sits 5 deg under its upper limit at the tracking
    // home and has no room to be excited there.  Both of those used to mean
    // restarting this process per joint, with a different --lock-joints list and
    // a different startup pose, so a seven-joint sweep was seven manual restarts.
    // With the override the lock set and the hold target arrive in the command
    // block instead, and one bridge instance covers the whole sweep.
    //
    // The target Python sends is a *request*.  What gets tracked is
    // `q_hold_eff`, which walks toward the request at `hold_slew` rad/s and is
    // clamped inside the robot's own joint limits.  So the worst a bug on the
    // Python side can do is ask for a legal pose and get there slowly.
    bool override_active = false;
    bool locked_rt[kNumJoints] = {false};
    double q_hold_req[kNumJoints] = {0};
    double q_hold_eff[kNumJoints] = {0};
    bool q_hold_eff_set = false;
    double hold_slew = 0.30;  // rad/s
    double q_min[kNumJoints] = {0};
    double q_max[kNumJoints] = {0};
    bool have_limits = false;

    // --sim only: a fake arm state so the IPC path can be exercised.  Defaults
    // to the `tracking` policy's home pose (PolicySpec.q_default) rather than
    // zeros, so the deploy script's keep-in guards pass and the command path
    // actually gets tested.  Override with --sim-q.
    double sim_q[kNumJoints]
        = {-0.225705, 0.003626, 0.507476, 2.438249, 0.002759, 0.864276, -1.290931};
    /** --sim only: drift rad/s added to sim_q, to exercise the position hold. */
    double sim_drift = 0.0;
    /** --sim only: the fake plant's velocity, and last tick's streamed torque. */
    double sim_dq[kNumJoints] = {0};
    double sim_tau_last[kNumJoints] = {0};
};

// --sim plant constants: order-of-magnitude figures for a Rizon 4S, present so
// that --sim responds to torque at all and a caller's park/lock/excite sequence
// can be tested end to end without a robot.  These are NOT identified values and
// nothing should be concluded from --sim torques.
constexpr double kSimInertia[kNumJoints] = {4.4, 4.4, 2.2, 2.2, 0.2, 0.2, 0.15};
constexpr double kSimDamping[kNumJoints] = {1.5, 1.5, 1.0, 1.0, 0.3, 0.3, 0.25};
constexpr double kSimFriction[kNumJoints] = {1.6, 1.9, 2.5, 2.5, 0.44, 1.0, 0.4};

/**
 * Read the freshest command via the seqlock. Returns false if no *new* command
 * has arrived (the caller then keeps holding the previous torque, which is the
 * decimation behaviour) or if every retry tore.
 */
bool TryReadCommand(BridgeCtx& ctx, double out[kNumJoints])
{
    for (int attempt = 0; attempt < 4; ++attempt) {
        const uint64_t s0 = U64(kOffCmdSeq)->load(std::memory_order_acquire);
        if (s0 & 1ULL) {
            continue;  // writer mid-update
        }
        double tmp[kNumJoints];
        double hold_tmp[kNumJoints];
        std::memcpy(tmp, Vec(kOffTauCmd), kVec);
        std::memcpy(hold_tmp, Vec(kOffQHold), kVec);
        const uint64_t over = U64(kOffLockOverride)->load(std::memory_order_relaxed);
        const uint64_t mask = U64(kOffLockMask)->load(std::memory_order_relaxed);
        const uint64_t epoch = U64(kOffCmdEpoch)->load(std::memory_order_relaxed);
        const uint64_t s1 = U64(kOffCmdSeq)->load(std::memory_order_acquire);
        if (s0 != s1) {
            continue;  // torn
        }
        if (epoch == ctx.last_cmd_epoch) {
            return false;  // valid, but nothing new
        }
        ctx.last_cmd_epoch = epoch;
        std::memcpy(out, tmp, kVec);
        // The override travels with the torque through the same seqlock, so the
        // lock set can never be one generation out of step with the command it
        // was meant to accompany.
        ctx.override_active = (over != 0ULL);
        if (ctx.override_active) {
            for (int i = 0; i < kNumJoints; ++i) {
                ctx.locked_rt[i] = ((mask >> i) & 1ULL) != 0ULL;
                ctx.q_hold_req[i] = hold_tmp[i];
            }
        }
        return true;
    }
    return false;
}

void PublishState(const std::vector<double>& q, const std::vector<double>& dq,
    const std::vector<double>& tau, const std::vector<double>& tau_ext, uint64_t tick,
    bool fault)
{
    auto* seq = U64(kOffStateSeq);
    const uint64_t s = seq->load(std::memory_order_relaxed);
    seq->store(s + 1, std::memory_order_release);  // odd: write in progress
    WriteVec(kOffQ, q);
    WriteVec(kOffDq, dq);
    WriteVec(kOffTau, tau);
    WriteVec(kOffTauExt, tau_ext);
    U64(kOffStateTick)->store(tick, std::memory_order_relaxed);
    U64(kOffFault)->store(fault ? 1 : 0, std::memory_order_relaxed);
    seq->store(s + 2, std::memory_order_release);  // even: consistent
}

/** The 1 kHz task. Must not allocate, log, or block. */
void PeriodicTask(BridgeCtx& ctx)
{
    try {
        if (U64(kOffStopFlag)->load(std::memory_order_acquire) != 0) {
            g_stop_sched = true;
            return;
        }

        std::vector<double> q, dq, tau, tau_ext;
        bool fault = false;

        if (ctx.robot != nullptr) {
            fault = ctx.robot->fault();
            const auto& st = ctx.robot->states();
            q = st.q;
            dq = st.dq;
            tau = st.tau;
            tau_ext = st.tau_ext;
        } else {
            // Integrate last tick's streamed torque against the fake plant.  One
            // tick of delay is what the real robot has too, so the caller sees a
            // faithful-enough loop to test its own state machine against.
            for (int i = 0; i < kNumJoints; ++i) {
                const double v = ctx.sim_dq[i];
                const double s = (v > 0.01) ? 1.0 : ((v < -0.01) ? -1.0 : 0.0);
                double net = ctx.sim_tau_last[i] - kSimDamping[i] * v
                             - kSimFriction[i] * s;
                if (s == 0.0 && std::fabs(ctx.sim_tau_last[i]) <= kSimFriction[i]) {
                    net = 0.0;  // stiction: below breakaway nothing moves
                }
                ctx.sim_dq[i] = v + 0.001 * net / kSimInertia[i];
                ctx.sim_q[i] += 0.001 * ctx.sim_dq[i] + ctx.sim_drift * 0.001;
            }
            q.assign(ctx.sim_q, ctx.sim_q + kNumJoints);
            dq.assign(ctx.sim_dq, ctx.sim_dq + kNumJoints);
            tau_ext.assign(kNumJoints, 0.0);
        }

        if (!ctx.q_init_set) {
            std::memcpy(ctx.q_init, q.data(), kVec);
            ctx.q_init_set = true;
        }

        if (fault) {
            PublishState(q, dq, tau, tau_ext, ctx.tick, true);
            g_stop_sched = true;
            return;
        }

        double fresh[kNumJoints];
        if (TryReadCommand(ctx, fresh)) {
            std::memcpy(ctx.tau_hold, fresh, kVec);
            ctx.last_cmd_tick = ctx.tick;
            if (!ctx.have_first_cmd) {
                ctx.have_first_cmd = true;
                ctx.first_cmd_tick = ctx.tick;
            }
        }

        // Staleness fade: hold, then ramp to zero, then pure gravity comp.
        const double stale_s = static_cast<double>(ctx.tick - ctx.last_cmd_tick) * 0.001;
        double scale = 1.0;
        if (stale_s > ctx.cmd_timeout_s) {
            const double over = stale_s - ctx.cmd_timeout_s;
            scale = (ctx.fade_s <= 0.0) ? 0.0 : std::max(0.0, 1.0 - over / ctx.fade_s);
        }

        std::vector<double> tau_cmd(kNumJoints);
        for (int i = 0; i < kNumJoints; ++i) {
            tau_cmd[i] = ctx.tau_hold[i] * scale;
        }

        // Resolve which joints are held and about what angle.  Without the
        // runtime override this is exactly the old behaviour: --lock-joints,
        // pinned to the pose captured at startup.
        const bool* locked = ctx.override_active ? ctx.locked_rt : ctx.locked;
        bool any_locked = ctx.any_locked;
        if (ctx.override_active) {
            any_locked = false;
            for (int i = 0; i < kNumJoints; ++i) {
                any_locked = any_locked || ctx.locked_rt[i];
            }
        }

        // Walk the tracked target toward the requested one.  Slewing rather than
        // jumping is what makes a runtime target safe: the PD gains here are stiff
        // (3000 Nm/rad on the shoulder), so assigning a target 30 deg away would
        // be a torque step into the cap, whereas at hold_slew rad/s the joint is
        // driven as a position move.
        const double* q_hold = ctx.q_init;
        if (ctx.override_active) {
            if (!ctx.q_hold_eff_set) {
                std::memcpy(ctx.q_hold_eff, q.data(), kVec);
                ctx.q_hold_eff_set = true;
            }
            const double step = ctx.hold_slew * 0.001;
            for (int i = 0; i < kNumJoints; ++i) {
                double want = ctx.q_hold_req[i];
                if (ctx.have_limits) {
                    // Never track a target near a stop, whatever Python asked for.
                    const double lo = ctx.q_min[i] + kHoldLimitMargin;
                    const double hi = ctx.q_max[i] - kHoldLimitMargin;
                    want = (lo <= hi) ? std::max(lo, std::min(hi, want))
                                      : 0.5 * (ctx.q_min[i] + ctx.q_max[i]);
                }
                const double d = want - ctx.q_hold_eff[i];
                ctx.q_hold_eff[i] += std::max(-step, std::min(step, d));
            }
            q_hold = ctx.q_hold_eff;
        } else {
            ctx.q_hold_eff_set = false;  // re-seed from q if the override comes back
        }

        // Position hold. Two users share it:
        //   - the startup hold, crossfading into the policy over release_s;
        //   - the lock set, which pins chosen joints to q_hold *permanently*,
        //     ignoring whatever the policy asks of them.
        // `a` is the policy's weight: 0 before the first command, ramping to 1
        // over release_s once it lands.
        const bool fading = ctx.hold_enabled
                            && !(ctx.have_first_cmd && ctx.release_s <= 0.0);
        if (fading || any_locked) {
            double a = 0.0;
            if (ctx.have_first_cmd) {
                const double since
                    = static_cast<double>(ctx.tick - ctx.first_cmd_tick) * 0.001;
                a = (ctx.release_s <= 0.0)
                        ? 1.0
                        : std::min(1.0, since / ctx.release_s);
            }
            for (int i = 0; i < kNumJoints; ++i) {
                if (!locked[i] && (!fading || a >= 1.0)) {
                    continue;
                }
                double h = ctx.hold_scale
                           * (ctx.hold_kp[i] * (q_hold[i] - q[i])
                               - ctx.hold_kd[i] * dq[i]);
                const double lim = ctx.hold_tau_frac * ctx.tau_max[i];
                h = std::max(-lim, std::min(lim, h));
                // A locked joint never blends: the policy has no say over it.
                tau_cmd[i] = locked[i] ? h : h * (1.0 - a) + tau_cmd[i] * a;
            }
            if (fading && a >= 1.0) {
                ctx.hold_enabled = false;  // handover complete
            }
        }

        if (ctx.robot != nullptr) {
            ctx.robot->StreamJointTorque(tau_cmd, ctx.gravity_comp, ctx.soft_limits);
        } else {
            // --sim: echo back what *would* have been streamed (post-fade), so
            // the watchdog is observable without a robot, and feed it to the fake
            // plant on the next tick.
            tau = tau_cmd;
            std::memcpy(ctx.sim_tau_last, tau_cmd.data(), kVec);
        }

        // Published after the command is resolved so `tau` reflects this tick.
        PublishState(q, dq, tau, tau_ext, ctx.tick, false);

        ctx.tick++;
    } catch (const std::exception& e) {
        spdlog::error("PeriodicTask: {}", e.what());
        g_stop_sched = true;
    }
}

void PrintHelp()
{
    std::cout
        << "Usage: rt_torque_bridge [robot_sn] [options]\n"
           "  robot_sn              e.g. Rizon4s-063501 (omit only with --sim)\n"
           "  --sim                 no robot: run the loop + IPC against a fake state\n"
           "  --no-gravity-comp     disable on-robot gravity compensation\n"
           "  --no-soft-limits      disable soft joint limits (NOT recommended)\n"
           "  --timeliness-limit P  robot's tolerated %% of late commands (default: robot's)\n"
           "  --cmd-timeout-ms N    hold a stale command this long before fading (default 50)\n"
           "  --fade-ms N           fade-to-zero duration once stale (default 100)\n"
           "  --enable              enable the robot (release brakes) if not operational\n"
           "  --sim-q q1..q7        --sim only: fake joint angles (default: tracking home)\n"
           "  --sim-drift R         --sim only: rad/s drift on every joint, to test the hold\n"
           "  --no-hold             do NOT hold position before the first policy command\n"
           "  --hold-scale S        scale the hold gains (default 1.0)\n"
           "  --hold-tau-frac F     cap the hold at this fraction of tau_max (default 0.5)\n"
           "  --release-ms N        crossfade hold -> policy over this long (default 200)\n"
           "  --lock-joints LIST    pin these joints (1-7, comma/space list) to their\n"
           "                        startup angle for the whole run; the policy's torque\n"
           "                        for them is discarded.  e.g. --lock-joints 7\n"
           "  --hold-slew R         max rad/s at which a *runtime* hold target is\n"
           "                        tracked (default 0.30).  Only applies when the\n"
           "                        client sets the layout-v2 lock override, which lets\n"
           "                        it change the lock set and the hold pose without\n"
           "                        restarting this process -- see identify_rizon.py.\n"
           "\nNOTE: rdk::Scheduler needs RT priority -- run this under sudo.\n";
}

bool HasArg(int argc, char** argv, const std::string& flag)
{
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) {
            return true;
        }
    }
    return false;
}

bool ArgValue(int argc, char** argv, const std::string& flag, double& out)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) {
            out = std::stod(argv[i + 1]);
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 2 || HasArg(argc, argv, "-h") || HasArg(argc, argv, "--help")) {
        PrintHelp();
        return 1;
    }

    const bool sim = HasArg(argc, argv, "--sim");
    std::string robot_sn;
    if (argv[1][0] != '-') {
        robot_sn = argv[1];
    }
    if (robot_sn.empty() && !sim) {
        std::cerr << "error: robot_sn is required unless --sim is given\n";
        return 1;
    }

    BridgeCtx ctx;
    ctx.gravity_comp = !HasArg(argc, argv, "--no-gravity-comp");
    ctx.soft_limits = !HasArg(argc, argv, "--no-soft-limits");
    double v = 0.0;
    if (ArgValue(argc, argv, "--cmd-timeout-ms", v)) {
        ctx.cmd_timeout_s = v / 1000.0;
    }
    if (ArgValue(argc, argv, "--fade-ms", v)) {
        ctx.fade_s = v / 1000.0;
    }
    if (HasArg(argc, argv, "--no-hold")) {
        ctx.hold_enabled = false;
    }
    if (ArgValue(argc, argv, "--hold-scale", v)) {
        ctx.hold_scale = v;
    }
    if (ArgValue(argc, argv, "--hold-tau-frac", v)) {
        ctx.hold_tau_frac = v;
    }
    if (ArgValue(argc, argv, "--release-ms", v)) {
        ctx.release_s = v / 1000.0;
    }
    if (ArgValue(argc, argv, "--sim-drift", v)) {
        ctx.sim_drift = v;
    }
    if (ArgValue(argc, argv, "--hold-slew", v)) {
        if (v <= 0.0 || v > 1.0) {
            std::cerr << "error: --hold-slew must be in (0, 1.0] rad/s, got " << v << "\n";
            return 1;
        }
        ctx.hold_slew = v;
    }
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string("--lock-joints") != argv[i]) {
            continue;
        }
        std::string spec_str = argv[i + 1];
        for (char& c : spec_str) {
            if (c == ',') {
                c = ' ';
            }
        }
        std::istringstream ss(spec_str);
        int j = 0;
        while (ss >> j) {
            if (j < 1 || j > kNumJoints) {
                std::cerr << "error: --lock-joints takes 1.." << kNumJoints << ", got " << j
                          << "\n";
                return 1;
            }
            ctx.locked[j - 1] = true;  // user-facing indices are 1-based (j1..j7)
            ctx.any_locked = true;
        }
        break;
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string("--sim-q") == argv[i] && i + kNumJoints < argc) {
            for (int j = 0; j < kNumJoints; ++j) {
                ctx.sim_q[j] = std::stod(argv[i + 1 + j]);
            }
            break;
        }
    }

    // Shared memory ---------------------------------------------------------
    const int fd = shm_open(kShmName, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        spdlog::error("shm_open({}) failed: {}", kShmName, std::strerror(errno));
        return 1;
    }
    if (ftruncate(fd, kShmSize) != 0) {
        spdlog::error("ftruncate failed: {}", std::strerror(errno));
        return 1;
    }
    // rdk::Scheduler needs RT thread priority, so this process runs under sudo
    // (see flexiv_rdk README: "Root privilege is required if ... Scheduler is
    // used").  The policy runs as the normal user, so hand the segment back to
    // whoever invoked sudo -- rather than widening it to 0666 for everyone.
    if (const char* sudo_uid = std::getenv("SUDO_UID")) {
        const char* sudo_gid = std::getenv("SUDO_GID");
        const uid_t uid = static_cast<uid_t>(std::stoul(sudo_uid));
        const gid_t gid = sudo_gid ? static_cast<gid_t>(std::stoul(sudo_gid)) : (gid_t)-1;
        if (fchown(fd, uid, gid) != 0) {
            spdlog::warn("fchown of shm to uid {} failed: {}", uid, std::strerror(errno));
        }
    }
    if (fchmod(fd, 0660) != 0) {
        spdlog::warn("fchmod of shm failed: {}", std::strerror(errno));
    }
    void* map = mmap(nullptr, kShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        spdlog::error("mmap failed: {}", std::strerror(errno));
        return 1;
    }
    g_shm = static_cast<uint8_t*>(map);
    std::memset(g_shm, 0, kShmSize);

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    try {
        std::unique_ptr<rdk::Robot> robot;

        if (!sim) {
            spdlog::info("Connecting to {} ...", robot_sn);
            robot = std::make_unique<rdk::Robot>(robot_sn);

            if (robot->fault()) {
                spdlog::warn("Fault present, clearing ...");
                if (!robot->ClearFault()) {
                    spdlog::error("Fault cannot be cleared, exiting");
                    return 1;
                }
            }
            if (!robot->operational()) {
                if (!HasArg(argc, argv, "--enable")) {
                    spdlog::error(
                        "Robot is not operational. Re-run with --enable to release the "
                        "brakes, after confirming the workspace is clear.");
                    return 1;
                }
                spdlog::info("Enabling robot (releasing brakes) ...");
                robot->Enable();
                while (!robot->operational() && !g_signalled) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            spdlog::info("Robot is operational");

            double limit = 0.0;
            if (ArgValue(argc, argv, "--timeliness-limit", limit)) {
                robot->SetTimelinessFailureLimit(limit);
                spdlog::info("Timeliness failure limit set to {}%", limit);
            }

            const auto info = robot->info();
            if (info.DoF != kNumJoints) {
                spdlog::error("robot DoF {} != {}", info.DoF, kNumJoints);
                return 1;
            }
            WriteVec(kOffQMin, info.q_min);
            WriteVec(kOffQMax, info.q_max);
            WriteVec(kOffDqMax, info.dq_max);
            WriteVec(kOffTauMax, info.tau_max);
            WriteVec(kOffKqNom, info.K_q_nom);
            U64(kOffDoF)->store(info.DoF, std::memory_order_relaxed);
            for (int i = 0; i < kNumJoints; ++i) {
                ctx.tau_max[i] = info.tau_max[i];  // the hold clamps against this
                ctx.q_min[i] = info.q_min[i];      // and a runtime hold target
                ctx.q_max[i] = info.q_max[i];      // against these
            }
            ctx.have_limits = true;
            ctx.robot = robot.get();
        } else {
            spdlog::warn("--sim: no robot connection; publishing a fake static state");
            const std::vector<double> ones(kNumJoints, 1.0);
            // Roughly the Rizon 4S envelope, so --sim exercises the same
            // limit-margin arithmetic the real robot's limits would.
            const std::vector<double> sim_q_min
                = {-2.97, -2.44, -2.97, -1.05, -2.97, -2.01, -2.97};
            const std::vector<double> sim_q_max
                = {2.97, 2.44, 2.97, 2.77, 2.97, 2.01, 2.97};
            WriteVec(kOffQMin, sim_q_min);
            WriteVec(kOffQMax, sim_q_max);
            for (int i = 0; i < kNumJoints; ++i) {
                ctx.q_min[i] = sim_q_min[i];
                ctx.q_max[i] = sim_q_max[i];
            }
            ctx.have_limits = true;
            WriteVec(kOffDqMax, std::vector<double>(kNumJoints, 2.0));
            WriteVec(kOffTauMax, std::vector<double>(kNumJoints, 100.0));
            WriteVec(kOffKqNom, std::vector<double>(kNumJoints, 3000.0));
            U64(kOffDoF)->store(kNumJoints, std::memory_order_relaxed);
        }

        // Publish the header last, so Python never reads a half-filled one.
        U64(kOffVersion)->store(kLayoutVersion, std::memory_order_relaxed);
        U64(kOffMagic)->store(kMagic, std::memory_order_release);
        U64(kOffInfoReady)->store(1, std::memory_order_release);

        if (!sim) {
            spdlog::info("Switching to RT_JOINT_TORQUE");
            ctx.robot->SwitchMode(rdk::Mode::RT_JOINT_TORQUE);
        }

        rdk::Scheduler scheduler;
        scheduler.AddTask(
            std::bind(PeriodicTask, std::ref(ctx)), "rt_torque", 1, scheduler.max_priority());
        scheduler.Start();
        spdlog::info("1 kHz bridge running. gravity_comp={} soft_limits={}. Ctrl-C to stop.",
            ctx.gravity_comp, ctx.soft_limits);
        if (ctx.any_locked) {
            std::string js;
            for (int i = 0; i < kNumJoints; ++i) {
                if (ctx.locked[i]) {
                    js += (js.empty() ? "" : ",") + std::to_string(i + 1);
                }
            }
            spdlog::warn("LOCKED joints {}: pinned to their startup angle, policy torque "
                         "for them is discarded.",
                js);
        }
        if (ctx.hold_enabled) {
            spdlog::info(
                "Holding startup pose (impedance, <= {:.0f}% of tau_max) until the first "
                "policy command; {:.0f} ms crossfade on handover.",
                ctx.hold_tau_frac * 100.0, ctx.release_s * 1000.0);
        } else {
            spdlog::warn("--no-hold: zero stiffness until the first command; the arm WILL "
                         "drift if left unattended.");
        }

        while (!g_stop_sched && !g_signalled) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        scheduler.Stop();
        spdlog::info("Scheduler stopped after {} ticks", ctx.tick);

        if (ctx.robot != nullptr) {
            try {
                ctx.robot->Stop();
                ctx.robot->SwitchMode(rdk::Mode::IDLE);
                spdlog::info("Robot returned to IDLE");
            } catch (const std::exception& e) {
                spdlog::warn("stop/IDLE failed: {}", e.what());
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        munmap(g_shm, kShmSize);
        shm_unlink(kShmName);
        return 1;
    }

    munmap(g_shm, kShmSize);
    shm_unlink(kShmName);
    return 0;
}
