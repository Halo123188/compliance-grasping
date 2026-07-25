# Compliant grasping via teacher–student tracking-RL

A single proprioception-only policy for a 7-DoF arm + parallel-jaw gripper that

* executes a reach-and-grasp,
* yields compliantly when a human pushes or pulls the arm at any point,
* returns and **resumes the task from where it left off** when released,
* and, once it has grasped, keeps holding the object at constant grasp force
  while the arm still yields to pushes.

The behaviour is not specified by a reward. It is specified by an analytic
privileged teacher, and the reward is nothing but tracking error against it.

## Why teacher–student and not a behavioural reward

The obvious alternative is a weighted sum of compliance / task-progress / force
terms. That approach has a specific failure mode: the desired behaviour is a
*trade-off between conflicting objectives*, so it lives at a particular ratio of
weights that has to be re-found by hand every time anything changes — the robot,
the payload, the perturbation distribution. And the optimum of the weighted sum
is only ever incidentally the behaviour you wanted.

Here, the trade-off is resolved once, analytically, in the teacher. The reward is
convex, dense, and has an unambiguous optimum: *be where the teacher says*.

It is also not behaviour cloning. A BC loss is only defined on the teacher's own
state distribution, so small errors compound off-distribution with nothing
pulling back. On-policy RL against the same target signal samples the student's
distribution by construction.

## The teacher (`mdp/teacher.py`, `mdp/path.py`, `mdp/perturbation.py`)

Four mechanisms, in composition order:

**Path parameter, not time.** The nominal reach-and-grasp trajectory `x_ref(s)`
is indexed by `s ∈ [0,1]`, never by wall-clock time. Phases are laid out along
`s`: approach → descend → closure dwell → lift.

**Freezing (§1.2).** `s` advances at the nominal rate only inside a tube around
the reference; deviation scales `ṡ` smoothly to zero (smoothstep between 1.5 cm
and 4 cm). A push therefore consumes no trajectory, and on release the reference
is exactly where the arm left it. *"Resume where you left off" is a property of
the target signal, not something the policy has to invent.*

**Admittance integration (§1.3).** The compliant target is **not** `x_ref + F/k`.
A quasi-static displacement law has no transient — it snaps on contact and snaps
back on release, and a policy trained to track that learns to snap too. Instead
we integrate a reference impedance

```
m ẍ_t + d ẋ_t + k (x_t − x_ref(s)) = F_ext,    ζ = 0.9
```

giving dynamically consistent yielding and a correctly damped return.

**Perturbation (§1.4).** A spring to a moving anchor, not white noise: the anchor
ramps (min-jerk) to a randomized offset, holds, then releases. Force is saturated
at 40 N. Because the force depends on the arm's *response*, a compliant arm sees
a decaying force and a stiff arm sees a sustained one — which an open-loop force
could not reproduce. Onsets are keyed to `s`, so "push during the reach" and
"push after the grasp" stay well-defined even though `s` freezes. ~25 % of
episodes have no perturbation at all, so clean tracking stays in-distribution.

**Scripted grasp + weld (§1.5, §1.6).** Closure fires when `s` reaches the
closure phase with small deviation; the finger force target ramps to `F_grasp`
and holds. Once established, the object is **welded** to the hand. Grasp
stability under an external push is exactly the regime where the contact solver
is least trustworthy, and RL will farm solver artifacts given the chance.

## The student

**Actor — proprioception only.** Joint position/velocity/torque, EE position and
velocity, task-goal error, previous action; plus finger position, measured grasp
force, and a binary grasp-phase flag in Stage B. Explicitly withheld: `F_ext`,
`K_h`, the hand anchor, `x_t`, `s`. External force is observable *only* as a
signature in the joint-torque history, which is why the actor is a GRU: a single
frame of `τ_meas` cannot distinguish "a hand is pulling me" from "I am
accelerating".

**Critic — privileged.** Teacher target, `s`, `ṡ`, `F_ext`, `K_h`, anchor, and
the number of perturbation events still scheduled. It is never deployed, so
hiding this buys nothing and costs value-estimate variance: the return depends
strongly on how hard and how long you are about to be pushed.

**Action space.** Arm: `Δx_ref` (3) + `log K` per axis (3, 50–2000 N/m), tanh
squashed; damping is derived structurally as `D = 2ζ√(K·m_virtual)`, never an
action, so the closed loop stays passive by construction. Orientation is locked
at fixed rotational stiffness and the null space is damped; the policy controls
neither. Stage B adds one grasp scalar.

`Δx_ref` **integrates** the commanded reference (`x_cmd += Δx_ref`, leashed to
10 cm around the EE) rather than offsetting a fixed anchor — see "Design
decisions" below.

## Reward (§3.2) — tracking only

```
r = 2.0·exp(−‖x_ee − x_t‖²/0.05²)
  + 0.5·exp(−‖ẋ_ee − ẋ_t‖²/0.25²)
  + 1.0·exp(−(F_finger − F_target)²/5²)        [Stage B]
  − 0.01·‖a_t − a_{t−1}‖²
```

No progress term, no force penalty, no stiffness penalty, no compliance term.
Adding one would be a second, weaker, hand-tuned specification of something the
teacher already encodes exactly, competing with the first. The action-rate term
is not an exception to this: it regularizes the actuation, not the trajectory.

## Stages (§3.3)

One environment with pieces switched on, so that everything already verified
stays byte-identical as each new failure mode is added.

| Task ID | Adds |
|---|---|
| `Mjlab-ComplianceTracking-StageA-Flexiv` | reach + perturbations + freezing + admittance. No fingers, no object. |
| `Mjlab-ComplianceTracking-StageB-Flexiv` | scripted grasp, object, weld, post-grasp pushes. |
| `Mjlab-ComplianceTracking-StageC-Flexiv` | transfer randomization: transmission friction, joint damping, rotor inertia, encoder bias, finger gains, payload. |

Stage C has **no contact randomization**, which is not an oversight: perturbations
enter as a wrench and the object is welded, so there is no contact channel for it
to act on.

## Running

```sh
# Environment checks — run these before training anything.
uv run python -m mjlab.tasks.compliance_tracking.scripts.smoke_test --stage A
uv run python -m mjlab.tasks.compliance_tracking.scripts.smoke_test --stage B

# Train.
uv run train Mjlab-ComplianceTracking-StageA-Flexiv --env.scene.num-envs 1024

# Evaluate; --baseline runs the privileged oracle for the tracking ceiling.
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval --stage A --baseline
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval --stage A \
    --checkpoint logs/.../model.pt

# Full runs on SLURM: <stage> <seed> <iterations>. Each job evaluates its final
# checkpoint *and* the oracle on the same fixed batch, so the two are always
# read on identical episodes.
sbatch --job-name ct-A-s0 src/mjlab/tasks/compliance_tracking/slurm/train.sbatch A 0 5000

# Aggregate several seeds' eval blocks into one student-vs-ceiling table.
uv run python -m mjlab.tasks.compliance_tracking.scripts.collect_results \
    slurm_log/ct-A-s*.log
```

## What has actually been verified

Environment and pipeline only — **no policy has been trained past 30 iterations**,
so nothing below is a claim about learned behaviour.

`scripts/smoke_test.py`, 64 envs × 800 steps on GPU, driven by the analytic
oracle, all three stages passing:

| check | Stage A | Stage B | Stage C |
|---|---|---|---|
| oracle tracking error, mean | 1.38 cm | 2.13 cm | 1.67 cm |
| unperturbed episodes | 25 % | 28 % | 20 % |
| `s` reaches 1.0 when untouched | ✓ | ✓ | ✓ |
| `ṡ` while pulled vs free (× nominal) | 0.18 / 0.49 | 0.22 / 0.47 | 0.22 / 0.49 |
| deviation under a pull | 5.5 cm | 5.5 cm | 5.7 cm |
| peak ‖F_ext‖ | 40.0 N (at the limit) | 40.0 N | 40.0 N |
| object stays welded to the hand | — | ≤ 5.0 cm | ≤ 2.3 cm |
| grasp-force error while pushed post-grasp | — | 4.51 N | 1.83 N |

The freezing law works (`ṡ` drops ~3× while the hand pulls and recovers after),
the perturbation saturates exactly at its limit rather than spiking, and the
weld holds the object through post-grasp pushes while the grasp force stays near
`F_grasp = 15 N`.

`scripts/eval.py --baseline`, 128 envs × 800 steps, Stage A (the oracle is the
tracking *ceiling*, so these are the numbers a student is measured against):

```
track_err_all_cm  1.22   track_err_unperturbed_cm  1.46
track_err_pull_cm 2.58   track_err_release_cm      2.40
s_rate_pulled     0.224  s_rate_free               0.926
release_overshoot_cm 0.008          peak_f_ext_N   40.0
```

`s` runs at ~93 % of nominal untouched, falls to ~22 % while the hand pulls, and
resumes on release — with essentially no overshoot returning to `x_ref`, which is
what integrating the reference impedance buys over a quasi-static `Δx = F/k`.

The PPO+GRU loop runs end to end: 30 iterations at 512 envs, ~0.55 s/iter on an
RTX PRO 6000, mean reward rising monotonically (Stage A 620, Stage C 45.5 →
978.5). That is a sanity check that learning is wired up, **not** evidence that
the task is solved.

`tests/test_compliance_tracking.py` pins the teacher's structural invariants
(path waypoints/continuity/phase ordering, `velocity` consistent with
`d position/dt`, the freeze gate's endpoints and monotonicity, force saturation,
onset ramping, and that `s`-keyed onsets never fire early).

One number worth understanding: the oracle's `K_anisotropy_pull` reads exactly
1.000, because it commands a fixed isotropic stiffness by construction. That is
the *correct* reading, and it is why the metric is a useful discriminator — the
analytic ceiling on tracking is not a compliance exemplar.

## Reading the diagnostics (§3.4)

Tracking reward alone cannot tell the two hypotheses apart — it rises both when
the policy gets stiffer and when it becomes genuinely compliant.

**The metric that decides whether the run worked is `K_anisotropy_pull`**
(`K_∥/K_⊥` about the pull direction, while being pulled). A policy that tracks
`x_t` by stiffening every axis is *resisting* the teacher's yield and dragging
the human along, and it can still post a good position error. `K_∥/K_⊥ < 1` while
pulled is the signature of the intended behaviour.

The rest: tracking error split by regime (the aggregate is dominated by
unperturbed steps and hides everything interesting), `s_rate_pulled` vs
`s_rate_free` (freezing works), `release_overshoot_cm` (the return transient), and
`grasp_force_err_pushed_N` (the §4 claim).

## Result: the tracking-only reward does not produce directional compliance

Stage A, 3 seeds × 5000 iterations, 1024 envs, evaluated on a fixed 256-env ×
800-step batch against the analytic oracle:

| | s0 | s1 | s2 | pooled | oracle |
|---|---|---|---|---|---|
| `K_parallel_pull` (N/m) | 188 | 172 | 170 | 177 | 600 |
| `K_perp_pull` (N/m) | 140 | 121 | 106 | 122 | 600 |
| **`K_∥/K_⊥`** (ratio of means) | 1.34 | 1.42 | 1.60 | **1.44** | 1.00 |
| `track_err_all_cm` | 2.31 | 2.26 | 3.42 | 2.66 | 1.22 |
| `release_overshoot_cm` | .008 | .008 | .009 | .008 | .008 |

The policy commands ~44 % **more** stiffness along the pull than orthogonal to
it — the opposite of the intended behaviour, reproducible across seeds.

Note it is *not* bracing: absolute stiffness is 122–177 N/m near the bottom of
the 50–2000 range, and at K = 177 a 40 N pull implies `F_ext/K` = 23 cm, so the
arm is not rejecting the force at all. It is soft everywhere, just less soft
along the pull.

**Read `K_∥/K_⊥` as a ratio of means.** The per-step ratio averaged instead is
convex in `K_∥` and upward-biased by Jensen whenever it is heavy-tailed —
measured at 1.80 against a true 1.09 on the Stage B policy, a bias larger than
the effect. `eval.py` reports both (`K_anisotropy_pull` is now the ratio of
means, `K_anisotropy_meanratio` the biased one) so the gap stays visible.

**Why: hypothesis tested and NOT supported.** The proposal was that yielding is
a *fast* motion (admittance ωₙ ≈ 12 rad/s, so `x_t` moves ~0.6 m/s during a
yield against ~0.14 m/s along the nominal path, and only along the pull axis),
that tracking it needs bandwidth, and that in this action space bandwidth *is*
stiffness. Prediction: slow the transient and the anisotropy falls toward 1.

Measured (`slurm/diag_bandwidth.sbatch`, admittance mass 2 → 8 kg, halving ωₙ
with the steady-state yield `F/k` unchanged):

| | `K_∥` | `K_⊥` | `K_∥/K_⊥` |
|---|---|---|---|
| m = 2 kg (3 seeds) | 177 | 122 | 1.445 |
| m = 8 kg (2 seeds) | 165 | 120 | **1.375** |

A ~5 % shift with overlapping seed ranges — directionally consistent but
nowhere near explanatory. **The bandwidth account does not explain the
anisotropy.** The cause remains open.

What survives regardless: `K_∥/K_⊥ ≥ 1` in *every* configuration measured
(Stage A 1.44, slow-admittance 1.38, Stage B 1.09) and never below it. The
policy never softens along the pull, whatever the reason.

**The deeper problem.** "Compliant along the push" is an *impedance* property;
the reward only constrains the *trajectory* `x_ee(t)`. "Yielded because it was
soft" and "yielded because it stiffly drove its reference away" produce
identical `x_ee(t)`, so no tracking reward can distinguish them. The spec's note
that the action space admits no constant-spring degenerate solution is correct,
but not-degenerate does not imply it learns the intended content.

**Stage B (1 seed, 6000 iters) splits into a clear win and the same loss.**

| | student | oracle |
|---|---|---|
| `grasp_force_err_N` | **0.52** | 3.03 |
| `grasp_force_err_pushed_N` | **2.45** | 3.19 |
| `release_overshoot_cm` | **0.0064** | 0.0102 |
| `K_anisotropy_pull` | 1.80 | 1.00 |
| `track_err_all_cm` | 3.41 | 2.00 |

The grasp behaviour works and beats the analytic baseline: 0.52 N of a 15 N
target, 2.45 N while the arm is being pushed post-grasp — learned from noisy
fingertip sensors, better than a hand-tuned integral force controller. That is
the §4 claim ("the arm still yields, but the grasp force is maintained").

The compliance failure persists, though *less* severely than Stage A once read
correctly: `K_∥/K_⊥` = 454/417 = **1.09**, i.e. near-isotropic, at a much higher
absolute stiffness (Stage A was 177/122). The `K_anisotropy_meanratio` column
shows 1.80 for this policy, which is the Jensen bias described above and should
not be read as the anisotropy.

Caveats: one seed only, and the student's arm tracking is well short of the
oracle (3.41 vs 2.00 cm) with `s_rate_free` 0.37 vs 0.78 — it deviates enough
that the teacher's own freeze law throttles it even unperturbed. Some of that is
probably undertraining rather than a structural limit.

**Discriminating test** (`slurm/diag_bandwidth.sbatch`): raise the admittance
mass 2 → 8 kg, halving ωₙ while leaving the steady-state yield `F/k` unchanged.
If anisotropy falls toward 1, the bandwidth account holds; if it stays ~1.5, the
account is wrong.

## Stiffness supervision applied — two rounds, both negative

Candidate fix 1 above (supervise the impedance, not only its output `x_t`) was
implemented: the teacher emits a target Cartesian stiffness `K_target` and a new
reward term `w_K·exp(−mean_i(ln K_i − ln K_target_i)²/σ²)` scores the commanded
impedance against it (`mdp.stiffness_tracking`, `supervise_stiffness=True`). This
departs from pure tracking on purpose — `K_target` is a hand-authored impedance
profile — and it is gated behind the `…-Sup-Flexiv` task variants so the pure
tracking tasks stay byte-identical.

**Round 1 — isotropic target** (`k_soft` on every axis while pushed, `k_precision`
when free), swept `w_K ∈ {0.5, 1, 2}`, *and* a widened freeze band (4 → 10 cm) to
unstick the lift. Clean 256-env × 800-step eval:

| | K∥/K⊥ (pull) | K_∥ (N/m) | s_peak | welded | grasp_err |
|---|---|---|---|---|---|
| baseline (unsup B) | 1.09 | ~450 | 0.84 | 0.54 | 0.52 |
| Sup-B, best w_K | ~1.0 | 596–924 | 0.28–0.74 | 0.02–0.47 | 0.6–5.1 |

Failed twice over: the arm did **not** go soft (K stayed 390–924, target 200, and
higher `w_K` gave *higher* K), and the task regressed (s stalls, grasp collapses).
Confounded, too — the freeze band moved at the same time.

*Why isotropic softness cannot work:* the teacher's `x_t` yields only along the
push; perpendicular it stays on `x_ref`. Softening every axis loses the
perpendicular precision the dominant position reward needs, so the policy
correctly refuses to soften at all.

**Round 2 — anisotropic target** (soft only along the pull `u`, stiff
perpendicular), freeze band **restored to baseline** so stiffness supervision is
the only variable, grid `w_K ∈ {0.5,1,2} × σ_pull ∈ {0.05,0.10}` (`σ_pull`
relaxes the position tolerance only during a pull, so softening is not a pure
tracking loss). Stage A, 6 runs:

| | K∥/K⊥ | K_∥ (N/m) | K_⊥ (N/m) | s_peak |
|---|---|---|---|---|
| all 6 runs | **0.98–1.01** | 628–836 | 620–840 | 0.64–0.82 |

Zero anisotropy produced, at a stiffness (~660) *higher* than the unsupervised
baseline (~440). `σ_pull` had no effect. The control worked: with the freeze band
back at baseline, `s_peak` recovered to 0.64–0.82 (vs round-1's 0.28–0.5), so
round-1's task collapse was the freeze-band change, not the stiffness reward.

**Mechanism (quantified).** Pull directions are uniform-random on the 3D sphere
(`perturbation.py`, `randn` then normalize). For a random `u`, each axis' target
is `900 − 700·E[u_i²] = 900 − 700/3 ≈ 667`. The policy learns to output exactly
that **isotropic mean** (measured 660) instead of tracking the per-axis
structure. If it *had* tracked, the metric would read `E[K_∥]/E[K_⊥] ≈ 0.63`; it
reads 1.00, i.e. zero directional response. To track the per-axis target the
policy must, every step, know `u` and rotate a diagonal stiffness onto it — and
it is blocked twice:

1. **Observability.** The actor is proprioception-only; the generic-3D pull
   direction is not directly visible, only noisily inferable from joint torques.
2. **Representation.** `log K` is three *axis-aligned* stiffness knobs (base
   frame). A diagonal stiffness ellipsoid cannot tilt onto an off-axis `u` even
   if the direction were known.

So both stiffness-supervised rounds converge to the same behaviour the pure
tracking reward already gave: isotropic, moderate-to-high stiffness, no
directional compliance. The supervision reward gets satisfied by the isotropic
mean and buys nothing.

**Next, if pursued — a decisive cheap diagnostic:** feed the *privileged* `u`
into the actor for one run. If `K_∥/K_⊥` then drops below 1, observability is the
binding wall and the deployable fix is a proprioception-derived force estimate (a
momentum observer on joint torques) as an actor observation. If it stays ~1, the
diagonal-K representation is the wall and the action space needs a rotatable
stiffness (e.g. output a soft-axis direction + two stiffness scalars, or command
`K` in a frame aligned to the force estimate).

**Remaining options:**

1. Run the diagnostic above, then either add the force-observer observation or
   the rotatable-stiffness action depending on which wall it identifies.
2. Add an unobservable secondary load, so high stiffness stops being free (the
   trick the Stage-1 experiment used) — orthogonal to the two walls above.
3. Accept it and state the ceiling: the *motion* is right (yield, return, resume,
   hold grasp force — all deployable), the *directional impedance* is not, for
   the two structural reasons above.

## The diagnostic run, and where the wall actually is (exp-1, exp-2, aux)

The "if pursued" diagnostic above was run, and it moved the wall twice.

**exp-1 — feed the privileged `u` to the actor** (`…-SupDiag-Flexiv`, diagonal-K
action). `K∥/K⊥` dropped from ~1.0 to **0.65** — exactly the diagonal-K floor
`E[u_i²]`-mean predicts. So in the diagonal-K regime the binding wall was
*observability*: hand the direction over and the policy softens as far as an
axis-aligned stiffness structurally can.

**exp-2 — direct joint-torque action** (`…-Torque-Flexiv`). Replacing the
impedance action with a policy that outputs joint torque directly, tracking the
teacher's compliant torque target (`mdp.torque_tracking`, a *rotatable*
anisotropic impedance about `x_ref`), removes the representation wall. Under the
steady-state probe (`_stiffness_probe.py`: one sustained generic push, `k_par`
measured only near force saturation), the torque policy reaches **k_par ~390 N/m
/ ~12 cm yield** — genuinely softer than the diagonal-K action's ~600–825. The
*action space*, not sensing, was the binding constraint here.

**aux — learn the force estimate** (`…-TorqueAux-Flexiv`, `rl_aux.py`). The
deployable version of exp-1: a train-time head off the GRU latent regresses the
privileged `F_ext` (label normalized by the force limit), so the actor learns to
*infer* the push from its own joint-torque history with no privileged input at
deploy. The head learns it well — final normalized MSE 0.0006–0.003, i.e. ~1–2 N
RMS on a 40 N scale. But steady-state compliance does **not** improve:

| torque variant | k_par (N/m) | yield |
|---|---|---|
| learned sensing (aux 0.5 / 2.0, ×2 seeds) | 350, 391, 621, 349 | 9–13 cm |
| no sensing (baseline) | 389 | 11.8 cm |
| privileged direction (TorqueDiag) | 407 | 11.0 cm |

All ~390 on `k_par`, and the aux runs did not even reduce seed variance (one
went to 621). So *for the aux hypothesis* the result is negative: **force
observability is not the binding constraint for the torque policy** — neither
learning the estimate nor handing the direction in pushes `k_par` below the ~390
the no-sensing baseline already reaches.

**But `k_par` alone was the wrong lens, and the two-direction probe
(`_stiffness_probe2.py`) corrects the headline.** It keeps the main push along
`u` and injects a smaller perpendicular test force, so it reads both `k_par`
(along) and `k_perp` (across):

| torque variant | k_par | k_perp | **k∥/k⊥** | yield ∥ / ⊥ |
|---|---|---|---|---|
| aux 0.5 (s0, s1) | 357, 407 | 1431, 837 | **0.25, 0.49** | ~12 / ~1 cm |
| aux 2.0 (s0, s1) | 583, 367 | 4246, 2228 | **0.14, 0.16** | ~10 / ~0.4 cm |
| no sensing (baseline) | 430 | 1288 | **0.33** | 10.9 / 0.8 cm |
| privileged dir (TorqueDiag) | 450 | 972 | **0.46** | 10.6 / 1.0 cm |

Every torque policy is **strongly directional**: `K∥/K⊥ ≈ 0.14–0.49`, yielding
~10–12 cm along the push but only 0.2–1.2 cm across it. The `~390 k_par` is not
"stiff" — against `k_perp ~1300` it is *soft along the push and stiff
perpendicular*, exactly the §4 goal. This is the real headline of exp-2: the
**direct-torque action achieves directional compliance the diagonal-K action
structurally could not** (~1.0), from a tracking-only-plus-torque-target reward.
(The measurement is conservative: adding the test force tilts the sensed axis
~14°, so `k_perp` reads slightly *low*, i.e. true directionality is at least this
strong.)

The aux loss remains a null result *relative to the baseline* — it does not
improve `K∥/K⊥` over no-sensing (0.14–0.49 vs 0.33), consistent with observability
not being the torque policy's binding constraint. The `k_par` floor (~390 vs
target `k_soft`=200) and the residual seed variance are set by the reward-level
soft-vs-precision tension (the torque target is an impedance *about `x_ref`*, so
yielding past ~12 cm costs tracking reward), which a better force estimate cannot
change. Video: `compliance_torqueaux_aux0.5_s0.mp4` (scene + joint-torque input +
the aux head's live force estimate vs the true push + live `k_par`).

## Known ceilings

State these in any writeup; do not claim more.

1. **Compliance quality is upper-bounded by the teacher's admittance law.** The
   student can at best reproduce `m, d, k`. It cannot discover a better yielding
   policy than the one specified, because nothing rewards it for doing so.
2. **Grasp robustness is upper-bounded by the weld + constant-force
   approximation.** Real-world slip and regrasp are out of scope for this
   pipeline, by construction and not by omission. The policy will not learn slip
   recovery from this sim, and evaluating it on slip would be measuring nothing.
3. What the learning actually buys is narrow and worth stating precisely: it
   folds implicit external-force estimation from joint-torque history, behaviour
   switching, and grasp timing into **one proprioception-only policy deployable
   without a wrist F/T sensor**. That is the claim; nothing beyond it is
   supported by this pipeline.

## Design decisions worth knowing about

**The impedance anchor is the policy's own integrated reference.** Two
alternatives were rejected. Anchoring at the teacher target `x_t` hands the
policy the answer and collapses the optimum to "output maximum stiffness" —
there would be nothing to learn. Anchoring at the fixed task goal bounds the
equilibrium within a few cm of the goal, so the reachable target set cannot
express "hold back here while the hand pulls" or the closure dwell, which are
precisely the parts of `x_t` the student must reproduce. The spec's "±3 cm around
current commanded reference" is read as a per-step increment; the value used is
1 cm/step (a 1 m/s reference slew limit at 100 Hz), which is ~7× the nominal path
speed and above the peak of the admittance return transient, while keeping useful
commands away from the first 1 % of the action range.

**The perturbation lives inside the teacher command term**, not in a separate
event term. The manager order runs commands before step events, so a split would
put a one-step skew between the force applied to the arm and the force the
admittance law integrates.

**Post-reset pose capture is deferred by one step.** `site_xpos` is one
`forward()` stale inside `reset()` (reset events write state; the env forwards
once afterwards), so both the teacher's path start and the impedance controller's
orientation lock are seeded on the first `process_actions` instead. This also
fixes a latent bug in the shared impedance term, where the orientation lock was
being captured from the *previous* episode's final pose.
