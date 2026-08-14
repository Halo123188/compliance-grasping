"""Task registrations for the WIDE claw on the foam bench.

Deliberately short. The sibling ``flexiv_two_finger`` package registers 29 task
ids, and 26 of them are ablation arms whose conclusions are recorded in its
comments -- the alignment gate, the touch kernel, the one-way finger limits, the
wrist experiments. Those conclusions are inherited here as ONE recipe
(``V11_ALIGN_KWARGS``, the kwargs of run 46984 at 80.5% deployment success)
rather than re-run, because re-running an ablation whose answer is known costs a
GPU-day per arm and the answers were about the reward stack, which has not
changed.

What HAS changed is the robot and the bench, so the small number of things that
genuinely could not carry over -- the finger travel limits, the action scales,
the pad geometry -- are re-measured (``scripts/wide_calibrate.py``) rather than
re-searched, and the few arms below exist only where a measurement left a real
choice open.

Ids are ``TwoFingerWide`` so both packages can be imported at once and an old
checkpoint stays evaluable against the claw it was trained on.
"""

from typing import Any

from mjlab.tasks.manipulation.rl import (
  ManipulationDistillationRunner,
  ManipulationFinetuneRunner,
  ManipulationOnPolicyRunner,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  flexiv_two_finger_distill_env_cfg,
  flexiv_two_finger_grasp_env_cfg,
  flexiv_two_finger_vision_env_cfg,
)
from .rl_cfg import (
  flexiv_two_finger_distill_runner_cfg,
  flexiv_two_finger_finetune_runner_cfg,
  flexiv_two_finger_grasp_ppo_runner_cfg,
  flexiv_two_finger_vision_ppo_runner_cfg,
)

# One-way finger travel, re-derived for THIS claw. The old claw's version was
# ((0.50, 1.60), (-1.60, -0.40)): proximal may only spread OUTWARD from the
# hover, distal may only curl INWARD.
#
# The distal half carries over unchanged in spirit -- curling out is useless on
# any two-finger claw -- but the PROXIMAL half does not, and copying it would
# have been fatal. On the old claw the hover angle WAS the closed end of the
# useful range, so pinning the lower bound at the hover cost nothing. On this
# claw pad separation is affine in the proximal angle, sep(q) = 50.1 + 134.0 q
# mm, and the hover sits at q = +0.2746 (86.9 mm) with the GRIP at q ~ 0
# (50.1 mm, flush on the cube). A lower bound at the hover would put every
# grasping pose outside the action space.
#
# So the bound is the drive-through limit instead: q = -0.0754 is 40 mm of pad
# separation, a 10 mm squeeze on the 50 mm cube, and below that the 2 N.m servo
# saturates against interpenetration rather than gripping harder.
#
# The distal's upper bound is +0.10 rather than 0.0 so the hover default is not
# sitting exactly on a joint limit, which would clamp half the actuator's
# ctrlrange and make half of that action dimension dead.
_WIDE_DIRECTIONAL = ((-0.0754, 1.60), (-1.60, 0.10))

# The recipe. Every one of these was established on the old claw and every one of
# them is about the REWARD rather than the hardware:
#
#   goal_mode="above_object"  the goal sits over wherever the cube spawned, so
#                             the reward's position error IS the lift height,
#                             which is what `cube_lifted` actually scores.
#   bringing_std=0.12         narrowed to that 8-19 cm band.
#   reach_from="pads"         measure reaching from between the fingertips, not
#                             from a palm point the hand can park on the cube
#                             while splaying the fingers away.
#   touch_weight=1.0          score the tips ON the cube instead of any proxy
#                             for it; this also removes `pad_pinch`, whose
#                             optimum is both pads at the cube's CENTRE.
#   align_gate=0.3            jaw alignment as a MULTIPLIER on the grasp reward
#                             with a floor, never as its own term -- additive
#                             failed at 1.0 (hacked) and at 0.15 (ignored).
V11_ALIGN_KWARGS: dict[str, Any] = dict(
  goal_mode="above_object",
  bringing_std=0.12,
  finger_range=_WIDE_DIRECTIONAL,
  touch_weight=1.0,
  reach_from="pads",
  align_gate=0.3,
)

# init_std 0.8, carried from v11_align, and it is the EFFECTIVE noise that had to
# be preserved rather than the number. Exploration reaches roughly
# `scale * init_std` radians per joint, so what matters is that product:
#
#            action scale   init_std   noise      full close
#   old claw     0.60          0.8      0.48 rad   -0.60 rad = -1.25 sigma
#   this claw    0.35          0.8      0.28 rad   -0.35 rad = -1.25 sigma
#
# i.e. closing the jaw is exactly as reachable as it was on the run that worked.
# Dropping to 0.6 to match cubegrasp-env's 0.16 rad arm budget would push it to
# -1.67 sigma, and past ~1.5 sigma is the band where a reward reads 0.0000 for a
# whole run. The cost is that the ARM sees 0.40 * 0.8 = 0.32 rad of noise against
# that 0.16 rad budget; if it thrashes into the bench (watch
# `fingertip_table_contact` climbing while `lift` stalls) 0.6 is the lever.
_STD, _ENT = 0.8, 0.005

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv",
  env_cfg=flexiv_two_finger_grasp_env_cfg(**V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, **V11_ALIGN_KWARGS),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# Same recipe, plus the success bonus. On the old claw succeeding COST half the
# return and was paid nothing for it: `cube_lifted` lands in `terminated` rather
# than `time_outs`, so rsl_rl zeroes the bootstrap, while the dense stack pays
# ~5/s indefinitely for hovering near the cube. "Lift to 9 cm and stay there"
# strictly dominated "lift to 10 cm and hold". v11_align reached 80.5% in spite
# of that rather than because the exploit was unavailable, so this is the arm
# that removes a known ceiling; both halves are needed and neither works alone.
_SUCCESS_KWARGS: dict[str, Any] = dict(**V11_ALIGN_KWARGS, success_reward=2.0)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success",
  env_cfg=flexiv_two_finger_grasp_env_cfg(**_SUCCESS_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, **_SUCCESS_KWARGS),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# Calmer exploration. The one thing the new bench genuinely puts at risk: the arm
# action scale had to go 0.30 -> 0.40 to reach across the jitter box from a
# plate mount, which raises arm noise to 0.32 rad against the 0.16 rad budget
# cubegrasp-env validated. Held in reserve for the failure mode where the policy
# learns to RETREAT because table-crash penalties bury the sparse grasp reward.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Calm",
  env_cfg=flexiv_two_finger_grasp_env_cfg(**V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, **V11_ALIGN_KWARGS),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.6, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# --- vision (PPO from pixels, not used by the distillation path) -------------
for _cam in ("rgb", "depth", "rgbd"):
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-{_cam.capitalize()}",
    env_cfg=flexiv_two_finger_vision_env_cfg(_cam),
    play_env_cfg=flexiv_two_finger_vision_env_cfg(_cam, play=True),
    rl_cfg=flexiv_two_finger_vision_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
  )

# --- distillation ------------------------------------------------------------
# The teacher observation group these build IS whatever the state cfg builds, so
# the teacher kwargs are passed through rather than restated: the teacher's
# weights were fitted to that exact vector in that exact order, and a kwarg that
# drifts here silently feeds a trained policy a different observation.
#
# DAGGER MIXING IS ON BY DEFAULT, unlike the sibling package where it was one
# ablation arm among several. Its finding is not a preference: without it the
# student drives from iteration 0, knocks the cube off the bench within a few
# steps, and every subsequent label is the teacher's opinion about a cube on the
# floor. Runs 50273/50274/50426 all showed the behaviour loss pinned at the
# constant-predictor value for their whole length. Executing the teacher with
# probability beta, annealed 1 -> 0 over 500 iterations, is what makes the first
# few hundred iterations of labels be about the task.
_BETA_ITERS = 500

for _cam in ("depth", "rgbd"):
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-{_cam.capitalize()}",
    env_cfg=flexiv_two_finger_distill_env_cfg(_cam, **V11_ALIGN_KWARGS),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(_cam, play=True, **V11_ALIGN_KWARGS),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=_BETA_ITERS),
    runner_cls=ManipulationDistillationRunner,
  )

# Finer first conv stride. On the old claw this was the fix for a student that
# found the cube and missed on precision -- 78.7 mm median lift, 3.1% over the
# 100 mm bar -- because a 9-15 px cube in a 128x72 frame collapses to about one
# cell of a 16x9 feature map.
#
# Held as an ARM here rather than made the default, because the render changed
# with the camera: at 160x120 the stride-2 trunk already ends at 20x15 = 300
# cells against the old 144, so the plain trunk may be enough and this doubles
# the first layer's cost. Run it if the plain one stalls the same way.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Fine",
  env_cfg=flexiv_two_finger_distill_env_cfg("depth", **V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", play=True, **V11_ALIGN_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=_BETA_ITERS, fine_cnn=True
  ),
  runner_cls=ManipulationDistillationRunner,
)

# Control condition, not a deployable policy: the student is handed the
# teacher's own observation and no image, so it should fit almost perfectly and
# fast. A flat loss curve on the camera student is ambiguous between "perception
# is hard here" and "the distillation plumbing is broken", and this is the only
# thing that separates them.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Oracle",
  env_cfg=flexiv_two_finger_distill_env_cfg("depth", **V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", play=True, **V11_ALIGN_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(student_sees_state=True),
  runner_cls=ManipulationDistillationRunner,
)

# Same, for a teacher trained with the success bonus -- its observation group is
# identical (success_reward changes rewards and terminations only), but the
# kwargs must still match the teacher's or the env differs in ways the DAgger
# rollout sees.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success",
  env_cfg=flexiv_two_finger_distill_env_cfg("depth", **_SUCCESS_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg("depth", play=True, **_SUCCESS_KWARGS),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=_BETA_ITERS),
  runner_cls=ManipulationDistillationRunner,
)

# --- domain randomization ----------------------------------------------------
# Everything above trains and evaluates on ONE robot, ONE camera pose and ONE
# object: mass, friction, servo gains, joint friction, encoder zeros, bench
# height, start pose and camera extrinsics are all single fixed numbers. That is
# why the teacher and the student both read 100.0% -- the number is honest about
# the reward stack and the geometry being self-consistent, and says nothing
# about whether the policy survives the hardware being slightly different from
# the model. It will be lower here, and the lower number is the meaningful one.
#
# See dr_cfg.py for every range and its provenance. Registered as separate ids
# rather than switched on in place so the runs above stay reproducible and their
# checkpoints stay evaluable in the env they were fitted in.
_DR_KWARGS: dict[str, Any] = dict(**_SUCCESS_KWARGS, physics_dr=True)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr",
  env_cfg=flexiv_two_finger_grasp_env_cfg(**_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, **_DR_KWARGS),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# The student's env. Physics DR must match the teacher's exactly -- the teacher
# is the labeller, and a teacher asked about dynamics it never trained under
# gives confidently wrong labels rather than no labels. Perception DR is added
# ON TOP, and only here: the teacher reads privileged state and cannot see the
# camera at all, which is what makes this half re-runnable against a
# finished teacher instead of requiring another PPO run.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr",
  env_cfg=flexiv_two_finger_distill_env_cfg("depth", percept_dr=True, **_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", play=True, percept_dr=True, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=_BETA_ITERS),
  runner_cls=ManipulationDistillationRunner,
)

# Same env, longer DAgger handover. `_BETA_ITERS = 500` was fitted on a student
# whose only job was to find a cube in a FIXED camera; under perception DR the
# same schedule hands the wheel over before the student can see, and run 56493
# shows exactly that -- training success collapses 0.99 -> 0.27 across
# iterations 450-550, then spends 500 iterations climbing back out. The no-DR
# student never dipped at all, and its it1500 checkpoint scored 60.2% on
# deployment against the no-DR student's 100%.
#
# 1500 rather than 500, i.e. half the run supervised. Overrunning costs only
# that late labels are collected on-teacher rather than on-student;
# underrunning costs a student whose first act of autonomy is to knock the cube
# off the bench, after which every label is the teacher's opinion about a cube
# on the floor. Held as a separate id so the 500-iteration run stays comparable.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-Beta",
  env_cfg=flexiv_two_finger_distill_env_cfg("depth", percept_dr=True, **_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", play=True, percept_dr=True, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=1500),
  runner_cls=ManipulationDistillationRunner,
)

# Same env, finer first conv stride. Run 56493 scored 60.2% at iteration 1500
# and 60.9% at 2999 -- the last half of the run bought NOTHING on deployment
# while its training success climbed 0.89 -> 0.97. Whatever the student is
# missing, it is not more on-policy data, so `-Dr-Beta` is unlikely to be the
# whole answer and this is the other candidate.
#
# The evidence points at the WRIST specifically. corr(cube yaw, joint7) is
# -0.055: the student has stopped tracking cube orientation altogether, its jaw
# lands 17.2 deg off square against the teacher's 13.0, and joint7 wanders with
# a 20.6 deg spread against the teacher's 4.6. The teacher READS `cube_quat`;
# the student has to infer it from a 16-25 px cube, and under camera-pose
# jitter, ORIENTATION degrades far faster than position -- a few degrees of
# extrinsic error moves the cube's apparent yaw by a comparable amount while
# barely moving its centroid. That is a spatial-resolution problem, which is
# exactly what this arm addresses and what beta scheduling does not.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-Fine",
  env_cfg=flexiv_two_finger_distill_env_cfg("depth", percept_dr=True, **_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", play=True, percept_dr=True, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=_BETA_ITERS, fine_cnn=True
  ),
  runner_cls=ManipulationDistillationRunner,
)

# --- speed limiting ----------------------------------------------------------
# Everything above trains a policy that does the entire task -- descend 50 mm,
# close, lift 127 mm -- inside 0.6 s of a 20 s episode. Measured on w1_dr's
# final checkpoint (scripts/wide_speed_audit.py, job 60866, 128 envs):
#
#   fastest arm joint     p95 1.52   p99 2.23   max  4.05 rad/s
#   commanded joint rate  p95 2.40   p99 8.47   max 24.57 rad/s
#   tool-point speed      p95 0.40   p99 1.14   max  2.27 m/s
#   during approach       tcp p95 1.44 m/s, and it is 4.6% of the rollout
#
# On the bench that reads as the arm snatching at the cube, which is the
# complaint these arms exist to answer. The audit also prices the obvious fix
# and finds it weak: at the stock max_vel 0.5 / weight -0.1 the hinge penalty
# costs ~0.02 per step against `lift` at ~1.27, and because the fast part is a
# 4.6%-duty burst inside a 95%-stationary episode, the weight that would bite
# the burst is close to the weight already known to regress the task into
# "pinch and press down on the table".
#
# So the arms below split into two families that are NOT variations of one
# idea. A hard slew cap on the command (`rate_limit`) is a guarantee that also
# holds on hardware, at the cost of changing the action space the policy
# explores in. A velocity penalty (`-Slow`) leaves the action space and the
# whole deployment path untouched and can only ever produce a tendency.
#
# The v_max numbers are read off the audit rather than picked: 1.0 rad/s sits
# below the measured commanded p95 (2.40) but above the realized joint p95
# (1.52), which clips 76% of approach steps and only 14% of carry steps. The
# wrist gets 1.5 because it has to reach +-45 deg of roll before contact and is
# the one joint with a history of being frozen by penalties, and the fingers
# get 3.0 because their whole closing travel is 0.35 rad -- at 1.0 that alone
# would take 350 ms and change the grasp's timing rather than its speed.
_SLEW = {r"joint[1-6]": 1.0, r"joint7": 1.5, r"(left|right)_[12]_tf": 3.0}
_SLEW_TIGHT = {r"joint[1-6]": 0.5, r"joint7": 1.0, r"(left|right)_[12]_tf": 2.0}

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-Slew",
  env_cfg=flexiv_two_finger_grasp_env_cfg(rate_limit=_SLEW, **_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True, rate_limit=_SLEW, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# Half the cap. This is the arm that tests the one prediction the audit makes
# about where the approach stops working: with the arm scale at 0.40, a 0.5
# rad/s cap is 0.01 rad per control step, i.e. 0.025 of an action unit, while
# the measured p95 of the action's own step-to-step change is 0.12. The limiter
# would then be saturated on most steps, which turns a position-offset action
# into a direction-only one and should cost precision, not just time. If this
# arm holds up anyway, the cap can go lower without touching the action scale.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-SlewTight",
  env_cfg=flexiv_two_finger_grasp_env_cfg(rate_limit=_SLEW_TIGHT, **_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True, rate_limit=_SLEW_TIGHT, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# Same final cap as -Slew, reached on a schedule. This is the diagnostic arm:
# if -Slew scores badly, this separates "the cap is too tight for the task"
# from "the cap prevented the task from being discovered". Grasping emerges
# here around iterations 750-2000 (`grasp` reads exactly 0.0000 before ~750),
# so the schedule stays 3x loose through discovery and is at the final value
# with ~1500 iterations left to exploit it -- the shape that worked for
# align_std and failed when the same tightening was applied from step 0.
_SLEW_STAGES = [
  {"step": 0, "scale": 3.0},
  {"step": 1000 * 24, "scale": 2.0},
  {"step": 1750 * 24, "scale": 1.5},
  {"step": 2500 * 24, "scale": 1.0},
]

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-SlewCurr",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    rate_limit=_SLEW, rate_limit_stages=_SLEW_STAGES, **_DR_KWARGS
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True, rate_limit=_SLEW, rate_limit_stages=_SLEW_STAGES, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# The reward-only arm, and the only one whose checkpoint drops into the existing
# deployment path unchanged -- same action term, same ONNX graph, same
# deploy/policy.py. Worth running for that reason alone: if it is enough, none
# of the machinery above has to ship.
#
# max_vel 0.5 -> 0.25 and the final weight -0.1 -> -0.5, which the audit sizes
# at roughly 0.24 per step (against 0.02 today). The ramp starts at 1500 rather
# than 500 because `grasp` is still 0.0000 at iteration 600 and a penalty
# applied during the discovery window is what collapsed joint7's action std to
# 0.0083 rad. action_rate_l2 goes -0.01 -> -0.05, which charges the step-to-step
# action change the commanded-rate column is literally measuring.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-Slow",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    joint_vel_max=0.25,
    joint_vel_penalty_final=-0.5,
    joint_vel_ramp_iters=(1500, 2500),
    action_rate_weight=-0.05,
    **_DR_KWARGS,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    joint_vel_max=0.25,
    joint_vel_penalty_final=-0.5,
    joint_vel_ramp_iters=(1500, 2500),
    action_rate_weight=-0.05,
    **_DR_KWARGS,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# First-order low-pass instead of a cap: the cheapest thing that could work, and
# the one that bounds nothing. tau = 0.09 s is the continuous-time equivalent of
# the alpha = 0.8 per-control-step filter it is standing in for
# (alpha = exp(-0.02/0.09) = 0.80), written as a time constant so it means the
# same thing at 200 Hz inside the sim and at whatever rate the robot host
# publishes at. A 0.8 rad step then moves 0.16 rad on the first control step
# instead of all of it -- a 5x cut in the PEAK with the steady-state speed
# untouched, which is exactly the difference between this and -Slew.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-Ema",
  env_cfg=flexiv_two_finger_grasp_env_cfg(ema_tau=0.09, **_DR_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, ema_tau=0.09, **_DR_KWARGS),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
  runner_cls=ManipulationOnPolicyRunner,
)

# --- anti-chatter arms -------------------------------------------------------
# SlewCurr won on peak speed and then failed on a quantity nobody measured. Its
# distilled student visibly shakes, and the numbers say why: a slew cap bounds
# SPEED but not DIRECTION CHANGES, so a policy can sit at the cap and reverse
# every few steps. Saturation is what makes it do so -- past the cap only the
# sign of `target - cmd` reaches the sim, magnitude stops receiving gradient and
# drifts outward (measured: requests 16.7x the published rate, actions to 27.0
# against a trained range of |5|, job 61823), and the joint lands in bang-bang.
# The student never settles while carrying: TCP p95 0.74 m/s against its own
# teacher's 0.42 and the unlimited baseline's 0.36.
#
# Two root-cause fixes, ablated separately because they attack different halves:
#
#   SatPen   charge for the overdrive, removing the free lunch that lets the
#            request magnitude run away.
#   CmdObs   show the policy `q_cmd`, the limiter state it has been fighting
#            blind. The original term left it out on the argument that it is
#            recoverable from joint_pos plus the last action -- true only while
#            the limiter can follow, which is precisely when it does not matter.
#   Fix      both, which is the one expected to work if the diagnosis is right.
#   FixRate  both, plus a 5x action-rate penalty, since a saturation penalty
#            prices asking for too much and NOT reversing per se. If chatter
#            survives Fix, this is what says whether it was ever about the
#            limiter at all.
#
# All four keep SlewCurr's cap and schedule so they are comparable to it and to
# each other. CmdObs/Fix/FixRate change the observation dimension 43 -> 54, so
# none of them can warm-start from an existing checkpoint, in either direction.
_ANTI_CHATTER: dict[str, dict[str, Any]] = {
  "SatPen": dict(saturation_weight=-0.02),
  "CmdObs": dict(observe_command=True),
  "Fix": dict(saturation_weight=-0.02, observe_command=True),
  "FixRate": dict(
    saturation_weight=-0.02, observe_command=True, action_rate_weight=-0.05
  ),
  # The saturation weight is the one number here with no measurement behind it.
  # -0.02 against a trained overdrive of ~15 prices the term near 0.3 per step,
  # against `lift` at ~1.27 -- the same order that made the reward-only arm
  # regress into "pinch and press down". This is the hedge: a quarter of the
  # price, so a null result from `Fix` can be read as "wrong weight" or "wrong
  # idea" rather than being ambiguous between them.
  "FixWeak": dict(saturation_weight=-0.005, observe_command=True),
}
_CHATTER_BASE: dict[str, Any] = dict(rate_limit=_SLEW, rate_limit_stages=_SLEW_STAGES)

for _arm, _kw in _ANTI_CHATTER.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(**_CHATTER_BASE, **_kw, **_DR_KWARGS),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_CHATTER_BASE, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth", percept_dr=True, **_CHATTER_BASE, **_kw, **_DR_KWARGS
    ),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth", play=True, percept_dr=True, **_CHATTER_BASE, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=1500),
    runner_cls=ManipulationDistillationRunner,
  )


# --- the student-side fix ----------------------------------------------------
# SatPen's TEACHER is the clean one -- carry dwell 0.73 against the unlimited
# baseline's 0.86, |acc| 0.9 vs 0.8, requested/published 1.5x -- and its student
# throws all of it away: dwell 0.07, |acc| 32.0, and a published command pinned
# at the cap from the MEDIAN step upward (jobs 64722 / 64737).
#
# It throws it away because the saturation penalty that cured the teacher is a
# REWARD, and `Distillation.update` optimises a behaviour-cloning loss that
# never reads rewards. So the fix has to be moved into the student's own
# objective; `SmoothedDaggerDistillation` does that two ways, and the module
# docstring in `mjlab.tasks.manipulation.rl.distillation` has the argument.
#
# The weights are deliberately left at 0 here. Their right scale is set by the
# ratio to the behaviour loss, which is ~0.088 at convergence and cannot be
# guessed from outside a run -- the class logs `saturation` and `action_rate`
# unweighted for exactly that reason, so a short probe sizes them and the real
# run passes them on the CLI.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-SatPen-Smooth",
  env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    percept_dr=True,
    **_CHATTER_BASE,
    **_ANTI_CHATTER["SatPen"],
    **_DR_KWARGS,
  ),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    play=True,
    percept_dr=True,
    **_CHATTER_BASE,
    **_ANTI_CHATTER["SatPen"],
    **_DR_KWARGS,
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=1500, relabel_achievable=True
  ),
  runner_cls=ManipulationDistillationRunner,
)


# Same task with four frames of PROPRIOCEPTION behind the student. The camera
# group stays at one frame on purpose: stacking a 160x120 depth image multiplies
# the CNN input and the image buffer that already forced num_envs to 1024,
# whereas these terms are a few dozen scalars. If a memory the width of the
# teacher's own is enough to average the per-frame perception residual, this is
# where it shows up, and cheaply.
register_mjlab_task(
  task_id=(
    "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-SatPen-Smooth-Hist"
  ),
  env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    percept_dr=True,
    student_history=4,
    **_CHATTER_BASE,
    **_ANTI_CHATTER["SatPen"],
    **_DR_KWARGS,
  ),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    play=True,
    percept_dr=True,
    student_history=4,
    **_CHATTER_BASE,
    **_ANTI_CHATTER["SatPen"],
    **_DR_KWARGS,
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=1500, relabel_achievable=True
  ),
  runner_cls=ManipulationDistillationRunner,
)


# The INCREMENTAL student. Relabelling fixed what the student asks for but left
# the regression badly conditioned: every achievable label sits within one
# limiter allowance of the command, 0.02 rad against the arm's 0.40 rad action
# scale, so 95% of the output range predicts targets that get clipped. Here the
# action IS the step, a full-range action is exactly one allowance, and overdrive
# is not expressible at all rather than merely penalised.
#
# The teacher is UNCHANGED and not retrained: `SmoothedDaggerDistillation` reads
# its absolute output through `scale`/`offset` and converts, both for the label
# and for what the DAgger mixture executes.
#
# Deployment is the open cost -- `deploy/policy.py` integrates nothing today, so
# this checkpoint is not drop-in the way the others are.
register_mjlab_task(
  task_id=(
    "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-SatPen-Smooth-Incr"
  ),
  env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    percept_dr=True,
    incremental=True,
    **_CHATTER_BASE,
    **_ANTI_CHATTER["SatPen"],
    **_DR_KWARGS,
  ),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    play=True,
    percept_dr=True,
    incremental=True,
    **_CHATTER_BASE,
    **_ANTI_CHATTER["SatPen"],
    **_DR_KWARGS,
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=1500, relabel_achievable=True
  ),
  runner_cls=ManipulationDistillationRunner,
)


# --- students for the speed arms ---------------------------------------------
# One distillation id per teacher, because the student's env must match the
# teacher's in every respect that the DAgger rollout touches -- and the action
# term is very much one of those: the teacher labels states reached under ITS
# limiter, so a student rolled out without one is being taught to imitate
# actions that mean something different. Registered for all five and run only
# for whichever teacher wins, at 1500 beta iterations (the schedule the no-DR
# 500 could not carry: under perception DR, training success collapsed
# 0.99 -> 0.27 across iterations 450-550 and spent 500 more climbing back).
_SPEED_ARMS: dict[str, dict[str, Any]] = {
  "Slew": dict(rate_limit=_SLEW),
  "SlewTight": dict(rate_limit=_SLEW_TIGHT),
  "SlewCurr": dict(rate_limit=_SLEW, rate_limit_stages=_SLEW_STAGES),
  "Slow": dict(
    joint_vel_max=0.25,
    joint_vel_penalty_final=-0.5,
    joint_vel_ramp_iters=(1500, 2500),
    action_rate_weight=-0.05,
  ),
  "Ema": dict(ema_tau=0.09),
}

for _arm, _kw in _SPEED_ARMS.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth", percept_dr=True, **_kw, **_DR_KWARGS
    ),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth", play=True, percept_dr=True, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=1500),
    runner_cls=ManipulationDistillationRunner,
  )


# --- grasp POSE arms ---------------------------------------------------------
# Every arm above is scored on success, and success has been 100% since the
# success bonus went in -- so none of them can see that the grasp itself is bad.
# `scripts/wide_grasp_pose_audit.py` measures the pose off the contact points
# and says how, identically on the g1_satpen teacher and its 99.2% student
# (128 envs, carry phase):
#
#   horiz margin  0.3-1.0 mm   a pad pinching ON a vertical edge
#   grip height    +10-11 mm   holding the upper half, not across the middle
#   pad site lat      18 mm    the force lands nowhere near the pad site
#   cube tilt         26 deg   the cube hangs crooked once it is lifted
#
# `contact_face_centring` is the one term that scores all four at once; see its
# docstring for why nothing already in the stack does, and env_cfgs for why the
# kernel is 20 mm rather than a guess.
#
# The arms differ in ONE thing each, so a win is attributable:
#
#   Face      the gate on `pad_touch` + `grasp`, the two terms the alignment
#             gate already established as the right pair to hold (one non-zero
#             from iteration 0 so a gradient exists early, one that requires
#             contact so it cannot be banked).
#   FaceLift  the same plus `lift`, so the CARRY is scored on posture too --
#             the 26 deg tilt happens after lift-off, and nothing currently
#             charges for it.
#   FaceWide  a 40 mm kernel instead of 25 mm. 25 mm is sized to maximise the
#             gradient at the measured 27.6 mm, but the same sizing logic said
#             15 mm for the alignment kernel and that lost 13.5 points by
#             biting before grasping existed. This is the hedge, and it makes a
#             null result readable as "wrong width" rather than "wrong idea".
_FACE_BASE: dict[str, Any] = dict(**_CHATTER_BASE, **_ANTI_CHATTER["SatPen"])
_FACE_ARMS: dict[str, dict[str, Any]] = {
  "Face": dict(face_gate=0.3, face_std=0.025, face_on=("touch", "grasp")),
  "FaceLift": dict(face_gate=0.3, face_std=0.025, face_on=("touch", "grasp", "lift")),
  "FaceWide": dict(face_gate=0.3, face_std=0.040, face_on=("touch", "grasp")),
}

for _arm, _kw in _FACE_ARMS.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(**_FACE_BASE, **_kw, **_DR_KWARGS),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_FACE_BASE, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )


# --- round 2: the same gate, with the hole round 1 fell through closed --------
# p1_Face and p1_FaceLift both improved the horizontal fault they were aimed at
# (`horiz margin` 0.6 -> 6.1 mm) and then bought it with a worse one, because
# the round-1 term scored the horizontal offset ALONE. A contact at the centre
# of the TOP face has almost no horizontal offset, so the audit found
# `pad z spread` 45 mm on a 50 mm cube and `vert margin` 0.0 -- one finger over
# the top and one dug under the cube through the foam, pinching it vertically.
# Side-face contacts 90% -> 58%, cube tilt 25 -> 39 deg, and success stayed at
# 99.2% throughout, so only the pose audit could see any of it.
#
# p1_FaceWide's looser 40 mm kernel never pushed hard enough to find the hole
# and came out a clean win on every column (horiz margin 3.6 mm, jaw yaw
# 5.8 -> 4.3 deg, tilt 25.5 -> 23.3, 100% success), which is what says the idea
# is right and only the shape was wrong.
#
# `face_z_free` / `face_z_std` add the missing axis back as a HINGE rather than
# an error, because the claw NEEDS to grip 10-20 mm high and the exploit lives
# at 22-25 mm; see `contact_face_centring`. The arms vary one thing each:
#
#   FaceZ       the round-1 tight kernel (20 mm) with the hinge. If the hinge is
#               the whole story this is the best arm, since 20 mm gave the
#               largest horizontal gain before it cheated.
#   FaceZWide   p1_FaceWide's proven 40 mm kernel plus the hinge -- the safe bet,
#               improving the one axis that arm left on the table.
#   FaceZHard   20 mm with the hinge biting earlier (14 mm free, 4 mm). Tests
#               whether the grip can be pushed BELOW what the scripted expert
#               manages, or whether 18 mm really is the hardware's floor.
_FACE_Z: dict[str, Any] = dict(face_z_free=0.018, face_z_std=0.005)
_FACE_Z_ARMS: dict[str, dict[str, Any]] = {
  "FaceZ": dict(face_gate=0.3, face_std=0.020, **_FACE_Z),
  "FaceZWide": dict(face_gate=0.3, face_std=0.040, **_FACE_Z),
  "FaceZHard": dict(face_gate=0.3, face_std=0.020, face_z_free=0.014, face_z_std=0.004),
}

for _arm, _kw in _FACE_Z_ARMS.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(**_FACE_BASE, **_kw, **_DR_KWARGS),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_FACE_BASE, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )


# --- round 3: the two things round 2 got wrong -------------------------------
# Round 2 closed round 1's top-face exploit completely (side-face contacts
# 90% -> 97.5-99.7%, `pad z spread` 45 -> 5 mm, opposite faces 49% -> 72-84%)
# and fixed the fault it was aimed at (`horiz margin` 0.6 -> 3.2-7.1 mm), at
# 99.2% success. One column went the wrong way, `cube tilt` 24 -> 30-39 deg, and
# the `jaw tilt` column added afterwards says it was two separate causes:
#
#   THE WRIST ROLL. p2_FaceZHard's 14 mm height hinge is satisfiable by ROLLING
#   THE HAND rather than by approaching differently -- tipping the finger drops
#   the contact for free -- and it rolled 45.7 deg, with the cube hanging off
#   the tilted pinch line at 39.4. Same shape as round 1: constrain one
#   quantity, and a degree of freedom nothing scores absorbs it.
#
#   THE PENDULUM. Round 2 drove grip height to 0 and tilt rose even on the arm
#   that did NOT roll (p2_FaceZ: jaw tilt 19.9, better than the baseline's 21.2,
#   yet cube tilt 30.0 against 24.0). A pinch above the centre of mass
#   self-rights; a pinch through it is neutrally stable, so rotation picked up
#   during the lift stays. The baseline's +9.3 mm was load-bearing and centring
#   threw it away. "Centred" was simply the wrong target on this axis.
#
# Both new pieces are sized from a MEASURED value, not a chosen one: 6 mm from
# the baseline's own self-righting +9.3 mm, and 20 deg from the baseline's own
# jaw tilt of 21.2 -- so neither asks for anything a working policy has not
# already been observed to do, which is the trap this task keeps setting.
_FACE3: dict[str, Any] = dict(face_gate=0.3, face_std=0.020, **_FACE_Z)
_FACE3_ARMS: dict[str, dict[str, Any]] = {
  # The pendulum alone: hold the grip above the COM, leave the roll unscored.
  "FaceBand": dict(**_FACE3, face_z_min=0.006),
  # The roll alone: free to the baseline's 21 deg, charged past it.
  "FaceLevel": dict(**_FACE3, face_level_free=0.35, face_level_std=0.30),
  # Both, which is the arm expected to win if the two causes are independent.
  "FaceBandLevel": dict(
    **_FACE3, face_z_min=0.006, face_level_free=0.35, face_level_std=0.30
  ),
}

for _arm, _kw in _FACE3_ARMS.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(**_FACE_BASE, **_kw, **_DR_KWARGS),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_FACE_BASE, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )


# --- the winner, and its camera-only student ---------------------------------
# p3_FaceLevel is the arm to build on. Against the g1_satpen baseline, carry
# phase, 128 envs, at 100% success (the baseline's own number):
#
#                  baseline   FaceLevel
#   horiz margin      0.6        5.3 mm   pinching an edge -> not
#   cube tilt        24.0       11.1 deg  carried crooked -> nearly level
#   jaw tilt         21.2        8.0 deg
#   jaw yaw           6.0        2.3 deg  the wrist finally tracks the cube
#   grip force       14.1       18.0 N
#   OPPOSITE faces   49.4%      79.1%
#   on a SIDE face   90.2%      97.1%
#   friction         14.4       12.2 deg  less of the grip carried by luck
#
# AND IT FALSIFIES THE PENDULUM ARGUMENT THAT MOTIVATED `face_z_min`. That term
# was added on the reasoning that a pinch above the centre of mass self-rights
# and one through it does not, so round 2's centred grip explained the tilt
# regression. The arms say otherwise, and clearly:
#
#   FaceLevel       no z_min, grip height -2.2 mm (through the COM), tilt 11.1
#   FaceBandLevel   z_min on,  grip height +9.5 mm (above it),      tilt 34.6
#   FaceBand        z_min alone: jaw tilt 44.6 deg, tilt 36.8, and `horiz
#                   margin` back to 0.2 mm -- the worst arm of the nine
#
# So cube tilt tracks JAW TILT and essentially nothing else, and `face_z_min` is
# actively harmful: pushing the grip up is another quantity the policy satisfies
# by ROLLING THE WRIST, which is the same hack for the third time. It is kept
# only as a defaulted-off parameter so the finding stays reproducible; do not
# turn it on.
_FACE_LEVEL: dict[str, Any] = dict(
  face_gate=0.3, face_std=0.020, face_level_free=0.35, face_level_std=0.30, **_FACE_Z
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-FaceLevel-Smooth",
  env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", percept_dr=True, **_FACE_BASE, **_FACE_LEVEL, **_DR_KWARGS
  ),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth", play=True, percept_dr=True, **_FACE_BASE, **_FACE_LEVEL, **_DR_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=1500, relabel_achievable=True
  ),
  runner_cls=ManipulationDistillationRunner,
)

# The same student with four frames of PROPRIOCEPTION behind it. Carried over
# from the anti-chatter line, where it was the only arm that cost nothing:
# s_g1_hist took success 99.2 -> 100.0% at s_g1_relabel's smoothness. Here it
# has 2.3% to recover, since s_facelevel scores 97.7% against its teacher's
# 100.0%. It does NOT address the chatter -- s_g1_hist still reverses 10.45
# times a second -- because the per-frame residual enters through the CAMERA and
# proprioception cannot average that. Its role in this batch is to raise the
# ceiling the RL arm below then has an incentive to reach for.
register_mjlab_task(
  task_id=(
    "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-FaceLevel-Smooth-Hist"
  ),
  env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    percept_dr=True,
    student_history=4,
    **_FACE_BASE,
    **_FACE_LEVEL,
    **_DR_KWARGS,
  ),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    "depth",
    play=True,
    percept_dr=True,
    student_history=4,
    **_FACE_BASE,
    **_FACE_LEVEL,
    **_DR_KWARGS,
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(
    beta_decay_iters=1500, relabel_achievable=True
  ),
  runner_cls=ManipulationDistillationRunner,
)


# PPO ON THE DISTILLED STUDENT. Same env as the distillation above -- it already
# carries all four observation groups and the full reward manager, whose ten
# terms are computed on every DAgger step and discarded, because a behaviour-
# cloning loss never reads rewards. Here they are the objective.
#
# The gap this exists to close, carry phase (jobs 67452 / 67455):
#
#                       dwell   |acc|   tcp jerk   rev/s
#   teacher FaceLevel    0.65     0.7          7    0.29
#   student s_facelevel  0.18     7.6         30   10.36
#
# The student's residual against its teacher is 1.6 deg per joint and TEMPORALLY
# INDEPENDENT, so it becomes ~2 rad/s of command rate against a 1.0 rad/s cap.
# No amount of further BC touches it: the loss is per-timestep and scores a
# steady 1.6 deg offset exactly the same as a +-1.6 deg alternation.
#
# `s_g1_ramp` already proved the target is REACHABLE with camera input -- it hit
# |acc| 2.8 against the teacher's 0.7 -- and that a BC objective cannot afford
# it, since success fell to 64.8%. Under PPO the task reward is what pays.
def _hold_forever(cfg):
  """Drop the success TERMINATION so holding the cube keeps paying.

  MEASURED, job 67482 -- the first fine-tune, which collapsed to 0.0% success
  with 120 of 128 envs never lifting at all:

                       teacher   s_fl_rl
    mean episode len       87       966     of a 1000 cap
    lift (dense)        0.148      1.19     8x, purely from the longer episode
    cube_lifted ends       48         0     never fired after the first ~20 iters
    lift_hold          0.0014    0.0000

  `cube_lifted` ENDS the episode, so `lift_hold` -- which pays 2.0 EVERY STEP
  while the cube is gripped above the line -- can only be collected for the one
  second of hold before the episode is cut off. Parking just under the line
  collects the dense stack for all 1000 steps instead. The policy stretched its
  episodes 11x and harvested exactly that.

  This is the exploit `lift_hold_bonus` and the `success_reward` block in
  env_cfgs.py were written to describe, and they do not close it: making the
  termination a time_out restores the bootstrap and paying per step raises the
  price of holding, but BOTH are dominated as long as succeeding still stops the
  clock. Those mitigations were enough for a from-scratch run, which learns to
  lift before it can discover hovering. A fine-tune starts from a policy that
  already lifts, so the first thing its local search finds is the hover.

  With the termination gone the comparison is per-step and one-sided: holding
  above the line pays 2.0/step against the dense stack's ~0.056/step, for the
  rest of the episode either way. `cube_dropped` is deliberately KEPT, so
  letting go still ends the money.

  This also matches how the policy is scored: the audits run 300 steps with
  terminations off, i.e. they already measure "lift and keep holding".
  """
  del cfg.terminations["cube_lifted"]
  _clip_action_obs(cfg)
  return cfg


# The raw action's trained range is about |5| (see `action_saturation_penalty`,
# which measured the SlewCurr teacher running to 27.0), so this is a no-op on
# anything the distilled student actually does and only bites on a runaway.
_ACTION_OBS_CLIP = (-5.0, 5.0)


def _clip_action_obs(cfg) -> None:
  """Bound the fed-back raw action in the STUDENT observation.

  Necessary because of the normalizer freeze, not despite it. Until now a
  runaway action was absorbed -- disastrously -- by the running std growing to
  meet it (job 68179; see `_freeze_obs_normalizer`). With the std pinned at the
  distilled value, the same runaway would instead reach the network as an input
  of several hundred, and a warm-started MLP has no defence against that.

  Why a runaway is possible at all: past the slew cap only the SIGN of the
  request reaches the simulation, so the magnitude receives no gradient and is
  free to drift outward. `action_saturation` is the term that prices that drift,
  and on job 68179 it was paying -0.018/step against `lift`'s +1.2 -- 1.5% of
  the signal. Deliberately NOT retuned here: this clip removes the mechanism
  that turned the drift into a collapse, and raising the weight is a separate
  variable to change separately if the action still wanders.

  The clip applies to the STUDENT group only. The teacher's `actor` group is
  the vector its frozen weights were fitted to and is never touched.
  """
  term = cfg.observations["student"].terms.get("actions")
  if term is None:
    raise KeyError("the student observation group has no `actions` term to clip")
  term.clip = _ACTION_OBS_CLIP


register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFingerWide-Flexiv-Finetune-Depth-Success-Dr-FaceLevel",
  env_cfg=_hold_forever(
    flexiv_two_finger_distill_env_cfg(
      "depth", percept_dr=True, **_FACE_BASE, **_FACE_LEVEL, **_DR_KWARGS
    )
  ),
  play_env_cfg=_hold_forever(
    flexiv_two_finger_distill_env_cfg(
      "depth", play=True, percept_dr=True, **_FACE_BASE, **_FACE_LEVEL, **_DR_KWARGS
    )
  ),
  rl_cfg=flexiv_two_finger_finetune_runner_cfg(),
  runner_cls=ManipulationFinetuneRunner,
)

# Round 4: the level term removed the roll hack, so the horizontal kernel can be
# tightened without it being absorbed elsewhere -- `horiz margin` is still only
# 5.3 mm of an available 24.8. One variable each.
_FACE4_ARMS: dict[str, dict[str, Any]] = {
  "LevelTight": dict(**{**_FACE_LEVEL, "face_std": 0.012}),
  "LevelFlat": dict(**{**_FACE_LEVEL, "face_level_free": 0.0}),
  "LevelTightFlat": dict(**{**_FACE_LEVEL, "face_std": 0.012, "face_level_free": 0.0}),
}

for _arm, _kw in _FACE4_ARMS.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(**_FACE_BASE, **_kw, **_DR_KWARGS),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_FACE_BASE, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )


# =============================================================================
# COACD + wider domain randomization.  Round 1.
# =============================================================================
# The collision representation moves to `coacd_t0.2` here, and it is a TASK
# PARAMETER rather than an environment variable on purpose: it changes the
# physics (the same full-close command grips at 12.4 N on boxes and 7.0 N on
# hulls), so a checkpoint is only valid under the setting it was trained with,
# and an env var would make that mismatch silent. Measured cost of the switch:
# 170.6k -> 164.1k env-steps/s at 4096 envs, i.e. 3.8%.
#
# ONE VARIABLE PER ARM. A0/B0 carry no DR change at all, which is what makes the
# rest readable: without them a regression in A1-A4 cannot be attributed to the
# DR rather than to the collision switch.
_COACD: dict[str, Any] = dict(collision="coacd_t0.2")
# GATED, not chosen. `scripts/wide_spawn_gate.py` sweeps the box against the
# D435i and the arm's action envelope; +-110/+-140 FAILS on frame coverage
# (worst frame coord 1.053, i.e. the cube leaves the image at the far corner)
# while the range and reach are fine. This box reads 0.901 -- 10% of margin,
# which the +-2.5 deg camera-yaw DR then eats into. +-105/+-128 was 0.964 and
# too tight for that.
#   this box:  nearest corner 235.6 mm (min 200), frame 0.901, reach 0.78 sigma
_WIDE_SPAWN = (0.100, 0.120)  # from +-80/+-100
_WIDE_SIZE = (0.85, 1.15)  # 42.5-57.5 mm; jaw clears the diagonal to 61.4 mm
_WIDE_MASS = (0.03, 0.30)  # kg; friction capacity is 7-16.8 N against 2.94 N

_R1_ARMS: dict[str, dict[str, Any]] = {
  "R1-SatPen": dict(),
  "R1-SatPen-Spawn": dict(spawn_jitter=_WIDE_SPAWN),
  "R1-SatPen-Size": dict(cube_scale_range=_WIDE_SIZE),
  "R1-SatPen-Mass": dict(cube_mass_range=_WIDE_MASS),
  "R1-SatPen-All": dict(
    spawn_jitter=_WIDE_SPAWN, cube_scale_range=_WIDE_SIZE, cube_mass_range=_WIDE_MASS
  ),
}
_R1_FACE: dict[str, dict[str, Any]] = {
  "R1-FaceLevel": dict(),
  "R1-FaceLevel-All": dict(
    spawn_jitter=_WIDE_SPAWN, cube_scale_range=_WIDE_SIZE, cube_mass_range=_WIDE_MASS
  ),
}

for _arm, _kw in _R1_ARMS.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(
      **_FACE_BASE, **_COACD, **_kw, **_DR_KWARGS
    ),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_FACE_BASE, **_COACD, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )

for _arm, _kw in _R1_FACE.items():
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(
      **_FACE_BASE, **_FACE_LEVEL, **_COACD, **_kw, **_DR_KWARGS
    ),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_FACE_BASE, **_FACE_LEVEL, **_COACD, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )

# Distillation of the two round-1 combined arms, both with `relabel_achievable`
# -- established as behaviour-preserving and strictly better, so it is not a
# variable here.
for _arm, _kw in (
  ("R1-SatPen-All", {**_R1_ARMS["R1-SatPen-All"]}),
  ("R1-FaceLevel-All", {**_FACE_LEVEL, **_R1_FACE["R1-FaceLevel-All"]}),
):
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-{_arm}",
    env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth", percept_dr=True, **_FACE_BASE, **_COACD, **_kw, **_DR_KWARGS
    ),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth", play=True, percept_dr=True, **_FACE_BASE, **_COACD, **_kw, **_DR_KWARGS
    ),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(
      beta_decay_iters=1500, relabel_achievable=True
    ),
    runner_cls=ManipulationDistillationRunner,
  )


# =============================================================================
# Small objects.  Round 2.
# =============================================================================
# Four COUPLED changes, which is exactly why they are not folded into round 1 --
# a regression there would be unattributable among six knobs.
#
#   finger range   lower bound -0.0754 -> -0.32. Measured with
#                  `scripts/wide_finger_profile.py`: the jaw is narrowest ~12-16
#                  mm above the tip, so a 15 mm object is pinched there, and
#                  reaching 15 mm of separation needs about -0.26. The bound
#                  stops at -0.32 because the fingers CROSS by -0.38/-0.45.
#   finger scale   0.35 -> 0.45. At 0.35 that closure is -1.70 sigma, past the
#                  ~1.5 this env has lost four runs to; at 0.45 it is -1.32.
#   cube size obs  REQUIRED, see `object_half_extent`. Without it a size-blind
#                  full close becomes ~0.24 rad of over-travel on a 50 mm cube.
#   size range     0.30-1.15 = 15-57.5 mm.
#
# The arms separate the two that could each break the base task on their own
# (the reach change, and the size range) from their combination.
_R2_REACH: dict[str, Any] = dict(
  finger_range=((-0.32, 1.60), (-1.60, 0.10)),
  finger_scale=0.45,
  observe_object_size=True,
)
_R2_BASE: dict[str, Any] = dict(
  **_FACE_BASE, **_COACD, spawn_jitter=_WIDE_SPAWN, cube_mass_range=_WIDE_MASS
)

_R2_ARMS: dict[str, dict[str, Any]] = {
  # The reach change alone, on the CURRENT size range: does opening the finger
  # bound and rescaling the action break a task that already works?
  "R2-Reach": dict(**_R2_REACH),
  # The full small-object range.
  "R2-Small": dict(**_R2_REACH, cube_scale_range=(0.30, 1.15)),
}

for _arm, _kw in _R2_ARMS.items():
  _kwargs = {k: v for k, v in V11_ALIGN_KWARGS.items() if k != "finger_range"}
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-{_arm}",
    env_cfg=flexiv_two_finger_grasp_env_cfg(
      **_R2_BASE, **_kw, success_reward=2.0, physics_dr=True, **_kwargs
    ),
    play_env_cfg=flexiv_two_finger_grasp_env_cfg(
      play=True, **_R2_BASE, **_kw, success_reward=2.0, physics_dr=True, **_kwargs
    ),
    rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=_STD, entropy_coef=_ENT),
    runner_cls=ManipulationOnPolicyRunner,
  )

# Distillation of both round-2 arms. R2-Small is the one the round is for, but
# it cannot be read on its own: if its student regresses there are two candidate
# causes -- a 15 mm cube is 3-6 px at 160x120 and may simply not be visible, or
# the wider finger range and rescaled action are harder to imitate. R2-Reach
# carries the reach change on the CURRENT size range, so it holds the second
# cause fixed and the pair separates them.
#
# The teacher kwargs are the same expression as the teacher registration above,
# not a re-derivation: the `actor` observation group has to be bit-identical to
# the vector the teacher's weights were fitted to.
for _arm, _kw in _R2_ARMS.items():
  _kwargs = {k: v for k, v in V11_ALIGN_KWARGS.items() if k != "finger_range"}
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-{_arm}",
    env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth",
      percept_dr=True,
      **_R2_BASE,
      **_kw,
      success_reward=2.0,
      physics_dr=True,
      **_kwargs,
    ),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth",
      play=True,
      percept_dr=True,
      **_R2_BASE,
      **_kw,
      success_reward=2.0,
      physics_dr=True,
      **_kwargs,
    ),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(
      beta_decay_iters=1500, relabel_achievable=True
    ),
    runner_cls=ManipulationDistillationRunner,
  )

# The same two students with the object's SIZE withheld.
#
# `observe_object_size` was added for the teacher, which reads privileged state
# by definition. The student inherited it only because `_PRIVILEGED_TERMS`
# predates the term, so the round-2 students above are handed the exact half-edge
# of the cube they are supposed to be looking at -- and at zero noise under
# `play`, since the student group's corruption is off there. Their small-object
# result therefore answers "can it grasp a 15 mm cube it has been TOLD is 15 mm",
# which is a different and easier question than the one the camera is for.
#
# These withhold it. Same teacher, same physics, same rewards: the ONLY
# difference from the arm above is one scalar in the student's observation, so
# the pair prices what reading size off a 160x120 depth image is worth. The
# teacher's `actor` group is untouched either way -- `size_privileged` reaches
# only the student -- so both arms distil from the same R2 checkpoints.
#
# A side effect worth having: these students are 34-dimensional, which is the
# layout `deploy/policy.py` already assembles. The 35-dimension ones cannot be
# deployed without inserting a measured object size into the observation vector
# between the last action and the goal height.
for _arm, _kw in _R2_ARMS.items():
  _kwargs = {k: v for k, v in V11_ALIGN_KWARGS.items() if k != "finger_range"}
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-{_arm}-SizeBlind",
    env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth",
      percept_dr=True,
      size_privileged=True,
      **_R2_BASE,
      **_kw,
      success_reward=2.0,
      physics_dr=True,
      **_kwargs,
    ),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(
      "depth",
      play=True,
      percept_dr=True,
      size_privileged=True,
      **_R2_BASE,
      **_kw,
      success_reward=2.0,
      physics_dr=True,
      **_kwargs,
    ),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(
      beta_decay_iters=1500, relabel_achievable=True
    ),
    runner_cls=ManipulationDistillationRunner,
  )
