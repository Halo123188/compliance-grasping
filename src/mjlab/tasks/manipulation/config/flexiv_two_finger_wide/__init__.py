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
