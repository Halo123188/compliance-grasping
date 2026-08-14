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

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv",
  env_cfg=flexiv_two_finger_grasp_env_cfg(),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Goal directly above the cube, so the reward's position error is purely the lift
# height -- matching what `cube_lifted` actually scores -- with a bringing kernel
# narrowed to that 8-19 cm range. Under the stock "dynamic" goal the policy learns
# a firm two-finger grasp and then just holds the cube down on the table.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal",
  env_cfg=flexiv_two_finger_grasp_env_cfg(goal_mode="above_object", bringing_std=0.12),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True, goal_mode="above_object", bringing_std=0.12
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# --- Why the LiftGoal jaw closes wrong, three independent candidates ----------
# Run 45530 (LiftGoal, 3000 iters) got the first non-zero grasp reward this task
# has produced (0.449) and still lifted nothing: the hand settles skewed, with
# the knuckles jammed 0.21 mm apart, the pads 24 mm apart, and both distals
# curled OUTWARD -- the mirror image of the curl the pull-out test measured at
# 5.2-6.1 N holding force.
#
# Three things could each cause that, and they are not alternatives to each
# other -- all three are true of the current config, so each gets its own arm
# against the plain LiftGoal baseline (which now also carries the pad-site fix,
# so the baseline is NOT run 45530 and must be rerun to compare).
#
# Deliberately NOT tested: coupling the two fingers onto one action. It would
# make the skewed pose unrepresentable, but two real motors are never exactly
# mirrored, so a policy that can only command symmetric pairs has no way to
# express what the hardware actually does. R14Mirror is the soft version.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-Antipodal",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object", bringing_std=0.12, grasp_shaping="antipodal"
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    grasp_shaping="antipodal",
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-Mirror",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object", bringing_std=0.12, symmetry_penalty=-0.5
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True, goal_mode="above_object", bringing_std=0.12, symmetry_penalty=-0.5
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Left-finger (proximal, distal) travel; mirrored onto the right inside the cfg.
_RANGE = ((-0.20, 1.60), (-0.85, 1.60))

# Left-finger range; mirrored onto the right inside the cfg. -0.20 is where the
# mirrored jaw is shut (9.9 mm) and where the colliders first touch; the upper
# bound is left at the XML's 1.6 because opening wide is harmless. The grip pose
# settles at left_1 = +0.23, so this leaves plenty of margin on the closing side.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-Range",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)


# --- Why the grip lands high and on a corner, three candidates ---------------
# Measured on v2_range's 3000-iter policy (the retired diag_grasp_alignment.py, 64
# envs): the pad midpoint ends a median 17.2 mm ABOVE the cube centre -- 69% of
# the way up a 25 mm half-height, so on the top edge -- and the jaw sits a median
# 13.9 deg off a cube face, with 16% of episodes past 30 deg, nearer a corner
# than a face. A flat 28 mm pad tilted 14 deg bears on one pad CORNER, which is
# point contact, and a cube held on a point tips and slides out.
#
# Nothing in the reward stack constrains either: `lift` measures reaching from
# grasp_site on the PALM with a 20 cm kernel, so a 17 mm height error is nearly
# free, and no term mentions orientation at all. All three arms below run on the
# Range variant, which is the best of the previous round, plus the fingertip
# pad sites -- so each differs from v3_tipsite_range by exactly one thing.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-ReachPads",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object", bringing_std=0.12, finger_range=_RANGE, reach_from="pads"
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
    reach_from="pads",
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-Align",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object", bringing_std=0.12, finger_range=_RANGE, align_weight=1.0
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
    align_weight=1.0,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Same term at weight 0.15 instead of 1.0. At 1.0 (run 45657) the policy simply
# stopped grasping: jaw_alignment reached 0.946 while pad_pinch collapsed from
# 0.555 to 0.033 and grasp went to exactly 0. Rolling the wrist is cheap and
# grasping is hard, so at equal weight the cheap term is the whole return. 0.15
# caps it at ~9% of what a grasp is worth, which is a tiebreak rather than a
# living.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-Align015",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
    align_weight=0.15,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
    align_weight=0.15,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# z_std 0.06 -> 0.045. NOT tighter: the pad midpoint starts ~100 mm above the
# cube, which is 1.67 sigma at 0.06 and 2.2 at 0.045. At 0.03 it would be 3.3
# sigma and the term would read ~0 from the start pose -- the failure that killed
# the antipodal arm in run 45593. 0.045 makes a 17 mm error cost 0.14 of the term
# instead of 0.08 while staying reachable.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TightZ",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object", bringing_std=0.12, finger_range=_RANGE, pinch_z_std=0.045
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_RANGE,
    pinch_z_std=0.045,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)


# --- Stop constraining the pose, score the fingertips instead ----------------
# Everything above shapes the grasp through a proxy -- pad separation, pad
# midpoint, jaw yaw -- and each proxy turned out to be satisfiable with the
# fingertips nowhere near the cube. The pad-plane theory that motivated the
# `Align` arms was wrong on top of that: measured on the mesh
# (the retired diag_finger_shape.py), the finger is a hook whose tuned tip site
# reaches the cube by SPREADING the proximal and CURLING the distal, and the
# `Range` limits below cap the distal at -0.85, which blocks exactly that.
#
# What the tip sites actually do, swept on the model:
#
#   proximal  distal   site separation   site height
#     +0.50   -0.40        104.8 mm         -107.0   <- the hover start
#     +0.50   -1.05         50.4 mm          -99.9   <- pinches the 50 mm cube
#     +0.60   -1.25         50.9 mm          -95.0
#     +0.75   -1.55         51.7 mm          -86.3
#
# so the grasp family is roughly distal ~ -2 * proximal - 0.05, and reaching it
# from the hover is a distal move of -0.65, which at scale 0.6 is -1.08 sigma --
# inside the explorable band, unlike the -1.83 sigma the old deep-curl idea
# needed. Two arms, both scoring `pad_site_touch` (the tuned tip ON the cube
# surface, product of the two sides) instead of a proxy.

# No finger limits at all: back to the XML's +-1.6, which is what the hand
# shipped with before any of the hand-picked ranges here.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchFree",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object", bringing_std=0.12, touch_weight=1.0
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True, goal_mode="above_object", bringing_std=0.12, touch_weight=1.0
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# One-way motors: the proximal may only spread OUTWARD from the hover, the distal
# may only curl INWARD. Directions measured, not assumed -- for the left finger
# proximal-positive opens (site separation 13.6 mm at -0.20 rising to 226 mm at
# +1.60) and distal-negative closes (186 mm at +1.20 down to 19 mm at -1.60).
# Mirrored onto the right inside the cfg. This makes the scissoring, one-finger
# and crossed poses unrepresentable without forbidding any pose in the grasp
# family above: the hover (+0.50, -0.40) is the corner of the box, and the
# pinch (+0.50, -1.05) is inside it.
_DIRECTIONAL = ((0.50, 1.60), (-1.60, -0.40))

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchOneWay",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)


# TouchOneWay again, with the reaching kernel moved off the PALM and onto the two
# pad sites. Run 46846 (v8) ran the palm version for 3000 iterations and the
# rollout shows why it stalled: the hand parks square over the cube with the
# fingers open and the fingertips ABOVE the cube's top face, and never descends
# (videos/v8_strip.png -- the cube is untouched for 213 of 300 frames). Closing
# from there would only pinch air, so the policy correctly leaves the fingers
# alone, and pad_touch plateaus at 0.22 with grasp at exactly 0.
#
# `lift` is weight 1.0 and the largest term, and it measures from grasp_site on
# the PALM through a 20 cm kernel, so hovering 5 cm high costs almost nothing.
# That is the documented `fingertip_proximity` loophole -- a palm site is a point
# the hand can hold on the object while the fingertips stay clear -- and no
# amount of tuning the small grasp terms outvotes the big term pointing at the
# wrong place. reaching_std stays 0.20: an earlier attempt moved the kernel to
# the pads AND narrowed it, and the whole lift reward collapsed (smoke 45443).
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchOneWayPads",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)


# Same env as TouchOneWayPads, but with the exploration knobs turned down. The
# reward and the mechanics have both now been cleared by measurement:
#
#   * the grasp at (+0.50, -1.05) holds 19.75 N -- 40x the cube's weight -- and
#     it holds at EVERY jaw-vs-cube yaw from 0 to 45 deg, because the fingers
#     rotate the cube square as they close (at 45 deg the cube turns -41.4 deg).
#     The earlier "0.5 N pull-out, contact topology is the wall" conclusion was
#     measured at the wrong closing depth (-0.40..-0.85) with the cube floating
#     in zero gravity, and is retracted.
#   * replaying the policy's own finger flailing moves the cube 17.5 mm and
#     leaves it on the table, so closing is not being suppressed by a fear of
#     knocking the cube away either.
#
# What is left is the exploration itself: iid per-step noise at std 1.95 gives
# the finger a new random target every control step, which the servo low-passes
# into nothing. See the note in rl_cfg.flexiv_two_finger_grasp_ppo_runner_cfg.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalm",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# --- Two arms off TouchCalm, the first config to lift anything (26% success) ---
# Both change TouchCalm by exactly one thing, so either result is attributable.

# (a) Stop punishing success. See the long note at cfg.terminations["cube_lifted"]
# in env_cfgs: succeeding ends the episode with the bootstrap zeroed and pays
# nothing, forfeiting ~26 of a ~52 return. Weight 10 against a stack that is
# otherwise all 1.0 is the magnitude ladder this reward set has never had -- the
# per-step dense terms cap around 5.0 together, so holding the cube up is finally
# worth more than parking next to it.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalmSuccess",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    success_reward=10.0,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    success_reward=10.0,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)

# (b) Wrist alignment as a MULTIPLIER on the grasp reward rather than a term of
# its own -- additive failed at 1.0 (hacked) and at 0.15 (ignored). floor 0.3, so
# a fully misaligned jaw still earns 30% of the grasp reward and squaring up is
# worth 3.3x. Note the mechanics do not obviously need this: a static close holds
# 19.75 N at every yaw 0-45 deg because the fingers rotate the cube square. What
# is untested is whether that 41 deg rotation and 17 mm shove is survivable
# during a moving approach, which is what the 74% of failed episodes might be.
# The kwargs of the best policy to date, `v11_align` (run 46984, 80.5%
# deployment success), pulled out as a constant because the distillation tasks
# below have to reproduce this env EXACTLY: their teacher observation group is
# whatever this cfg builds, and the teacher's weights were fitted to that
# vector in that order. Editing these kwargs changes what a v11_align
# checkpoint is being fed.
V11_ALIGN_KWARGS: dict[str, Any] = dict(
  goal_mode="above_object",
  bringing_std=0.12,
  finger_range=_DIRECTIONAL,
  touch_weight=1.0,
  reach_from="pads",
  align_gate=0.3,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalmAlign",
  env_cfg=flexiv_two_finger_grasp_env_cfg(**V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, **V11_ALIGN_KWARGS),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# --- Three arms off TouchCalmAlign (36% success), one change each -------------
# Failure analysis of run 46984 (the retired diag_failure_modes.py, 64 envs,
# deterministic policy): 81% succeed, and NOTHING fails from losing the grasp --
# dropped, slipped-early, knocked-away and never-gripped are all exactly 0. Every
# failure is a lift that comes up short, and yaw error is the discriminator:
#
#     success       52   median peak 107.5 mm   median yaw err   8.9 deg
#     hold-timeout   3   median peak 108.8 mm   median yaw err  15.0 deg
#     short-lift     9   median peak  91.1 mm   median yaw err  25.1 deg
#
# and corr(yaw error, peak height) = -0.360. Note the arm-posture story does NOT
# hold: corr(yaw error, reach) = -0.025 and corr(reach, peak) = +0.566, i.e.
# reaching further goes with lifting HIGHER, the opposite way round.
# (A) Tighter kernel. The short-lift failures sit at 25 deg, which at std 0.436
# still collects exp(-1) = 0.37 of the gate -- not painful enough. At 0.26
# (15 deg) the same 25 deg error is worth 0.06.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-AlignTight",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_std=0.26,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_std=0.26,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)

# (B) Gate the carry as well as the grip. `lift` is the largest term and is a
# pure position kernel -- nothing in it cares what orientation the cube is
# carried in.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-AlignLift",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_on=("grasp", "lift"),
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_on=("grasp", "lift"),
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)

# (C) Gate `pad_touch` too, which is the one that unfreezes the wrist. Measured
# on 46984: joint7 parks at +19.1 deg with a 4.3 deg spread while cube yaw spans
# the full circle, corr(cube yaw, joint7) = -0.089 -- the wrist roll is not
# aligning anything, and its action std had collapsed to 0.0083 (+-0.5 deg). It
# froze because `grasp` reads 0.0000 for the first ~2000 iterations, so a gate
# multiplied only onto `grasp` gives alignment NO gradient during exactly the
# window in which the movement penalties squeeze an unrewarded joint to a stop.
# `pad_touch` is ~0.2 from the start, so gating it puts an alignment gradient in
# front of the wrist while the wrist can still move.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-AlignTouch",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_on=("grasp", "touch"),
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_on=("grasp", "touch"),
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# (D) The tight kernel as a SCHEDULE rather than a setting. Run 47063 held 15 deg
# for the whole run and lost 13.5 points (22.5% vs the 36.0% baseline), because a
# tight gate multiplies the already-sparse grasp reward by ~0.1 during the window
# where grasping still has to be discovered. Staged 25 -> 20 -> 15 deg at
# iterations 0 / 1000 / 2000 it costs nothing early: `grasp` is 0.0000 at
# iteration 600 and 0.57 by ~900, and success does not leave zero until ~2000, so
# the kernel is still wide while grasping is being found and is tight for the
# last third, when there is a grasp worth sharpening.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-AlignCurr",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_curriculum=True,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    align_curriculum=True,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# --- Two arms off TouchCalmAlign that attack EXPLORATION, not the reward -----
# All four alignment-reward variants above lost to the baseline, and the reason
# is visible in the policy rather than the reward: joint7 is parked (+24.1 deg,
# 3.1 deg spread across 128 envs spanning the full yaw circle) with an action std
# of 0.0083 rad. A reward cannot move a joint whose exploration has collapsed.
#
# The payoff is real and bounded (scripts/probe_joint7_value.py): locking cube
# yaw square to the frozen wrist takes success 78.1% -> 89.1%, and the gain sits
# entirely in the 25% of envs whose yaw error exceeds 22.5 deg, where success
# falls off a cliff to 60% and 50%. Below 22.5 deg alignment does not matter at
# all (85.1% vs 86.2%).
#
# (E) Stop charging joint7 for moving. During the ~2000 iterations when `grasp`
# reads 0.0000, action_rate_l2 and joint_vel_hinge are the only gradients on the
# wrist, and both point at "hold still".
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-FreeWrist",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    free_wrist_penalties=True,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    free_wrist_penalties=True,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# (F) Randomize joint7's reset angle over +-45 deg, the exact span squaring the
# jaw to an arbitrary cube yaw needs. Removing a penalty only permits variance;
# this injects it, so the critic learns what wrist angle is worth from iteration
# 0 rather than waiting on an actor whose std has already collapsed. The project
# precedent is that exploration, not reward shape, was what unlocked this task:
# init_std 1.5 -> 0.8 took `grasp` from 0.0000 (16 consecutive runs) to 0.483.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-WristRand",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    wrist_reset_range=0.785,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    wrist_reset_range=0.785,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# (G) Stop learning the wrist and SOLVE it. d(jaw yaw)/d(joint7) = -1.000 exactly
# (scripts/diag_wrist_sign.py), so the angle that squares the jaw is
# `q7 + fold(jaw_yaw - cube_yaw)`, re-solved from measured state every substep.
# The policy keeps a 0.15 rad trim on joint7 -- inside the 0-22.5 deg band where
# the probe showed alignment does not affect success at all -- so it cannot undo
# the solution but still has a gradient there. Everything else is v11_align.
#
# This uses the cube's pose, which the state task already has as a privileged
# observation. It does NOT transfer to the vision variants, where yaw has to be
# estimated from the image.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-SolvedWrist",
  env_cfg=flexiv_two_finger_grasp_env_cfg(
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    solve_wrist=True,
  ),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(
    play=True,
    goal_mode="above_object",
    bringing_std=0.12,
    finger_range=_DIRECTIONAL,
    touch_weight=1.0,
    reach_from="pads",
    align_gate=0.3,
    solve_wrist=True,
  ),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(init_std=0.8, entropy_coef=0.005),
  runner_cls=ManipulationOnPolicyRunner,
)


# Ablation: the stock joint_vel_hinge ramp (down to -1.0 at iter 1000) instead of
# the capped -0.1 default. The last state run regressed from brief lifts into
# "pinch and press down" exactly as that ramp landed; this pins down whether the
# ramp is the cause.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-StockVel",
  env_cfg=flexiv_two_finger_grasp_env_cfg(joint_vel_penalty_final=-1.0),
  play_env_cfg=flexiv_two_finger_grasp_env_cfg(play=True, joint_vel_penalty_final=-1.0),
  rl_cfg=flexiv_two_finger_grasp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-Rgb",
  env_cfg=flexiv_two_finger_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=flexiv_two_finger_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=flexiv_two_finger_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-Depth",
  env_cfg=flexiv_two_finger_vision_env_cfg(cam_type="depth"),
  play_env_cfg=flexiv_two_finger_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=flexiv_two_finger_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-Rgbd",
  env_cfg=flexiv_two_finger_vision_env_cfg(cam_type="rgbd"),
  play_env_cfg=flexiv_two_finger_vision_env_cfg(cam_type="rgbd", play=True),
  rl_cfg=flexiv_two_finger_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)


# --- DAgger: distil v11_align into a camera-only student ---------------------
# The three vision tasks above learn from scratch with PPO, which asks a CNN to
# discover the grasp AND the perception at the same time. These instead freeze
# the state policy and regress a camera student onto its actions, so the only
# thing being learned is "recover from the image what the teacher was told".
#
# The env is v11_align verbatim (V11_ALIGN_KWARGS) plus a camera and a proprio
# "student" group. Run with:
#
#   train.py Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth \
#     --teacher-checkpoint <.../v11_align/model_2999.pt>
#
# Rewards still tick and still get logged, but nothing optimizes them --
# `Metrics/lift_height/episode_success` is the number to watch, and it is the
# student's, since the student is what acts.
for _cam in ("depth", "rgb", "rgbd"):
  register_mjlab_task(
    task_id=f"Mjlab-Grasp-TwoFinger-Flexiv-Distill-{_cam.capitalize()}",
    env_cfg=flexiv_two_finger_distill_env_cfg(cam_type=_cam, **V11_ALIGN_KWARGS),
    play_env_cfg=flexiv_two_finger_distill_env_cfg(
      cam_type=_cam, play=True, **V11_ALIGN_KWARGS
    ),
    rl_cfg=flexiv_two_finger_distill_runner_cfg(),
    runner_cls=ManipulationDistillationRunner,
  )

# DAgger mixing arm: execute the TEACHER's action with probability beta,
# annealed 1 -> 0 over the first 500 iterations. Without it the student drives
# from iteration 0, knocks the cube off the table within a few steps, and every
# subsequent label is the teacher's opinion about a cube on the floor -- the
# behaviour loss then sits at the constant-predictor value forever (runs
# 50273/50274/50426). The `cube_dropped` termination in the env attacks the same
# problem from the other side by resetting instead of preventing; this arm
# separates which of the two is doing the work.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth-Beta",
  env_cfg=flexiv_two_finger_distill_env_cfg(cam_type="depth", **V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    cam_type="depth", play=True, **V11_ALIGN_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(beta_decay_iters=500),
  runner_cls=ManipulationDistillationRunner,
)

# Finer first conv stride: 32x18 output cells instead of 16x9. The std=0.02 arm
# lifts to a 78.7 mm median but clears the 100 mm bar 3.1% of the time, which is
# a precision failure rather than a detection failure, and a 9-15 px cube
# collapses to about one cell of a 16x9 map.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth-Fine",
  env_cfg=flexiv_two_finger_distill_env_cfg(cam_type="depth", **V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    cam_type="depth", play=True, **V11_ALIGN_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(fine_cnn=True),
  runner_cls=ManipulationDistillationRunner,
)

# Control condition, not a deployable policy: the student is handed the
# teacher's own observation and no image, so it should fit almost perfectly. A
# flat loss curve on the camera students is ambiguous between "perception is
# hard here" and "the distillation plumbing is broken"; this separates them.
register_mjlab_task(
  task_id="Mjlab-Grasp-TwoFinger-Flexiv-Distill-Oracle",
  env_cfg=flexiv_two_finger_distill_env_cfg(cam_type="depth", **V11_ALIGN_KWARGS),
  play_env_cfg=flexiv_two_finger_distill_env_cfg(
    cam_type="depth", play=True, **V11_ALIGN_KWARGS
  ),
  rl_cfg=flexiv_two_finger_distill_runner_cfg(student_sees_state=True),
  runner_cls=ManipulationDistillationRunner,
)
