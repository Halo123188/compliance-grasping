"""Which checkpoint this is, and the handful of things that change with it.

`calib.py` holds what is true of the BENCH -- the robot, the camera, the foam,
the gripper's calibration -- and every one of those numbers is the same whatever
policy is flown. This file holds what is true of the POLICY, and those numbers
are not: two checkpoints trained a week apart read different observations,
command different action scales and are allowed different finger travel.

Keeping them here rather than in `calib.py` is the point. The alternative --
editing four constants by hand before each run -- is exactly the mistake nobody
notices, because every one of them fails silently:

  observation layout   a 35-d checkpoint fed a 34-d vector raises at load, which
                       is the benign case. A 34-d checkpoint fed a vector with
                       an extra number in the middle would not, if the widths
                       happened to line up -- so the layout is a property of the
                       profile and the width is asserted against the graph.
  action scale         0.35 vs 0.45 on the fingers is a 29% error in every
                       finger command, in the direction of gripping harder.
  finger travel        the sim STOPS the joint at its range; the hardware does
                       not. See `joint_limits`.
  slew limit           the one that is a crash rather than a regression, and the
                       one nothing can check. See `Profile.rate_limit`.
  depth frame          the frame size and the CNN's stride are one fact, not
                       two: a 30x40 spatial-softmax grid is a 160x120 frame at
                       stride 1/2/2 and a 320x240 one at 2/2/2, and the weights
                       record only the grid. See `Profile.cnn_stride`.

Selection is automatic and evidence-based: two of the four are readable off the
ONNX (the observation width off the graph, the action scale off the metadata),
and the run name pins the rest. `select()` refuses rather than guesses when the
evidence is short -- `deploy.run --profile` is the override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import calib

# ((prox_lo, prox_hi), (dist_lo, dist_hi)), given for the LEFT finger. The env's
# `_robot_cfg` mirrors them onto the right as [-hi, -lo], because closing is
# left-negative / right-positive and one symmetric range cannot express that.
FingerRange = tuple[tuple[float, float], tuple[float, float]]

# The action scale, in JOINT_NAMES order. Only the finger block ever changes;
# the arm's 0.40 and the wrist's 1.10 have been fixed since the claw was fitted.
_ARM_SCALE = (0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 1.10)

# The slew limit every checkpoint since the speed work trained under, rad/s, in
# JOINT_NAMES order. joint7 gets 1.5 because the wrist roll needs +-45 deg of
# travel before contact; the fingers get 3.0 because a grasp is a fast motion
# over a short distance.
_LIMITED = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 3.0, 3.0, 3.0, 3.0)

# The low-pass constant the `-SlowEma` evaluation priced, seconds. One value for
# every profile that carries it, because it is what was measured -- 0.09 s is
# the `ema_tau` of the `Ema` / `SlowEma` task ids and nothing else has been run.
_EMA = 0.09

# Finger travel as the training env set it (`_WIDE_DIRECTIONAL` and round 2's
# replacement). The proximal lower bound is the drive-through limit: q = -0.0754
# is 40 mm of pad separation, a 10 mm squeeze on the 50 mm cube. Round 2 opens it
# to -0.32 so a 15 mm object can be pinched -- 7.2 mm of nominal separation, and
# the fingers CROSS at -0.38.
_V11_FINGERS: FingerRange = ((-0.0754, 1.60), (-1.60, 0.10))
_R2_FINGERS: FingerRange = ((-0.32, 1.60), (-1.60, 0.10))

# Sizes below are given as the cube's EDGE in mm, because that is what a caliper
# reads and what `calib.CUBE_EDGE_MM` records; the observation is the HALF
# extent in metres.


@dataclass(frozen=True)
class Profile:
  """One checkpoint family: everything about the policy that is not the weights."""

  name: str
  # (term name, width) in the order the training env's `student` group lists
  # them. ORDER IS LOAD-BEARING and is not recoverable from the ONNX: the
  # exporter writes the TEACHER's term names (`observation_names` lists
  # ee_to_cube and cube_quat, which the student never sees), so a mis-ordered
  # vector of the right width is a valid input that simply behaves worse.
  obs_terms: tuple[tuple[str, int], ...]
  action_scale: tuple[float, ...]
  finger_range: FingerRange
  # rad/s per joint, or None for a checkpoint trained without the limiter.
  #
  # NOTHING VERIFIES THIS, and getting it wrong in the direction of "None on a
  # limited policy" is a full-speed command into the bench on the first step: a
  # policy trained with a limiter learns to lean on it, because past the cap only
  # the SIGN of the request reaches the sim and the magnitude stops receiving
  # gradient. The reverse is merely bad -- a policy trained without one lags its
  # own plan. `run.py`'s `[limiter] overdrive` summary is the closest thing to a
  # check: a policy trained against this cap asks for a little more than it, one
  # trained against a looser cap asks for many times it on nearly every step.
  rate_limit: tuple[float, ...] | None
  # First-order low-pass time constant on the published command, seconds, or
  # None for no filtering.
  #
  # THE OPPOSITE OF `rate_limit` IN THE ONE WAY THAT MATTERS: it does not have
  # to match what the checkpoint trained under, and every value below was set
  # on a checkpoint that trained with no EMA at all. The action term lives
  # OUTSIDE the network -- the ONNX graph is obs+camera -> action and knows
  # nothing about either smoother -- so adding a lag to a finished policy is a
  # deployment setting, and the `-SlowEma` task ids exist to price exactly that
  # in sim: same weights, slower action term, 2048 episodes.
  #
  # It is here rather than in `calib.py` because the PRICE is per-checkpoint,
  # measured against each arm's own unsmoothed baseline (`r4/sweeps/sl2_*`):
  #
  #   square-varh (R3-High)   95.65% -> 96.78%   +1.13
  #   cube        (R3-Cube)   91.26% -> 90.58%   -0.68
  #   everyshape  (R4-Gen)    78.91% -> 78.27%   -0.64
  #   (R4-Box, no profile)    77.83% -> 83.84%   +6.01
  #
  # i.e. free, within the +-0.9 pt standard error of a 2048-episode eval on
  # three of the four and a clear gain on the fourth. That is the whole reason
  # to prefer it: the other way to slow the arm down, tightening the slew cap
  # to `_SLEW_TIGHT`, costs 2.2 to 10.4 points on the same four checkpoints.
  #
  # What it buys is the PEAK, not the steady state: at tau = 0.09 s and 50 Hz a
  # step of the command moves 20% of the way on the first control step, so a
  # 0.8 rad jump becomes 0.16 rad. It bounds nothing -- a policy that holds a
  # far target still gets there at full speed -- which is why it is not a
  # substitute for the slew limit, and why `SlowBoth` exists.
  ema_tau: float | None
  # The cube EDGE range the run trained over, mm. Always known; whether the
  # policy can SEE it is `observes_cube_size`. For a size-blind checkpoint this
  # is advisory -- it says which cubes the bench may hold -- and for a
  # size-aware one it is a hard bound on the operator's `--cube-mm`, because
  # this observation goes through the same baked normalizer that turned an
  # out-of-range goal_height into a |3641| action.
  cube_edge_mm: tuple[float, float]
  # Where the cube may be placed, base frame metres: ((x_lo, x_hi), (y_lo, y_hi)).
  # The training spawn box, i.e. where the policy has seen a cube; outside it the
  # depth frame is out of distribution in the one input nothing cross-checks.
  spawn_xy: tuple[tuple[float, float], tuple[float, float]]
  # The depth frame this checkpoint reads, (H, W). Every run so far is
  # 120x160 -- the D435's 848x480 centre-cropped to 640x480 for the trained
  # field of view, then box-averaged by 4 -- so this is a default rather than a
  # per-profile number. It is kept as a field because it is the one thing here
  # that is CHECKED: `perception.centre_crop_resize` takes it as its target and
  # `policy.StudentPolicy` asserts it against the graph's `camera` input, so a
  # mispaired profile raises at load instead of feeding a squashed image.
  depth_hw: tuple[int, int] = calib.DEPTH_HW
  # The vision CNN's per-layer stride. Deployment never runs the convolutions
  # itself -- it runs the exported graph -- but this is the fact that makes the
  # graph, so `scripts/export_student_onnx.py` reads it from here.
  #
  # It belongs in this file because it is NOT recoverable from a bare
  # checkpoint, and the way it fails is invisible: the spatial-softmax grid is a
  # buffer in the state dict, and a 30x40 grid is (stride 1/2/2, 120x160) or
  # (stride 2/2/2, 240x320) equally well. Both rebuild, both load with
  # `strict=True`, both export -- and the second is a different function of the
  # world than the one that was trained. The exporter cross-checks the pair
  # against the grid, which is only a check because both halves are declared.
  cnn_stride: tuple[int, ...] = (2, 2, 2)
  # Run directory names this profile IS, matched against the ONNX filename.
  runs: tuple[str, ...] = ()
  # Whether an unrecognised file with this profile's shape may fall back to it.
  # At most one profile per (frame layout, action scale) may set this.
  fallback: bool = False
  note: str = ""

  @property
  def term_names(self) -> tuple[str, ...]:
    return tuple(name for name, _ in self.obs_terms)

  @property
  def term_widths(self) -> tuple[int, ...]:
    return tuple(width for _, width in self.obs_terms)

  @property
  def frame_dim(self) -> int:
    """One control step of observation. The graph's `obs` is a multiple of it."""
    return sum(self.term_widths)

  @property
  def term_slices(self) -> tuple[slice, ...]:
    """Where each term lives inside one frame."""
    edges = np.cumsum((0, *self.term_widths))
    return tuple(
      slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)
    )

  @property
  def observes_cube_size(self) -> bool:
    return "cube_size" in self.term_names

  @property
  def cube_half_extent_range(self) -> tuple[float, float]:
    """The trained `cube_size` range, in the observation's own units (m)."""
    lo, hi = self.cube_edge_mm
    return (lo / 2000.0, hi / 2000.0)

  @property
  def joint_limits(self) -> tuple[tuple[float, float], ...]:
    """The last thing between a blown-up action and a hard stop, per joint.

    THE FINGERS ARE NOT THE URDF'S +-1.6, and using that was a real gap. In sim
    the joint range is a physical stop -- the servo saturates against it and the
    policy is trained inside it -- while on hardware the same command is simply
    published, and the motor drives until its current cap holds it. So the
    deployment clamp has to be the range the checkpoint trained under, which is
    per-profile because round 2 moved it: a full-close action that bottoms out
    at 40 mm of pad separation under `_V11_FINGERS` bottoms out at 7 mm under
    `_R2_FINGERS`.

    The arm half comes from `calib.ARM_JOINT_LIMITS`, which is the robot's, not
    the policy's, and does not vary.
    """
    prox, dist = self.finger_range
    return (
      *calib.ARM_JOINT_LIMITS,
      prox,  # left_1
      dist,  # left_2
      (-prox[1], -prox[0]),  # right_1, mirrored: closing is right-positive
      (-dist[1], -dist[0]),  # right_2
    )

  def describe(self) -> str:
    """The one-screen summary `run.py` prints before anything moves."""
    prox, dist = self.finger_range
    lo, hi = self.cube_edge_mm
    size = "observed" if self.observes_cube_size else "size-blind"
    rate = (
      "OFF"
      if self.rate_limit is None
      else f"arm {min(self.rate_limit[calib.ARM_SLICE]):.1f}"
      f"..{max(self.rate_limit[calib.ARM_SLICE]):.1f}, "
      f"fingers {max(self.rate_limit[calib.HAND_SLICE]):.1f} rad/s"
    )
    (x0, x1), (y0, y1) = self.spawn_xy
    out = (
      f"[profile] {self.name}: obs {self.frame_dim}-d "
      f"({'+'.join(f'{n}:{w}' for n, w in self.obs_terms)})",
      f"          action scale fingers {self.action_scale[calib.HAND_SLICE][0]:.2f}"
      f", finger travel prox {prox[0]:+.4f}..{prox[1]:+.2f} "
      f"dist {dist[0]:+.2f}..{dist[1]:+.2f} rad",
      f"          slew limit {rate}, command EMA "
      + (
        "OFF"
        if not self.ema_tau
        else f"{self.ema_tau:.3f} s "
        f"({100 * (1 - math.exp(-1 / (calib.CONTROL_HZ * self.ema_tau))):.0f}% "
        f"of a step on the first control step)"
      ),
      f"          cube {lo:.1f}-{hi:.1f} mm ({size}), spawn x {x0:.2f}..{x1:.2f} "
      f"y {y0:+.2f}..{y1:+.2f} m",
      f"          depth frame {self.depth_hw[0]}x{self.depth_hw[1]}, cnn stride "
      f"{'/'.join(str(s) for s in self.cnn_stride)}",
    )
    return "\n".join(out)


# The four terms every student has read since the wide claw was fitted.
_BASE_TERMS = (("joint_pos", 11), ("joint_vel", 11), ("actions", 11))
_V11_TERMS = (*_BASE_TERMS, ("goal_height", 1))
# Round 2 inserts the object's size BEFORE goal_height, because that is where
# `_apply_common` inserts the term into the group and mjlab concatenates terms
# in definition order. Confirmed against the checkpoints themselves: the baked
# normalizer's dim 33 has mean 0.0250 on R2-Reach and 0.0175 on R2-Small, which
# are the midpoints of those two arms' half-extent ranges, while dim 34 carries
# the 0.549 / 0.039 that is goal_height on every checkpoint ever exported. The
# same two numbers also settle the UNITS: 0.0250 is the half-edge of the 50 mm
# cube in metres, not its edge and not millimetres.
#
# The swap is the mistake worth naming, since both terms are one number wide and
# either order is a valid 35-vector. Measured on the synthetic bench frame at the
# home pose, feeding goal_height where cube_size belongs: max|a| goes from 0.39
# to 356 on R2-Reach and from 0.21 to 42.5 on R2-Small, so `--max-action` does
# catch it -- but only after the wrong vector has been assembled and flown.
_R2_TERMS = (*_BASE_TERMS, ("cube_size", 1), ("goal_height", 1))

# The spawn box, base frame. Round 1 widened it from +-80/+-100 mm about the
# manipulation centre to +-100/+-120, gated by `scripts/wide_spawn_gate.py`
# against the camera's frame coverage rather than against the table.
_V11_SPAWN = ((0.38, 0.54), (-0.10, 0.10))
_WIDE_SPAWN = ((0.36, 0.56), (-0.12, 0.12))

# Round 3's vision CNN. Same three layers, same channels and kernels; the first
# one stops halving, which doubles the softmax grid off an unchanged camera.
_FINE_STRIDE = (1, 2, 2)

PROFILES: dict[str, Profile] = {
  # --- before round 1: the 34-d students, on the narrow bench ----------------
  "v11": Profile(
    name="v11",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.35, 0.35, 0.35, 0.35),
    finger_range=_V11_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=None,
    cube_edge_mm=(45.0, 55.0),
    spawn_xy=_V11_SPAWN,
    runs=(
      "2026-08-11_06-12-17_s_facelevel",
      "2026-08-10_20-21-25_s_g1_relabel",
      "2026-08-11_00-52-56_s_g1_hist",
    ),
    fallback=True,
    note=(
      "`s_g1_hist` reads four control steps of this same frame; the depth is "
      "read off the graph, so nothing here changes for it."
    ),
  ),
  "v11-unlimited": Profile(
    name="v11-unlimited",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.35, 0.35, 0.35, 0.35),
    finger_range=_V11_FINGERS,
    rate_limit=None,  # trained before the action term had a limiter at all
    ema_tau=None,
    cube_edge_mm=(45.0, 55.0),
    spawn_xy=_V11_SPAWN,
    runs=(
      "2026-08-06_20-53-44_s3_dr_fine",
      "2026-08-06_17-29-41_s1_dr",
    ),
    note=(
      "the checkpoints from before the action term had a limiter at all -- their "
      "env.yaml carries no `rate_limit` key, not a null one."
    ),
  ),
  # --- round 1: COACD collision, wider spawn / size / mass -------------------
  # Deployment-identical to `v11` -- same observation, same scales, same finger
  # travel, same limiter -- so the profile exists for the BENCH numbers: these
  # two trained on cubes from 42.5 to 57.5 mm placed over a box 40 mm wider and
  # 40 mm deeper than their predecessors', and on masses from 30 to 300 g.
  "r1": Profile(
    name="r1",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.35, 0.35, 0.35, 0.35),
    finger_range=_V11_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=None,
    cube_edge_mm=(42.5, 57.5),
    spawn_xy=_WIDE_SPAWN,
    runs=(
      "2026-08-12_18-20-31_d-R1-FaceLevel-All",
      "2026-08-12_18-20-34_d-R1-SatPen-All",
    ),
    note=(
      "cube mass 30-300 g. The collision representation moved to coacd_t0.2, "
      "which changes the grip force in SIM (12.4 N on boxes, 7.0 N on hulls) "
      "and nothing on hardware."
    ),
  ),
  # --- round 2: small objects -----------------------------------------------
  # Four coupled changes, three of which are visible from here. The finger's
  # lower bound opens to -0.32 so a 15 mm object can be pinched, the finger
  # action scale rises to 0.45 so that closure is -1.32 sigma rather than -1.70,
  # and the policy is TOLD the object's size -- which is required rather than
  # nice to have, because the same full-close command that stops at a 40 mm gap
  # under the old bound becomes ~0.24 rad of over-travel on a 50 mm cube under
  # the new one (order 350 N). Round 3 went on to open the travel WITHOUT the
  # size term, so "required" turned out to be "required here": those arms are
  # size-blind on the same -0.32 bound, and lean on the slew limit and the
  # object's appearance in depth instead.
  "r2-reach": Profile(
    name="r2-reach",
    obs_terms=_R2_TERMS,
    action_scale=(*_ARM_SCALE, 0.45, 0.45, 0.45, 0.45),
    finger_range=_R2_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=None,
    cube_edge_mm=(45.0, 55.0),  # the reach change ALONE, on the current sizes
    spawn_xy=_WIDE_SPAWN,
    runs=("2026-08-12_18-20-48_d-R2-Reach",),
    fallback=True,
    note=(
      "the control arm: round 2's reach and scale on round 1's cube sizes, so a "
      "regression here is the reach change rather than the small objects."
    ),
  ),
  "r2-small": Profile(
    name="r2-small",
    obs_terms=_R2_TERMS,
    action_scale=(*_ARM_SCALE, 0.45, 0.45, 0.45, 0.45),
    finger_range=_R2_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=None,
    cube_edge_mm=(15.0, 57.5),
    spawn_xy=_WIDE_SPAWN,
    runs=("2026-08-12_18-20-19_d-R2-Small",),
    note=(
      "a 15 mm cube is 3-6 px at 160x120, so a failure here can be the object "
      "not being VISIBLE rather than the grasp being wrong -- which is what "
      "r2-reach is for."
    ),
  ),
  # --- round 3 (August 18): new object shapes, and a finer first conv -------
  # These four arrived as bare `model_N.pt` files, and what follows was read out
  # of their env.yaml afterwards rather than inferred from the weights. Worth
  # recording, because the inference had two of the four fields wrong and the
  # next bare checkpoint will invite the same two mistakes:
  #
  #   THE ACTION TERM IS ROUND 2's, ON A v11 OBSERVATION. Fingers at 0.45 and
  #   travel out to -0.32, but no `cube_size` term. So "0.45 implies 35-d" was a
  #   coincidence of the first two rounds rather than a rule, and a checkpoint
  #   cannot be dated from its observation layout. Inferring the v11 action term
  #   from the v11 frame would have been 22% too little finger command, and --
  #   worse -- would have clamped the fingers at -0.0754, where a 15 mm object
  #   can never be reached at all and nothing anywhere reports a fault.
  #
  #   THE DEPTH FRAME IS 120x160, the same as every run before them. The 30x40
  #   softmax grid on `cube` and `everyshape` is a STRIDE of 1/2/2, not a bigger
  #   picture: the first convolution runs at full resolution and only the last
  #   two halve. Same camera, same crop, same field of view, twice the spatial
  #   resolution carried into the softmax for ~4x the first layer's arithmetic.
  #
  # Shared by all four: the spawn box, the action term, and the object HEIGHT
  # range -- 15 to 57.5 mm of full edge, from a 25 mm nominal half-extent times
  # U(0.30, 1.15), with every half-extent clamped to 7.5..32.5 mm afterwards.
  # They differ in the object shape, in the stride, and in one reset pose.
  "square-fixh": Profile(
    name="square-fixh",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.45, 0.45, 0.45, 0.45),
    finger_range=_R2_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=None,
    cube_edge_mm=(15.0, 57.5),
    spawn_xy=_WIDE_SPAWN,
    runs=("square_fixH",),
    # No `ema_tau`: the `-SlowEma` sweep was run on the other three arms and on
    # R4-Box, not on this one, and the price is per-checkpoint. Add `_EMA` here
    # once it has a number of its own.
    note=(
      "round 3's control arm: upright cubes, no stretch, from a reset pose a "
      "fixed 68 mm above the object. The fixH/variableH in the two filenames is "
      "that RESET POSE, not `goal_height` -- the two runs' baked normalizers "
      "agree on goal_height to four decimals, so the same --goal-height applies "
      "to both."
    ),
  ),
  "square-varh": Profile(
    name="square-varh",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.45, 0.45, 0.45, 0.45),
    finger_range=_R2_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=_EMA,  # priced on R3-High; see the field comment
    cube_edge_mm=(15.0, 57.5),
    spawn_xy=_WIDE_SPAWN,
    runs=("square_variableH",),
    note=(
      "`square-fixh`'s objects from a reset pose drawn out of a solved hover "
      "family at +68 / +110 / +150 / +200 mm above the object. The claw leaves "
      "the depth frame above +143 mm, so roughly 46% of its training episodes "
      "begin with the claw not visible to itself -- which is the one thing this "
      "arm buys: it tolerates any start height in that band, where the other "
      "three expect +68 mm. Only the reset pose moves. The action term's "
      "default offset stays where it is, because lifting that too puts the "
      "descent at 1.51 sigma."
    ),
  ),
  "cube": Profile(
    name="cube",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.45, 0.45, 0.45, 0.45),
    finger_range=_R2_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=_EMA,  # priced on R3-Cube; see the field comment
    cube_edge_mm=(15.0, 57.5),
    spawn_xy=_WIDE_SPAWN,
    cnn_stride=_FINE_STRIDE,
    runs=("cube_model",),
    note=(
      "boxes only, stretched in the horizontal plane: x and y each draw "
      "U(0.7, 1.6), normalised so their geometric mean is 1, with z fixed at 1. "
      "So the HEIGHT is the 15-57.5 mm above and the two footprint edges are "
      "that times about 0.66..1.51 (plan aspect ratio p95 ~1.76, bounded at "
      "2.29). Despite the run's name these are not cubes; --cube-mm is the "
      "height, and the policy cannot see it either way."
    ),
  ),
  "everyshape": Profile(
    name="everyshape",
    obs_terms=_V11_TERMS,
    action_scale=(*_ARM_SCALE, 0.45, 0.45, 0.45, 0.45),
    finger_range=_R2_FINGERS,
    rate_limit=_LIMITED,
    ema_tau=_EMA,  # priced on R4-Gen; see the field comment
    cube_edge_mm=(15.0, 57.5),
    spawn_xy=_WIDE_SPAWN,
    cnn_stride=_FINE_STRIDE,
    runs=("everyShape",),
    note=(
      "box, cylinder and sphere at 50/25/25, on `cube`'s stretch -- but a "
      "cylinder or a sphere takes it on the RADIUS only and stays circular in "
      "plan, so what separates this arm from `cube` is the shape mix rather "
      "than the aspect. The only one of the four that stacks a history: its "
      "136-d observation is four control steps of the same 34-d frame, "
      "term-major, exactly as `s_g1_hist` did. --cube-mm is advisory here in a "
      "way it is not elsewhere -- the bench object need not be a box at all."
    ),
  ),
}


def by_name(name: str) -> Profile:
  try:
    return PROFILES[name]
  except KeyError:
    raise ValueError(
      f"unknown profile {name!r}; have {', '.join(sorted(PROFILES))}"
    ) from None


def select(onnx_path: str | Path, action_scale: np.ndarray, obs_width: int) -> Profile:
  """Which checkpoint family this ONNX is, from the file and from the graph.

  Two independent handles, and they are used in that order:

    THE RUN NAME. The exporter writes `<run>.onnx` inside the run directory, so
    an unrenamed file names the arm it came from exactly. This is the only
    handle on the things the graph cannot carry -- the slew limit, the finger
    travel, the trained size range -- so it is tried first.

    THE SHAPE. Failing that, the observation width and the finger action scale
    are read off the graph and the metadata. They separate less than they once
    did -- (34, 0.35) is everything before round 2 and (35, 0.45) is round 2,
    but round 3 is (34, 0.45) across all four of its arms, which is why none of
    them is a fallback and an unrecognised file of that shape is refused rather
    than assigned. Where a choice remains, the profile marked `fallback` wins,
    and it is
    marked on the SAFER member: `v11` rather than `v11-unlimited` (limiting an
    unlimited policy makes it lag; the reverse is a crash), and `r2-reach`
    rather than `r2-small` (the narrower size range, so an unusual cube has to
    be asked for rather than assumed).

  Raising here is a feature. The failure this replaces is a deployment that
  loads, runs, and is 29% wrong on every finger command.
  """
  stem = Path(onnx_path).stem
  named = [p for p in PROFILES.values() if any(run in stem for run in p.runs)]
  if len(named) == 1:
    return named[0]
  if len(named) > 1:  # two profiles claiming one run is a bug in this file
    raise ValueError(f"{stem} matches profiles {[p.name for p in named]}")

  scale = np.asarray(action_scale, dtype=np.float32)
  shaped = [
    p
    for p in PROFILES.values()
    if obs_width % p.frame_dim == 0
    and np.allclose(scale, np.asarray(p.action_scale, np.float32), atol=1e-3)
  ]
  if not shaped:
    # A new arm is a five-line entry above; guessing is what this refuses.
    raise ValueError(
      f"no profile matches obs {obs_width}-d with finger action scale "
      f"{scale[calib.HAND_SLICE][0]:.3f}. This is a checkpoint deploy/ has not "
      "been told about: add a Profile for it rather than flying the nearest one."
    )
  preferred = [p for p in shaped if p.fallback]
  if len(preferred) != 1:
    raise ValueError(
      f"{stem} is not a known run name and its shape (obs {obs_width}-d, finger "
      f"scale {scale[calib.HAND_SLICE][0]:.2f}) fits "
      f"{[p.name for p in shaped]}. Pass --profile to say which."
    )
  return preferred[0]
