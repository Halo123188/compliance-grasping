"""The deployed command smoother, against the training env's own action term.

`deploy/policy.py` reimplements `SmoothedJointPositionAction` in numpy, because
the robot host runs neither mjlab nor torch. The two have to agree: a policy
trained against a slew-limited command and deployed without one publishes the
raw request, which on this task is a full-speed move into the bench.

The parity that matters is the RATE, not the per-step delta -- sim integrates
the limiter once per physics substep at 200 Hz and the robot host once per
control step at 50 Hz, so the two never take the same steps and only the rad/s
bound is shared.

Also covers `deploy/profiles.py`, which decides WHICH of those settings apply.
Everything the profile carries -- the observation layout, the action scale, the
finger travel, the limiter -- fails silently when it is wrong, so the tests are
about selecting the right one and refusing to guess.

The ONNX session is stubbed rather than the policy subclassed, so what runs here
is the real `act()` down to the joint-limit clamp it ends with.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from deploy import calib, profiles
from deploy.policy import StudentPolicy

DT = 1.0 / calib.CONTROL_HZ
HOME = np.asarray(calib.DEFAULT_JOINT_POS, dtype=np.float32)
# The family every deployed checkpoint before round 2 belongs to; the round-2
# arms get their own tests below rather than being parametrised over, because
# what changed between them is exactly what these tests are about.
V11 = profiles.PROFILES["v11"]
R2 = profiles.PROFILES["r2-small"]
# The one arm trained without a slew cap, so the EMA can be watched on its own,
# and one that carries an EMA out of profiles.py.
UNLIMITED = profiles.PROFILES["v11-unlimited"]
EMA = profiles.PROFILES["square-varh"]
SCALE = np.asarray(V11.action_scale, dtype=np.float32)
FRAME = V11.frame_dim


class _StubInput:
  def __init__(self, name: str, shape: list):
    self.name, self.shape = name, shape


class _StubSession:
  """Enough of onnxruntime.InferenceSession to load and to return one action."""

  def __init__(
    self,
    action: np.ndarray,
    history: int = 1,
    width: int | None = None,
    profile: profiles.Profile = V11,
    camera_hw: tuple[int, int] | None = None,
  ):
    self.action = np.asarray(action, dtype=np.float32).reshape(1, 11)
    self.history = history
    self.profile = profile
    self.width = history * profile.frame_dim if width is None else width
    # The depth frame is a property of the GRAPH -- it is the CNN's own input
    # size -- so the stub follows the profile unless a test is about the two
    # disagreeing.
    self.camera_hw = profile.depth_hw if camera_hw is None else camera_hw
    self.last_obs: np.ndarray | None = None

  @staticmethod
  def _csv(values) -> str:
    # The exporter's `list_to_csv_str`, three decimals and all -- which is why
    # calib.py keeps the full-precision copies and policy.py compares at 1e-3.
    return ",".join(f"{v:.3f}" for v in values)

  def get_modelmeta(self):
    names = ",".join(
      n + "_tf" if n.startswith(("left", "right")) else n for n in calib.JOINT_NAMES
    )
    meta = {
      "joint_names": names,
      "default_joint_pos": self._csv(calib.DEFAULT_JOINT_POS),
      "action_scale": self._csv(self.profile.action_scale),
      "joint_stiffness": self._csv(calib.JOINT_STIFFNESS),
      "joint_damping": self._csv(calib.JOINT_DAMPING),
    }
    return type("Meta", (), {"custom_metadata_map": meta})()

  def get_inputs(self):
    return [
      _StubInput("obs", [1, self.width]),
      _StubInput("camera", [1, 1, *self.camera_hw]),
    ]

  def run(self, names, feeds):  # noqa: ARG002 -- the graph is what is stubbed
    self.last_obs = feeds["obs"]
    return [self.action]


@pytest.fixture
def make_policy(monkeypatch):
  def _make(
    action,
    history: int = 1,
    width: int | None = None,
    profile: profiles.Profile = V11,
    name: str | None = None,
    camera_hw: tuple[int, int] | None = None,
    **kwargs,
  ) -> StudentPolicy:
    session = _StubSession(action, history, width, profile, camera_hw)
    monkeypatch.setattr(
      "deploy.policy.ort.InferenceSession", lambda *a, **k: session, raising=True
    )
    # Named after one of the profile's own runs by default, so selection is
    # exercised end to end rather than handed the answer.
    return StudentPolicy(f"{name or profile.runs[0]}.onnx", **kwargs)

  return _make


@pytest.fixture
def limit() -> np.ndarray:
  assert V11.rate_limit is not None
  return np.asarray(V11.rate_limit, dtype=np.float32)


def _drive(policy: StudentPolicy, steps: int) -> np.ndarray:
  """Run `steps` inferences and return the last published command."""
  obs = np.zeros(policy.profile.frame_dim, np.float32)
  depth = np.zeros((1, *policy.depth_hw), np.float32)
  cmd = policy._cmd.copy()
  for _ in range(steps):
    cmd = policy.act(obs, depth)
  return cmd


def test_the_command_never_moves_faster_than_the_cap(make_policy, limit):
  """The guarantee the whole term exists for, under a saturating action."""
  policy = make_policy(np.full(11, 8.0))  # ~2x the reach of a trained action
  policy.reset(HOME)
  obs = np.zeros(FRAME, np.float32)
  depth = np.zeros((1, *policy.depth_hw), np.float32)
  prev = HOME.copy()
  for _ in range(50):
    cmd = policy.act(obs, depth)
    speed = np.abs(cmd - prev) / DT
    assert (speed <= limit + 1e-4).all(), f"{speed} exceeds {limit}"
    prev = cmd.copy()


def test_a_reachable_target_is_published_unchanged(make_policy, limit):
  """The limiter must be invisible when the policy stays inside it."""
  action = 0.1 * limit * DT / SCALE  # a tenth of one allowance per joint
  policy = make_policy(action)
  policy.reset(HOME)
  cmd = _drive(policy, 20)
  np.testing.assert_allclose(cmd, HOME + SCALE * action, atol=1e-6)
  assert (policy.last_overdrive == 0).all()


def test_it_starts_from_the_measured_pose_not_from_home(make_policy, limit):
  """Seeding from home would ramp across the home tolerance every trial."""
  measured = HOME + 0.05
  policy = make_policy(np.zeros(11))
  policy.reset(measured)
  cmd = _drive(policy, 1)
  # Action 0 asks for exactly the home pose, which from `measured` is 0.05 rad
  # away -- more than one allowance on the arm, so the command steps toward it
  # rather than jumping there.
  moved = np.abs(cmd - measured)[calib.ARM_SLICE]
  assert (moved <= limit[calib.ARM_SLICE] * DT + 1e-6).all()
  assert (moved > 0).all()


def test_overdrive_reports_how_far_past_the_cap_the_request_was(make_policy, limit):
  """The deployment's only handle on 'is this the right RATE_LIMIT'."""
  policy = make_policy(3.0 * limit * DT / SCALE)  # ask for exactly 3 allowances
  policy.reset(HOME)
  _drive(policy, 1)
  np.testing.assert_allclose(policy.last_overdrive, 2.0, atol=1e-3)


def test_the_raw_action_is_what_is_fed_back_not_the_command(make_policy, limit):
  """obs[22:33] is the request. Feeding back the published command is wrong."""
  action = np.full(11, 4.0, dtype=np.float32)
  policy = make_policy(action)
  policy.reset(HOME)
  _drive(policy, 3)
  np.testing.assert_allclose(policy.last_action, action)
  obs = policy.observe(HOME, np.zeros(11))
  np.testing.assert_allclose(obs[22:33], action)


def test_the_joint_limit_clamp_is_the_last_thing_applied(make_policy):
  """Whatever the smoother does, it cannot publish past a hard stop."""
  limits = np.asarray(V11.joint_limits, dtype=np.float32)
  policy = make_policy(np.full(11, 1e4))
  policy.reset(HOME)
  cmd = _drive(policy, 2000)  # long enough for the limiter to ramp into a stop
  assert (cmd <= limits[:, 1] + 1e-6).all()
  assert (cmd >= limits[:, 0] - 1e-6).all()


def test_ema_is_off_unless_the_profile_says_otherwise(make_policy):
  """v11 carries no EMA, so a stray alpha here is a bug in profiles.py."""
  policy = make_policy(np.zeros(11))
  assert (V11.ema_tau is None) == (policy.ema_alpha == 0.0)


def test_the_ema_shaves_the_first_step_of_a_command_jump(make_policy):
  """0.09 s at 50 Hz is 20% of the way on the first control step.

  This is the whole claim the -SlowEma evaluation rests on -- the lag cuts the
  PEAK of a step, which is what makes the fingers snap, while bounding nothing
  in steady state. The slew limit is turned off so the EMA is the only thing
  acting; with the cap on, a full-scale request is limited long before it is
  filtered.
  """
  policy = make_policy(np.ones(11), profile=UNLIMITED)
  policy.set_ema_tau(0.09)
  policy.reset()

  obs = np.zeros(policy.profile.frame_dim, np.float32)
  depth = np.zeros((1, *policy.depth_hw), np.float32)
  start = policy._cmd.copy()
  first = policy.act(obs, depth)

  jump = policy.last_request - start
  moved = (first - start) / jump
  assert np.allclose(moved, 1.0 - math.exp(-1.0 / (calib.CONTROL_HZ * 0.09)), atol=1e-6)
  assert 0.19 < float(moved[0]) < 0.21

  # And it converges on the request rather than capping it: nothing is bounded.
  # Against the CLAMPED request, since the joint-limit clip is downstream of
  # both smoothers and is the one bound the EMA does not remove.
  settled = np.clip(policy.last_request, policy.lo, policy.hi)
  assert np.allclose(_drive(policy, 200), settled, atol=1e-4)


def test_the_ema_can_be_overridden_at_the_bench(make_policy):
  """`--ema-tau` is legitimate in a way a slew-limit override is not.

  The lag is not something the weights trained against -- the action term is
  outside the ONNX graph -- so the operator may dial it. 0 turns it off, and a
  negative value is a typo rather than a setting.
  """
  policy = make_policy(np.zeros(11), profile=EMA)
  assert policy.ema_alpha == pytest.approx(math.exp(-1.0 / (calib.CONTROL_HZ * 0.09)))
  assert "command EMA" in policy.describe()

  policy.set_ema_tau(0.0)
  assert policy.ema_alpha == 0.0
  assert "OVERRIDDEN" in policy.describe()

  with pytest.raises(ValueError, match="non-negative"):
    policy.set_ema_tau(-0.01)


# --- the history-stacked student ---------------------------------------------
# 2026-08-11_00-52-56_s_g1_hist reads four control steps at once. Its 136 numbers
# are the same four terms as the 34-d students, but each term carries its own
# history and the terms are laid out one after another -- NOT four whole
# observations end to end. Both readings are 136 wide, so nothing raises; the
# wrong one just deploys a policy that behaves worse than it scored.

HISTORY = 4


def test_the_history_depth_comes_from_the_graph(make_policy):
  """34 or 136 is the checkpoint's to state, not the operator's to configure."""
  assert make_policy(np.zeros(11)).history_length == 1
  assert make_policy(np.zeros(11), history=HISTORY).history_length == HISTORY


def test_a_width_that_is_not_whole_frames_is_refused(make_policy):
  """100 is not a history of anything, and must not be rounded to one."""
  with pytest.raises(ValueError, match="whole number"):
    make_policy(np.zeros(11), width=100)


def _frames(policy: StudentPolicy) -> np.ndarray:
  """The (H, 34) view of the last obs the stub was fed, undoing the stacking."""
  session = cast(_StubSession, policy.session)
  assert session.last_obs is not None, "nothing has been fed to the graph yet"
  obs = session.last_obs.reshape(-1)
  h, out = policy.history_length, []
  offset = 0
  for width in policy.profile.term_widths:
    out.append(obs[offset : offset + width * h].reshape(h, width))
    offset += width * h
  return np.concatenate(out, axis=1)


def test_the_terms_are_laid_out_term_major_not_frame_major(make_policy):
  """The whole point. Each term's history is contiguous; the terms follow."""
  policy = make_policy(np.zeros(11), history=HISTORY)
  policy.reset(HOME)
  obs = policy.observe(HOME + 0.1, np.full(11, 2.0), goal_height=0.55)

  assert obs.shape == (HISTORY * FRAME,)
  # goal_height is one number per frame, so under term-major it is the LAST
  # four dimensions and under frame-major it would be dims 33/67/101/135.
  np.testing.assert_allclose(obs[-HISTORY:], 0.55)
  np.testing.assert_allclose(obs[:44].reshape(HISTORY, 11), 0.1, atol=1e-6)
  np.testing.assert_allclose(obs[44:88].reshape(HISTORY, 11), 2.0)


def test_the_history_is_backfilled_at_reset_not_zero_padded(make_policy):
  """Sim's first post-reset push fills every slot; three zero frames are a pose
  the policy never trained on, at the moment the arm is moving fastest."""
  policy = make_policy(np.zeros(11), history=HISTORY)
  policy.reset(HOME)
  obs = policy.observe(HOME + 0.3, np.full(11, 1.0))
  frames = obs[:44].reshape(HISTORY, 11)
  assert (frames == frames[0]).all(), "the first observation must be one pose x4"
  assert (frames != 0).all()


def test_the_newest_frame_is_last_and_one_step_is_one_push(make_policy):
  """Chronological, oldest first -- what `CircularBuffer.buffer` returns."""
  policy = make_policy(np.zeros(11), history=HISTORY)
  policy.reset(HOME)
  depth = np.zeros((1, *policy.depth_hw), np.float32)
  for q in (0.1, 0.2, 0.3, 0.4, 0.5):
    policy.act(policy.observe(HOME + q, np.zeros(11)), depth)

  jp = _frames(policy)[:, :11]
  # Five steps into a four-deep history: the first is gone, 0.2 is the oldest
  # kept and 0.5 is the newest. A reversed stack would put 0.5 first.
  np.testing.assert_allclose(jp[:, 0], [0.2, 0.3, 0.4, 0.5], atol=1e-6)


def test_reset_drops_the_previous_trial_rather_than_carrying_it(make_policy):
  """`reset()` between trials, or the new episode starts inside the old one."""
  policy = make_policy(np.zeros(11), history=HISTORY)
  policy.reset(HOME)
  policy.observe(HOME + 0.9, np.zeros(11))
  policy.reset(HOME)
  obs = policy.observe(HOME + 0.1, np.zeros(11))
  np.testing.assert_allclose(obs[:44].reshape(HISTORY, 11), 0.1, atol=1e-6)


def test_the_plain_students_are_untouched_by_any_of_this(make_policy):
  """H = 1 must still produce exactly the 34-d vector it always did."""
  policy = make_policy(np.zeros(11))
  policy.reset(HOME)
  obs = policy.observe(HOME + 0.1, np.full(11, 2.0), goal_height=0.55)
  expected = np.concatenate([np.full(11, 0.1), np.full(11, 2.0), np.zeros(11), [0.55]])
  np.testing.assert_allclose(obs, expected, atol=1e-6)


# --- round 2: the size-aware students -----------------------------------------
# `d-R2-Reach` and `d-R2-Small` read the object's half-edge between the actions
# and goal_height, command a 0.45 finger scale instead of 0.35, and are allowed
# down to -0.32 rad of proximal travel instead of -0.0754. All three are silent
# when wrong: the observation is a valid vector, the scale is a 29% error in the
# grip direction, and the travel is a clamp that simply never fires.

CUBE = 0.025  # half-edge, metres: the 50 mm bench cube


def test_the_cube_size_goes_between_the_actions_and_the_goal(make_policy):
  """The env inserts the term there, and nothing downstream can tell."""
  policy = make_policy(np.zeros(11), profile=R2, cube_half_extent=CUBE)
  policy.reset(HOME)
  obs = policy.observe(HOME + 0.1, np.full(11, 2.0), goal_height=0.55)

  assert obs.shape == (35,)
  np.testing.assert_allclose(obs[33], CUBE)
  np.testing.assert_allclose(obs[34], 0.55)


def test_the_cube_size_keeps_its_own_history_slot(make_policy):
  """Term-major applies to it like any other term, not to the frame."""
  policy = make_policy(np.zeros(11), history=4, profile=R2, cube_half_extent=CUBE)
  policy.reset(HOME)
  obs = policy.observe(HOME, np.zeros(11), goal_height=0.55)

  assert obs.shape == (4 * 35,)
  np.testing.assert_allclose(obs[132:136], CUBE)  # cube_size x4
  np.testing.assert_allclose(obs[136:140], 0.55)  # then goal_height x4


def test_a_size_aware_checkpoint_refuses_to_load_without_a_size(make_policy):
  """There is no default for an observation; a wrong one is out of distribution."""
  with pytest.raises(ValueError, match="observes the object's size"):
    make_policy(np.zeros(11), profile=R2)


def test_a_cube_outside_the_trained_range_is_refused(make_policy):
  """The mm/m and edge/half-edge mistakes both land tens of sigma out."""
  with pytest.raises(ValueError, match="outside r2-small's trained range"):
    make_policy(np.zeros(11), profile=R2, cube_half_extent=0.050)  # edge, not half


def test_the_finger_clamp_is_the_checkpoint_s_travel_not_the_urdf_s(make_policy):
  """Sim STOPS the joint at its range; hardware publishes the command anyway."""
  v11 = np.asarray(V11.joint_limits, np.float32)
  r2 = np.asarray(R2.joint_limits, np.float32)
  # left_1 closes negative, right_1 positive: the range mirrors, it is not
  # symmetric, and both are far inside the URDF's +-1.6.
  np.testing.assert_allclose(v11[7], (-0.0754, 1.60))
  np.testing.assert_allclose(v11[9], (-1.60, 0.0754))
  np.testing.assert_allclose(r2[7], (-0.32, 1.60))
  np.testing.assert_allclose(r2[9], (-1.60, 0.32))

  policy = make_policy(np.full(11, -1e4), profile=R2, cube_half_extent=CUBE)
  policy.reset(HOME)
  cmd = _drive(policy, 2000)
  assert cmd[7] == pytest.approx(-0.32)
  assert cmd[9] == pytest.approx(-1.60)


def test_round_two_commands_the_wider_finger_scale(make_policy):
  """0.45, not 0.35. Read off the metadata and asserted against the profile."""
  policy = make_policy(np.zeros(11), profile=R2, cube_half_extent=CUBE)
  np.testing.assert_allclose(policy.scale[calib.HAND_SLICE], 0.45)


# --- picking the profile ------------------------------------------------------


def test_an_unrenamed_export_selects_its_own_profile():
  """The exporter writes `<run>.onnx`, so the file names the arm it came from."""
  for name, profile in profiles.PROFILES.items():
    for run in profile.runs:
      chosen = profiles.select(
        f"/logs/{run}/{run}.onnx",
        np.asarray(profile.action_scale, np.float32),
        profile.frame_dim,
      )
      assert chosen.name == name


def test_a_renamed_export_falls_back_on_shape_but_only_where_it_is_safe():
  """34-d at 0.35 could be limited or unlimited; the SAFE one wins."""
  chosen = profiles.select(
    "policy.onnx", np.asarray(V11.action_scale, np.float32), V11.frame_dim
  )
  # Limiting an unlimited policy makes it lag; publishing an unlimited version
  # of a limited one is a full-speed command into the bench.
  assert chosen.rate_limit is not None
  assert chosen.name == "v11"


def test_a_shape_no_profile_claims_is_refused_rather_than_approximated():
  """A new arm is an entry in profiles.py, not the nearest existing one."""
  with pytest.raises(ValueError, match="no profile matches"):
    profiles.select("policy.onnx", np.full(11, 0.7, np.float32), 34)


def test_the_metadata_still_has_to_agree_with_the_chosen_profile(make_policy):
  """An r2-named export whose weights carry 0.35 is a mislabelled file.

  The name picks the profile and the metadata then has to corroborate it, so a
  run directory renamed onto the wrong checkpoint raises instead of commanding
  the fingers 29% harder than they were trained to be.
  """
  with pytest.raises(ValueError, match="action_scale mismatch"):
    make_policy(
      np.zeros(11),
      profile=V11,  # what the exported weights say
      name=R2.runs[0],  # what the file claims to be
      width=R2.frame_dim,
      cube_half_extent=CUBE,
    )


# --- round 3: a round-2 action term on a v11 observation ----------------------

CUBE_R3 = profiles.PROFILES["cube"]
SHAPES = profiles.PROFILES["everyshape"]
ROUND_3 = ("square-fixh", "square-varh", "cube", "everyshape")


def test_the_round_3_arms_pair_round_2_travel_with_a_v11_frame():
  """The trap these four set, and the reason a profile is not a date.

  Rounds 1 and 2 made it look like the action term could be read off the
  observation: 34-d meant fingers at 0.35 stopping at -0.0754, 35-d meant 0.45
  stopping at -0.32. Round 3 breaks the pairing -- round 2's fingers on a v11
  frame, with no `cube_size` term at all -- so a profile inferred from the
  layout would clamp these at -0.0754, where the closure a 15 mm object needs
  cannot be commanded and nothing reports a fault.
  """
  for name in ROUND_3:
    profile = profiles.PROFILES[name]
    assert profile.frame_dim == V11.frame_dim
    assert not profile.observes_cube_size
    assert profile.action_scale[calib.HAND_SLICE] == (0.45, 0.45, 0.45, 0.45)
    assert profile.finger_range[0][0] == pytest.approx(-0.32)


def test_an_unnamed_round_3_export_is_refused_rather_than_flown():
  """(34-d, 0.45) is four profiles and no fallback, so the shape cannot decide.

  A renamed export of one of these must not land on v11 -- the scale differs --
  nor on r2, whose frame is a term wider. There is nothing left to guess with,
  which is the intended outcome: `--profile` is how the operator says which.
  """
  with pytest.raises(ValueError, match="Pass --profile"):
    profiles.select(
      "policy.onnx", np.asarray(CUBE_R3.action_scale, np.float32), CUBE_R3.frame_dim
    )
  assert not any(profiles.PROFILES[n].fallback for n in ROUND_3)


def test_the_shape_arm_stacks_four_frames(make_policy):
  """136-d obs on the same 34-d frame every other round-3 arm reads once."""
  policy = make_policy(np.zeros(11), history=4, profile=SHAPES)
  assert (policy.history_length, policy.depth_hw) == (4, (120, 160))
  obs = policy.observe(HOME, np.zeros(11, np.float32))
  assert obs.shape == (4 * SHAPES.frame_dim,)


# --- the depth frame is the checkpoint's, not the bench's ---------------------


def test_a_profile_that_disagrees_with_the_graph_is_refused(make_policy):
  """The graph wins. It carries the CNN, so it carries the input size.

  Every checkpoint so far reads 120x160, so this has never fired in anger --
  but it is what keeps a future frame change from being a policy that quietly
  reads a squashed picture, and it is checked rather than trusted because the
  graph can settle it and nothing else in the profile can.
  """
  with pytest.raises(ValueError, match="expects camera"):
    make_policy(np.zeros(11), profile=CUBE_R3, camera_hw=(240, 320))


def test_the_declared_frame_and_stride_make_the_grid_the_weights_carry():
  """The pair, not either half, is what identifies the vision front end.

  A 30x40 spatial-softmax grid is a buffer in the checkpoint, and it is a
  120x160 frame at stride 1/2/2 or a 240x320 one at 2/2/2 with equal validity --
  both rebuild and both load with `strict=True`. Declaring both here is what
  lets `scripts/export_student_onnx.py` check one against the other; these are
  the grids the four round-3 checkpoints actually carry.
  """
  grids = {
    "square-fixh": (15, 20),
    "square-varh": (15, 20),
    "cube": (30, 40),
    "everyshape": (30, 40),
  }
  for name, want in grids.items():
    profile = profiles.PROFILES[name]
    h, w = profile.depth_hw
    for stride in profile.cnn_stride:  # same padding: ceil(dim / stride)
      h, w = -(-h // stride), -(-w // stride)
    assert (h, w) == want, name
