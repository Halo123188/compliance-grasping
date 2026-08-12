"""The deployed command smoother, against the training env's own action term.

`deploy/policy.py` reimplements `SmoothedJointPositionAction` in numpy, because
the robot host runs neither mjlab nor torch. The two have to agree: a policy
trained against a slew-limited command and deployed without one publishes the
raw request, which on this task is a full-speed move into the bench.

The parity that matters is the RATE, not the per-step delta -- sim integrates
the limiter once per physics substep at 200 Hz and the robot host once per
control step at 50 Hz, so the two never take the same steps and only the rad/s
bound is shared.

The ONNX session is stubbed rather than the policy subclassed, so what runs here
is the real `act()` down to the joint-limit clamp it ends with.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from deploy import calib
from deploy.policy import StudentPolicy

DT = 1.0 / calib.CONTROL_HZ
SCALE = np.asarray(calib.ACTION_SCALE, dtype=np.float32)
HOME = np.asarray(calib.DEFAULT_JOINT_POS, dtype=np.float32)


class _StubInput:
  def __init__(self, name: str, shape: list):
    self.name, self.shape = name, shape


class _StubSession:
  """Enough of onnxruntime.InferenceSession to load and to return one action."""

  def __init__(self, action: np.ndarray, history: int = 1, width: int | None = None):
    self.action = np.asarray(action, dtype=np.float32).reshape(1, 11)
    self.history = history
    self.width = history * calib.OBS_FRAME_DIM if width is None else width
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
      "action_scale": self._csv(calib.ACTION_SCALE),
      "joint_stiffness": self._csv(calib.JOINT_STIFFNESS),
      "joint_damping": self._csv(calib.JOINT_DAMPING),
    }
    return type("Meta", (), {"custom_metadata_map": meta})()

  def get_inputs(self):
    return [
      _StubInput("obs", [1, self.width]),
      _StubInput("camera", [1, 1, *calib.DEPTH_HW]),
    ]

  def run(self, names, feeds):  # noqa: ARG002 -- the graph is what is stubbed
    self.last_obs = feeds["obs"]
    return [self.action]


@pytest.fixture
def make_policy(monkeypatch):
  def _make(action, history: int = 1, width: int | None = None) -> StudentPolicy:
    session = _StubSession(action, history, width)
    monkeypatch.setattr(
      "deploy.policy.ort.InferenceSession", lambda *a, **k: session, raising=True
    )
    return StudentPolicy("stub.onnx")

  return _make


@pytest.fixture
def limit() -> np.ndarray:
  if calib.RATE_LIMIT is None:
    pytest.skip("calib.RATE_LIMIT is None; this checkpoint has no limiter")
  return np.asarray(calib.RATE_LIMIT, dtype=np.float32)


def _drive(policy: StudentPolicy, steps: int) -> np.ndarray:
  """Run `steps` inferences and return the last published command."""
  obs, depth = np.zeros(34, np.float32), np.zeros((1, *calib.DEPTH_HW), np.float32)
  cmd = policy._cmd.copy()
  for _ in range(steps):
    cmd = policy.act(obs, depth)
  return cmd


def test_the_command_never_moves_faster_than_the_cap(make_policy, limit):
  """The guarantee the whole term exists for, under a saturating action."""
  policy = make_policy(np.full(11, 8.0))  # ~2x the reach of a trained action
  policy.reset(HOME)
  obs, depth = np.zeros(34, np.float32), np.zeros((1, *calib.DEPTH_HW), np.float32)
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
  limits = np.asarray(calib.JOINT_LIMITS, dtype=np.float32)
  policy = make_policy(np.full(11, 1e4))
  policy.reset(HOME)
  cmd = _drive(policy, 2000)  # long enough for the limiter to ramp into a stop
  assert (cmd <= limits[:, 1] + 1e-6).all()
  assert (cmd >= limits[:, 0] - 1e-6).all()


def test_ema_is_off_unless_calib_says_otherwise(make_policy):
  """The paired checkpoints train with ema_tau null; a stray value is a bug."""
  policy = make_policy(np.zeros(11))
  assert (calib.EMA_TAU is None) == (policy.ema_alpha == 0.0)


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
  with pytest.raises(ValueError, match="multiple of 34"):
    make_policy(np.zeros(11), width=100)


def _frames(policy: StudentPolicy) -> np.ndarray:
  """The (H, 34) view of the last obs the stub was fed, undoing the stacking."""
  session = cast(_StubSession, policy.session)
  assert session.last_obs is not None, "nothing has been fed to the graph yet"
  obs = session.last_obs.reshape(-1)
  h, out = policy.history_length, []
  offset = 0
  for width in calib.OBS_TERM_WIDTHS:
    out.append(obs[offset : offset + width * h].reshape(h, width))
    offset += width * h
  return np.concatenate(out, axis=1)


def test_the_terms_are_laid_out_term_major_not_frame_major(make_policy):
  """The whole point. Each term's history is contiguous; the terms follow."""
  policy = make_policy(np.zeros(11), history=HISTORY)
  policy.reset(HOME)
  obs = policy.observe(HOME + 0.1, np.full(11, 2.0), goal_height=0.55)

  assert obs.shape == (HISTORY * calib.OBS_FRAME_DIM,)
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
  depth = np.zeros((1, *calib.DEPTH_HW), np.float32)
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
