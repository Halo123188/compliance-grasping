"""The trained student, as a plain function from sensors to joint targets.

Needs onnxruntime and numpy. Deliberately NOT mjlab or mujoco: the training run
exports ONNX on every checkpoint save (ManipulationDistillationRunner), the
export is the STUDENT (the deployable vision policy, not the teacher), and the
observation normalizer is baked into the graph. So the robot host runs the same
weights the cluster evaluated, with none of the simulator on it.

The one thing this file has to get right is the observation vector, because a
wrong one fails silently as a mediocre policy and never as an error:

    obs[0:11]   joint_pos - default_joint_pos      (rad)
    obs[11:22]  joint_vel                          (rad/s)
    obs[22:33]  the action returned by the PREVIOUS step, raw network units
    obs[33]     goal_height, metres above the FLOOR

`obs[22:33]` is the raw action, NOT the joint target it was turned into. Feeding
back the target instead is dimensionally plausible and completely wrong. It is
also the RAW action and not the smoothed command: the limiter below is a
property of the actuator, and the policy observes what it asked for.

SOME CHECKPOINTS STACK A HISTORY OF THAT VECTOR, and the stacking is TERM-MAJOR
rather than frame-major. The width is read off the ONNX at load and everything
below follows from it: 34 is one frame, 136 is four. See `stack_history` for the
layout and for why the plausible reading of "four stacked observations" is the
wrong one.

The second thing to get right is that the returned target is not
`default + scale * action` any more. Checkpoints trained with the env's
`SmoothedJointPositionAction` learn against a slew-limited command, and this
file reproduces that limiter -- see `calib.RATE_LIMIT` for which checkpoints
need it and why running one without it is a crash rather than a regression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import calib

# Where each term of one 34-d frame lives, in the order the training env's
# `student` group lists them. Derived rather than written out so the two cannot
# disagree.
_TERM_SLICES: tuple[slice, ...] = tuple(
  slice(a, b)
  for a, b in zip(
    (0, *np.cumsum(calib.OBS_TERM_WIDTHS)[:-1]),
    np.cumsum(calib.OBS_TERM_WIDTHS),
    strict=True,
  )
)


def stack_history(frames: np.ndarray) -> np.ndarray:
  """`(H, 34)` oldest-to-newest -> the flat `obs` a history checkpoint expects.

  TERM-MAJOR. Each term's own history is contiguous, and the terms follow one
  another -- for H = 4:

      obs[  0: 44]  joint_pos    at t-3, t-2, t-1, t
      obs[ 44: 88]  joint_vel    at t-3, t-2, t-1, t
      obs[ 88:132]  last_action  at t-3, t-2, t-1, t
      obs[132:136]  goal_height  at t-3, t-2, t-1, t

  FRAME-MAJOR -- four whole 34-d observations laid end to end -- is what "a
  history of the observation" sounds like, is the same 136 numbers, and is
  wrong. mjlab's group-level ``history_length`` is applied to each TERM (see
  `ObservationManager._prepare_terms`), so every term is separately buffered
  and only then concatenated. Nothing downstream can tell the two apart: a
  mis-ordered 136-vector is still a valid 136-vector, and the policy just
  behaves worse.

  Confirmed against the checkpoint itself rather than only read off the code.
  The baked-in normalizer's last four dimensions share one mean (0.5492) and
  one std (0.0391), which is goal_height four times over; under frame-major
  they would be dims 33, 67, 101 and 135.

  Within a term the order is chronological, oldest first, because that is what
  `CircularBuffer.buffer` returns.
  """
  return np.concatenate([frames[:, s].reshape(-1) for s in _TERM_SLICES]).astype(
    np.float32
  )


class StudentPolicy:
  """One ONNX session plus the last action. Stateful: call `reset()` per trial."""

  def __init__(self, onnx_path: str | Path, providers: list[str] | None = None):
    self.session = ort.InferenceSession(
      str(onnx_path), providers=providers or ["CPUExecutionProvider"]
    )
    # How many control steps of observation this checkpoint reads: 1 for the
    # plain students, 4 for a history-stacked one. Read off the graph because
    # the metadata cannot say -- `observation_names` describes the TEACHER (see
    # `_check_metadata`), so it is silent about both the student's terms and its
    # history.
    self.history_length = self._check_metadata(onnx_path)
    self._history = np.zeros(
      (self.history_length, calib.OBS_FRAME_DIM), dtype=np.float32
    )
    self._history_filled = False
    self.default = np.asarray(calib.DEFAULT_JOINT_POS, dtype=np.float32)
    self.scale = np.asarray(calib.ACTION_SCALE, dtype=np.float32)
    limits = np.asarray(calib.JOINT_LIMITS, dtype=np.float32)
    self.lo, self.hi = limits[:, 0], limits[:, 1]
    self.last_action = np.zeros(11, dtype=np.float32)

    # Command smoothing, mirroring the training env's action term. `_cmd` is the
    # target actually published, which is NOT the target the network asked for
    # once either of these is on -- see calib.RATE_LIMIT for the rule tying
    # these to the checkpoint. Sim runs the same update once per PHYSICS substep
    # at 200 Hz and the robot host once per control step at 50 Hz; both are
    # rad/s bounds, so the guarantee is the same and only the ramp's fine shape
    # inside one control period differs.
    self.dt = 1.0 / calib.CONTROL_HZ
    self.rate_limit = (
      None if calib.RATE_LIMIT is None else np.asarray(calib.RATE_LIMIT, np.float32)
    )
    self.ema_alpha = (
      0.0 if not calib.EMA_TAU else float(np.exp(-self.dt / calib.EMA_TAU))
    )
    self._cmd = self.default.copy()
    self.last_request = self.default.copy()
    # Per joint, how far the last request exceeded one limiter allowance: 0
    # means the command went out as asked, 1 means twice the permitted motion
    # was requested. This is the sim's `SmoothedJointPositionAction.saturation`,
    # so it is comparable to `Episode_Reward/action_saturation` in the run --
    # and a deployment far from what training logged is the symptom of a
    # RATE_LIMIT that does not match the checkpoint.
    self.last_overdrive = np.zeros(11, dtype=np.float32)

  def _check_metadata(self, onnx_path: str | Path) -> int:
    """Fail loudly if the checkpoint disagrees with calib.py. Returns history depth.

    The ONNX carries the scene constants it was trained with. calib.py holds a
    copy so the rest of the deployment can read them without onnx installed --
    which is only safe if the two are asserted equal at load.
    """
    meta = self.session.get_modelmeta().custom_metadata_map
    if not meta:
      raise ValueError(f"{onnx_path} has no metadata; is this an mjlab export?")

    def _floats(key: str) -> np.ndarray:
      return np.array([float(x) for x in meta[key].split(",")], dtype=np.float32)

    # joint_names in the export keep their `_tf` suffix from the asset; calib.py
    # drops it because the real robot has no such convention.
    exported = [n.removesuffix("_tf") for n in meta["joint_names"].split(",")]
    if tuple(exported) != calib.JOINT_NAMES:
      raise ValueError(
        f"joint order mismatch:\n  onnx  {exported}\n  calib {calib.JOINT_NAMES}"
      )
    for key, ours in (
      ("default_joint_pos", calib.DEFAULT_JOINT_POS),
      ("action_scale", calib.ACTION_SCALE),
      # The gains matter as much as the offsets once a real arm is in the loop:
      # `arm.FlexivArm` sets the impedance controller from calib.JOINT_STIFFNESS,
      # so a checkpoint trained under different gains would be tracked by a
      # controller it never saw.
      ("joint_stiffness", calib.JOINT_STIFFNESS),
      ("joint_damping", calib.JOINT_DAMPING),
    ):
      theirs = _floats(key)
      if not np.allclose(theirs, np.asarray(ours, dtype=np.float32), atol=1e-3):
        raise ValueError(
          f"{key} mismatch:\n  onnx  {theirs}\n  calib {np.asarray(ours)}"
        )

    # The exporter writes the ACTOR group's term names even for a distillation
    # export, so `observation_names` describes the teacher's 43-d vector, not
    # the student's 34-d one. Check the actual input shape instead.
    shapes = {i.name: i.shape for i in self.session.get_inputs()}
    width = shapes.get("obs", [None, None])[1]
    # A whole number of frames, and that number is the history depth. This
    # infers rather than asserts, because the two shipped widths are both
    # legitimate and an operator picking the wrong branch by hand is exactly
    # the silent failure the rest of this file exists to prevent.
    if not isinstance(width, int) or width < 1 or width % calib.OBS_FRAME_DIM:
      raise ValueError(
        f"expected obs a multiple of {calib.OBS_FRAME_DIM} wide, got {shapes}"
      )
    if list(shapes.get("camera", []))[1:] != [1, *calib.DEPTH_HW]:
      raise ValueError(f"expected camera (1,1,120,160), got {shapes}")
    return width // calib.OBS_FRAME_DIM

  def reset(self, joint_pos: np.ndarray | None = None) -> None:
    """Zero the action feedback. Sim does this on every episode reset.

    Pass the MEASURED joint angles when the command smoother is on. The sim
    seeds its command from the joints as they actually are after the reset
    events, and seeding from the nominal home pose instead would open a gap the
    limiter then ramps across at the start of every trial -- a scripted move
    nothing asked for.

    The observation history is dropped rather than zeroed. Sim zeroes the
    buffer on reset and then BACKFILLS every slot with the first frame on the
    next push (`CircularBuffer.append`), so a fresh episode's first observation
    is one pose repeated, never one pose behind three zero frames. Zeroing here
    and letting the history fill up over the first four steps would spend the
    start of every trial in a state the policy has never seen -- and the start
    of a trial is when the arm is furthest from the cube and moving fastest.
    """
    self.last_action[:] = 0.0
    self._history_filled = False
    self._cmd = (
      self.default.copy() if joint_pos is None else joint_pos.astype(np.float32).copy()
    )
    self.last_request = self._cmd.copy()
    self.last_overdrive[:] = 0.0

  def frame(
    self,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    goal_height: float = calib.GOAL_HEIGHT_M,
  ) -> np.ndarray:
    """One control step's obs[34]. Both inputs in JOINT_NAMES order.

    Pure: this is the vector, not the vector plus a push onto the history. Kept
    separate from `observe` so the assembly can be checked against the env's own
    `student` group without a history buffer in the way (`wide_onnx_parity.py`).
    """
    if joint_pos.shape != (11,) or joint_vel.shape != (11,):
      raise ValueError(
        f"expected 11 joints each, got {joint_pos.shape}/{joint_vel.shape}"
      )
    return np.concatenate(
      [
        (joint_pos - self.default).astype(np.float32),
        joint_vel.astype(np.float32),
        self.last_action,
        np.array([goal_height], dtype=np.float32),
      ]
    )

  def observe(
    self,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    goal_height: float = calib.GOAL_HEIGHT_M,
  ) -> np.ndarray:
    """Assemble the network's `obs` input, history and all.

    STATEFUL on a history checkpoint, and it advances the history by one step
    per call: sim pushes its buffers once per control step, so calling this
    twice for one step of the robot tells the policy that twice as much time
    passed as did. `step()` calls it exactly once, which is why run.py goes
    through `step()`.
    """
    frame = self.frame(joint_pos, joint_vel, goal_height)
    if not self._history_filled:
      self._history[:] = frame  # backfill, as sim does on the post-reset push
      self._history_filled = True
    else:
      self._history[:-1] = self._history[1:]
      self._history[-1] = frame
    return stack_history(self._history)

  def act(self, obs: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """obs + depth(1,120,160) in [0,1] -> the joint target to publish[11], rad.

    `obs` is whatever `observe()` returned -- 34 wide, or 34 x history_length.

    Stateful: the returned target is smoothed against the one returned last
    call, so calling this on anything other than the live 50 Hz sequence (or
    without a `reset()` in between) carries the previous trial's command into
    the new one.
    """
    action = self.session.run(
      ["actions"],
      {
        "obs": obs.reshape(1, -1).astype(np.float32),
        "camera": depth.reshape(1, 1, *calib.DEPTH_HW).astype(np.float32),
      },
    )[0].reshape(11)
    self.last_action = action.astype(np.float32)
    target = self.default + self.scale * self.last_action
    # The target the network ASKED for, before any smoothing. Kept so a caller
    # can tell a policy that is being held back from one that is being followed.
    self.last_request = target.astype(np.float32)

    # Smooth BEFORE the joint-limit clamp, so the clamp stays the last thing
    # between the command and the arm's hard stops. `_cmd` therefore carries the
    # UNCLAMPED command, which is what sim does -- there is no clamp there at
    # all -- so a policy that drives into a limit winds up here exactly as much
    # as it did in training, and reversing costs the same number of steps.
    if self.ema_alpha:
      target = self.ema_alpha * self._cmd + (1.0 - self.ema_alpha) * target
    if self.rate_limit is not None:
      step = self.rate_limit * self.dt
      backlog = target - self._cmd
      # Measured BEFORE the clamp: after it, the discarded part is gone.
      self.last_overdrive = np.maximum(np.abs(backlog) / step - 1.0, 0.0)
      target = self._cmd + np.clip(backlog, -step, step)
    self._cmd = target.astype(np.float32)

    return np.clip(target, self.lo, self.hi)

  def step(
    self,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    depth: np.ndarray,
    goal_height: float = calib.GOAL_HEIGHT_M,
  ) -> np.ndarray:
    """observe + act, the whole policy in one call."""
    return self.act(self.observe(joint_pos, joint_vel, goal_height), depth)
