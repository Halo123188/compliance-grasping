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

THAT LAYOUT IS THE CHECKPOINT'S, NOT THIS FILE'S. Round 2's students read the
object's size as well, between the actions and goal_height, which makes their
frame 35 wide -- so the terms come from `profiles.Profile`, selected from the
run name and asserted against the graph. Everything below is written against
the profile's term list rather than against a fixed 34.

SOME CHECKPOINTS ALSO STACK A HISTORY OF THAT VECTOR, and the stacking is
TERM-MAJOR rather than frame-major. The depth is read off the ONNX at load and
everything follows from it: 34 is one frame, 136 is four. See `stack_history`
for the layout and for why the plausible reading of "four stacked observations"
is the wrong one.

The second thing to get right is that the returned target is not
`default + scale * action` any more. Checkpoints trained with the env's
`SmoothedJointPositionAction` learn against a slew-limited command, and this
file reproduces that limiter -- see `profiles.Profile.rate_limit` for which
checkpoints need it and why running one without it is a crash rather than a
regression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import calib, profiles
from .profiles import Profile


def stack_history(frames: np.ndarray, terms: tuple[slice, ...]) -> np.ndarray:
  """`(H, F)` oldest-to-newest -> the flat `obs` a history checkpoint expects.

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

  `terms` is `Profile.term_slices` -- where each term sits inside one frame.
  Passed in rather than read from a module constant because it is the one thing
  round 2 changed about this layout, and a stacking function that quietly
  assumed four terms would produce a valid 140-vector out of a five-term frame.
  """
  return np.concatenate([frames[:, s].reshape(-1) for s in terms]).astype(np.float32)


class StudentPolicy:
  """One ONNX session plus the last action. Stateful: call `reset()` per trial."""

  def __init__(
    self,
    onnx_path: str | Path,
    providers: list[str] | None = None,
    profile: Profile | None = None,
    cube_half_extent: float | None = None,
  ):
    """`profile` None selects one from the run name and the graph; see
    `profiles.select`. `cube_half_extent` is the object's half-edge in metres
    and is REQUIRED by the checkpoints that observe it -- there is no sensible
    default for an observation, and the size-blind checkpoints ignore it.
    """
    self.session = ort.InferenceSession(
      str(onnx_path), providers=providers or ["CPUExecutionProvider"]
    )
    # How many control steps of observation this checkpoint reads: 1 for the
    # plain students, 4 for a history-stacked one. Read off the graph because
    # the metadata cannot say -- `observation_names` describes the TEACHER (see
    # `_check_metadata`), so it is silent about both the student's terms and its
    # history.
    self.profile, self.history_length = self._check_metadata(onnx_path, profile)
    # (H, W) of the depth frame this checkpoint reads, checked against the graph
    # in `_check_metadata`. `perception.centre_crop_resize` takes the same pair,
    # so the camera and the policy cannot drift apart.
    self.depth_hw = self.profile.depth_hw
    self._history = np.zeros(
      (self.history_length, self.profile.frame_dim), dtype=np.float32
    )
    self._history_filled = False
    self.default = np.asarray(calib.DEFAULT_JOINT_POS, dtype=np.float32)
    self.scale = np.asarray(self.profile.action_scale, dtype=np.float32)
    limits = np.asarray(self.profile.joint_limits, dtype=np.float32)
    self.lo, self.hi = limits[:, 0], limits[:, 1]
    self.last_action = np.zeros(11, dtype=np.float32)
    self.cube_half_extent = self._check_cube(cube_half_extent)

    # Command smoothing, mirroring the training env's action term. `_cmd` is the
    # target actually published, which is NOT the target the network asked for
    # once either of these is on -- see profiles.Profile.rate_limit for the rule
    # tying these to the checkpoint. Sim runs the same update once per PHYSICS
    # substep at 200 Hz and the robot host once per control step at 50 Hz; both
    # are rad/s bounds, so the guarantee is the same and only the ramp's fine
    # shape inside one control period differs.
    self.dt = 1.0 / calib.CONTROL_HZ
    rate = self.profile.rate_limit
    self.rate_limit = None if rate is None else np.asarray(rate, np.float32)
    self.ema_alpha = 0.0
    self.set_ema_tau(self.profile.ema_tau)
    self._cmd = self.default.copy()
    self.last_request = self.default.copy()
    # Per joint, how far the last request exceeded one limiter allowance: 0
    # means the command went out as asked, 1 means twice the permitted motion
    # was requested. This is the sim's `SmoothedJointPositionAction.saturation`,
    # so it is comparable to `Episode_Reward/action_saturation` in the run --
    # and a deployment far from what training logged is the symptom of a
    # RATE_LIMIT that does not match the checkpoint.
    self.last_overdrive = np.zeros(11, dtype=np.float32)

  def set_ema_tau(self, tau: float | None) -> None:
    """Set the command low-pass constant in seconds; 0 or None turns it off.

    Overriding this at the bench is legitimate in a way overriding
    `rate_limit` is not -- see `profiles.Profile.ema_tau`. The lag is not
    something the weights trained against, so the only cost of changing it is
    the one the `-SlowEma` sweep measured. Call it BEFORE `reset()`; changing
    it mid-trial leaves `_cmd` carrying the old filter's state.
    """
    if tau is not None and tau < 0.0:
      raise ValueError(f"ema_tau must be non-negative, got {tau}")
    self.ema_tau = tau
    self.ema_alpha = 0.0 if not tau else float(np.exp(-self.dt / tau))

  def _check_cube(self, half_extent: float | None) -> float:
    """The object's half-edge in metres, checked against the trained range.

    A HARD BOUND, not a preference, for the checkpoints that read it. This term
    goes through the same baked normalizer as goal_height, whose out-of-range
    value took the raw action from |0.95| to |3641| -- and unlike goal_height,
    which an operator picks from a table, this one is measured off a physical
    object and is therefore easy to get wrong by a factor of two (edge vs half
    edge) or a thousand (mm vs m). Both mistakes land tens of sigma out.

    Size-blind checkpoints keep the value for the record and never look at it,
    so `run.py` can report the cube it was flown with either way.
    """
    if half_extent is None:
      if self.profile.observes_cube_size:
        raise ValueError(
          f"{self.profile.name} observes the object's size; pass its half-edge "
          "in metres (deploy.run --cube-mm takes the EDGE in mm)."
        )
      return calib.CUBE_EDGE_MM / 2000.0
    lo, hi = self.profile.cube_half_extent_range
    if self.profile.observes_cube_size and not lo <= half_extent <= hi:
      raise ValueError(
        f"cube half-extent {half_extent:.4f} m is outside {self.profile.name}'s "
        f"trained range [{lo:.4f}, {hi:.4f}] m "
        f"({self.profile.cube_edge_mm[0]:.1f}-{self.profile.cube_edge_mm[1]:.1f} mm "
        "as an edge). This is an observation, so out of range is an "
        "out-of-distribution input, not a slightly wrong setting."
      )
    return float(half_extent)

  def _check_metadata(
    self, onnx_path: str | Path, profile: Profile | None
  ) -> tuple[Profile, int]:
    """Fail loudly if the checkpoint disagrees with what it is being flown as.

    Returns the profile and the history depth. The ONNX carries the scene
    constants it was trained with; calib.py and profiles.py hold copies so the
    rest of the deployment can read them without onnx installed -- which is only
    safe if the two are asserted equal at load.
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
    # The exporter writes the ACTOR group's term names even for a distillation
    # export, so `observation_names` describes the teacher's vector, not the
    # student's. Check the actual input shape instead.
    shapes = {i.name: i.shape for i in self.session.get_inputs()}
    width = shapes.get("obs", [None, None])[1]
    if not isinstance(width, int) or width < 1:
      raise ValueError(f"expected a fixed obs width, got {shapes}")
    # The action scale is the OTHER half of the evidence `profiles.select` runs
    # on -- round 2 moved the fingers from 0.35 to 0.45 -- so it is read here
    # rather than asserted, and asserted below once the profile is known.
    if profile is None:
      profile = profiles.select(onnx_path, _floats("action_scale"), width)

    for key, ours in (
      ("default_joint_pos", calib.DEFAULT_JOINT_POS),
      ("action_scale", profile.action_scale),
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
          f"{key} mismatch against profile {profile.name}:\n  onnx  {theirs}\n"
          f"  ours  {np.asarray(ours)}"
        )

    # A whole number of frames, and that number is the history depth. This
    # infers rather than asserts, because the shipped widths are all legitimate
    # and an operator picking the wrong branch by hand is exactly the silent
    # failure the rest of this file exists to prevent.
    if width % profile.frame_dim:
      raise ValueError(
        f"profile {profile.name} assembles a {profile.frame_dim}-d frame "
        f"({'+'.join(f'{n}:{w}' for n, w in profile.obs_terms)}) but the graph "
        f"reads obs {width}-d, which is not a whole number of them."
      )
    # The depth frame is the ONE profile field the graph can settle, so it is
    # asserted rather than trusted. Every checkpoint to date is 120x160, so this
    # has never yet fired -- but it is the difference between a wrong profile
    # being a load error and being a policy that quietly reads a squashed
    # picture, and the export it guards is one this repo produces offline.
    want = [1, *profile.depth_hw]
    if list(shapes.get("camera", []))[1:] != want:
      raise ValueError(
        f"profile {profile.name} expects camera (1,{want[0]},{want[1]},{want[2]}) "
        f"but the graph reads {shapes.get('camera')}. Either this export is not "
        f"{profile.name} or the profile's depth_hw is wrong; the graph is the "
        "authority."
      )
    return profile, width // profile.frame_dim

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
    cube_half_extent: float | None = None,
  ) -> np.ndarray:
    """One control step's observation. Both joint inputs in JOINT_NAMES order.

    Assembled TERM BY TERM from the profile rather than by concatenating four
    known blocks, so a checkpoint that reads a term this one does not raises
    here instead of being handed a vector of the right width in the wrong order.

    `cube_half_extent` defaults to the one the policy was constructed with,
    which is what the bench wants -- the cube does not change size mid-trial.
    The argument exists for the parity check, which drives many envs whose cubes
    differ.

    Pure: this is the vector, not the vector plus a push onto the history. Kept
    separate from `observe` so the assembly can be checked against the env's own
    `student` group without a history buffer in the way (`wide_onnx_parity.py`).
    """
    if joint_pos.shape != (11,) or joint_vel.shape != (11,):
      raise ValueError(
        f"expected 11 joints each, got {joint_pos.shape}/{joint_vel.shape}"
      )
    size = self.cube_half_extent if cube_half_extent is None else cube_half_extent
    values: dict[str, np.ndarray] = {
      "joint_pos": (joint_pos - self.default).astype(np.float32),
      "joint_vel": joint_vel.astype(np.float32),
      "actions": self.last_action,
      "cube_size": np.array([size], dtype=np.float32),
      "goal_height": np.array([goal_height], dtype=np.float32),
    }
    missing = set(self.profile.term_names) - set(values)
    if missing:
      raise ValueError(
        f"profile {self.profile.name} reads {sorted(missing)}, which this file "
        "does not know how to build. A new observation term is a deployment "
        "change, not a configuration one."
      )
    return np.concatenate([values[name] for name in self.profile.term_names])

  def observe(
    self,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    goal_height: float = calib.GOAL_HEIGHT_M,
    cube_half_extent: float | None = None,
  ) -> np.ndarray:
    """Assemble the network's `obs` input, history and all.

    STATEFUL on a history checkpoint, and it advances the history by one step
    per call: sim pushes its buffers once per control step, so calling this
    twice for one step of the robot tells the policy that twice as much time
    passed as did. `step()` calls it exactly once, which is why run.py goes
    through `step()`.
    """
    frame = self.frame(joint_pos, joint_vel, goal_height, cube_half_extent)
    if not self._history_filled:
      self._history[:] = frame  # backfill, as sim does on the post-reset push
      self._history_filled = True
    else:
      self._history[:-1] = self._history[1:]
      self._history[-1] = frame
    return self.stack_history(self._history)

  def stack_history(self, frames: np.ndarray) -> np.ndarray:
    """`stack_history` against this checkpoint's own term layout."""
    return stack_history(frames, self.profile.term_slices)

  def act(self, obs: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """obs + depth(1,H,W) in [0,1] -> the joint target to publish[11], rad.

    The depth frame is `self.depth_hw`, which is the profile's and is asserted
    against the graph at load. 120x160 for every checkpoint to date.

    `obs` is whatever `observe()` returned -- 34 wide, or 34 x history_length.

    Stateful: the returned target is smoothed against the one returned last
    call, so calling this on anything other than the live 50 Hz sequence (or
    without a `reset()` in between) carries the previous trial's command into
    the new one.
    """
    outputs = self.session.run(
      ["actions"],
      {
        "obs": obs.reshape(1, -1).astype(np.float32),
        "camera": depth.reshape(1, 1, *self.depth_hw).astype(np.float32),
      },
    )
    self.last_action = np.asarray(outputs[0], dtype=np.float32).reshape(11)
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

  def describe(self) -> str:
    """What is about to fly, in the terms that differ between checkpoints."""
    lines = [self.profile.describe()]
    if self.history_length > 1:
      lines.append(
        f"          observation: {self.history_length} stacked control steps "
        f"({self.history_length * self.profile.frame_dim}-d), term-major. The "
        "history is backfilled from the first frame at reset, as sim does."
      )
    if self.profile.observes_cube_size:
      lines.append(
        f"          cube: {self.cube_half_extent * 2000:.1f} mm edge "
        f"({self.cube_half_extent:.4f} m half-extent), fed to the policy every "
        "step."
      )
    if self.ema_tau != self.profile.ema_tau:
      # The line above came from the profile and is now stale, so say so rather
      # than leaving two contradictory numbers on one screen.
      was = "off" if not self.profile.ema_tau else f"{self.profile.ema_tau:.3f} s"
      now = "OFF" if not self.ema_tau else f"{self.ema_tau:.3f} s"
      lines.append(f"          command EMA OVERRIDDEN: {was} -> {now}")
    if self.profile.note:
      lines.append(f"          note: {self.profile.note}")
    return "\n".join(lines)
