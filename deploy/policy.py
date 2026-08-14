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
back the target instead is dimensionally plausible and completely wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import calib


class StudentPolicy:
  """One ONNX session plus the last action. Stateful: call `reset()` per trial."""

  def __init__(self, onnx_path: str | Path, providers: list[str] | None = None):
    self.session = ort.InferenceSession(
      str(onnx_path), providers=providers or ["CPUExecutionProvider"]
    )
    self._check_metadata(onnx_path)
    self.default = np.asarray(calib.DEFAULT_JOINT_POS, dtype=np.float32)
    self.scale = np.asarray(calib.ACTION_SCALE, dtype=np.float32)
    limits = np.asarray(calib.JOINT_LIMITS, dtype=np.float32)
    self.lo, self.hi = limits[:, 0], limits[:, 1]
    self.last_action = np.zeros(11, dtype=np.float32)

    # Command smoothing, mirroring the training env's action term. `_cmd` is the
    # target actually published, which is NOT the target the network asked for
    # once either of these is on -- see calib.RATE_LIMIT for the rule tying
    # these to the checkpoint.
    self.dt = 1.0 / calib.CONTROL_HZ
    self.rate_limit = (
      None if calib.RATE_LIMIT is None else np.asarray(calib.RATE_LIMIT, np.float32)
    )
    self.ema_alpha = (
      0.0 if not calib.EMA_TAU else float(np.exp(-self.dt / calib.EMA_TAU))
    )
    self._cmd = self.default.copy()

  def _check_metadata(self, onnx_path: str | Path) -> None:
    """Fail loudly if the checkpoint disagrees with calib.py.

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
    if shapes.get("obs", [None, None])[1] != 34:
      raise ValueError(f"expected obs of width 34, got {shapes}")
    if list(shapes.get("camera", []))[1:] != [1, *calib.DEPTH_HW]:
      raise ValueError(f"expected camera (1,1,120,160), got {shapes}")

  def reset(self, joint_pos: np.ndarray | None = None) -> None:
    """Zero the action feedback. Sim does this on every episode reset.

    Pass the MEASURED joint angles when the command smoother is on. The sim
    seeds its command from the joints as they actually are after the reset
    events, and seeding from the nominal home pose instead would open a gap the
    limiter then ramps across at the start of every trial -- a scripted move
    nothing asked for.
    """
    self.last_action[:] = 0.0
    self._cmd = (
      self.default.copy() if joint_pos is None else joint_pos.astype(np.float32).copy()
    )

  def observe(
    self,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    goal_height: float = calib.GOAL_HEIGHT_M,
  ) -> np.ndarray:
    """Assemble obs[34] from measured joint state. Both inputs in JOINT_NAMES order."""
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

  def act(self, obs: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """obs[34] + depth(1,120,160) in [0,1] -> joint position targets[11], rad."""
    action = self.session.run(
      ["actions"],
      {
        "obs": obs.reshape(1, 34).astype(np.float32),
        "camera": depth.reshape(1, 1, *calib.DEPTH_HW).astype(np.float32),
      },
    )
    # `session.run` is typed as returning a union that includes SparseTensor and
    # plain lists; the graph only ever produces a dense array here, and asarray
    # makes that explicit rather than assuming it.
    action = np.asarray(action[0]).reshape(11)
    self.last_action = action.astype(np.float32)
    target = self.default + self.scale * self.last_action

    # Smooth BEFORE the joint-limit clamp, so the clamp stays the last thing
    # between the command and the arm's hard stops.
    if self.ema_alpha:
      target = self.ema_alpha * self._cmd + (1.0 - self.ema_alpha) * target
    if self.rate_limit is not None:
      step = self.rate_limit * self.dt
      target = self._cmd + np.clip(target - self._cmd, -step, step)
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
