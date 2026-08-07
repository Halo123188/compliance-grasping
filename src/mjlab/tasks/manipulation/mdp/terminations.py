from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def cube_lifted(
  env: ManagerBasedRlEnv,
  object_name: str,
  height: float,
  hold_time: float,
  table_z: float = 0.0,
) -> torch.Tensor:
  """Success: the object's centre stays >= ``height`` above the table for
  ``hold_time`` seconds continuously.

  Keeps a per-env consecutive-step counter (reset whenever the object drops
  below the threshold), so a fleeting touch does not count -- the grasp must be
  stable. Terminates the episode when the hold is met.
  """
  obj: Entity = env.scene[object_name]
  above = obj.data.root_link_pos_w[:, 2] > (table_z + height)  # [B]

  key = f"_lift_hold_{object_name}"
  counter = getattr(env, key, None)
  if counter is None or counter.shape[0] != env.num_envs:
    counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, key, counter)
  counter[above] += 1
  counter[~above] = 0

  hold_steps = max(1, math.ceil(hold_time / env.step_dt))
  return counter >= hold_steps


def object_dropped(
  env: ManagerBasedRlEnv,
  object_name: str,
  table_z: float,
  margin: float = 0.05,
) -> torch.Tensor:
  """Failure: the object has fallen off the work surface and cannot be recovered.

  Without this the episode runs to its full length with the object on the floor.
  That is merely wasteful for a PPO run -- the rewards read zero and the policy
  learns nothing from those steps -- but it is fatal for DAgger, where every one
  of those steps becomes a labelled training sample. Measured on the two-finger
  distillation runs: the student knocks the cube off within the first few steps
  and `object_height` sits at 0.197 against a 0.400 table top for the remaining
  ~950 steps of a 1000-step episode, so essentially the entire training set is
  the teacher's opinion about a cube on the floor.
  """
  obj: Entity = env.scene[object_name]
  return obj.data.root_link_pos_w[:, 2] < (table_z - margin)


def nonfinite_state(
  env: ManagerBasedRlEnv,
  asset_names: tuple[str, ...] = ("robot", "cube"),
) -> torch.Tensor:
  """Terminate envs whose physics state has gone non-finite (NaN/Inf).

  mujoco-warp very occasionally blows a single env up on a rare high-speed
  contact; rsl_rl then aborts the whole run when that NaN reaches the actor
  observation. Detecting the bad env here routes it through the normal reset
  path (its state is re-initialised to finite values before observations are
  computed), so one unlucky env costs one episode instead of the entire run.
  """
  bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in asset_names:
    data = env.scene[name].data
    bad |= ~torch.isfinite(data.root_link_pos_w).all(dim=-1)
    bad |= ~torch.isfinite(data.root_link_lin_vel_w).all(dim=-1)
    if data.joint_pos is not None and data.joint_pos.numel() > 0:
      bad |= ~torch.isfinite(data.joint_pos).all(dim=-1)
      bad |= ~torch.isfinite(data.joint_vel).all(dim=-1)
  return bad


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)
