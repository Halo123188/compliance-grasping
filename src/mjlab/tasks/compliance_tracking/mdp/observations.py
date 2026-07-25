"""Observation terms: proprioception-only actor, privileged critic (spec §2).

The split is the entire point of the pipeline.  The **actor** sees only what the
real robot can measure without a wrist force/torque sensor — joint state, joint
torque, forward-kinematic EE state, the task goal, its own previous action, and
the finger state.  External force is *not* an input; it has to be inferred from
the joint-torque history, which is why the actor is recurrent.  Explicitly
withheld: ``F_ext``, ``K_h``, the hand anchor, the teacher target ``x_t``, and
``s``.  A binary grasp-phase flag is allowed — the real controller knows whether
it has commanded a close.

The **critic** gets all of it.  It is never deployed, so hiding the perturbation
state from it buys nothing and costs value-estimate variance: the return depends
strongly on a hidden variable (how hard and how long you are about to be pushed),
and a critic that cannot see it has to average over that, injecting noise into
every advantage estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_KH_NORM_REF = 1000.0
"""Reference stiffness (N/m) normalising log K_h into ~[0, 1] for the critic."""


def _teacher(env: ManagerBasedRlEnv, command_name: str) -> TeacherCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, TeacherCommand)
  return term


# --- actor (proprioception only) --------------------------------------------


def joint_total_torque(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Total measured joint torque ``τ_meas`` = actuator + external.

  The only channel carrying the external wrench.  Deliberately the *raw* total,
  not a model-based ``τ̂_ext`` estimate: an estimate bakes in the dynamics-model
  error we would otherwise be asking the policy to be robust to.
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt = asset_cfg.joint_ids
  return asset.data.qfrc_actuator[:, jnt] + asset.data.qfrc_external[:, jnt]


def ee_pos(env: ManagerBasedRlEnv, command_name: str = "teacher") -> torch.Tensor:
  """EE position relative to the env origin (origin-invariant)."""
  return _teacher(env, command_name).ee_pos_w() - env.scene.env_origins


def ee_lin_vel(env: ManagerBasedRlEnv, command_name: str = "teacher") -> torch.Tensor:
  return _teacher(env, command_name).ee_vel_w()


def goal_error(env: ManagerBasedRlEnv, command_name: str = "teacher") -> torch.Tensor:
  """``x_goal − x_ee``: the *task* goal (where the object is), not ``x_t``.

  Available at deployment (the robot is told what to pick up); says nothing
  about the teacher's yielded target or where along the path ``s`` currently is.
  """
  teacher = _teacher(env, command_name)
  return teacher.goal_pos - teacher.ee_pos_w()


def finger_state(
  env: ManagerBasedRlEnv,
  force_sensor_names: tuple[str, ...],
  normal_axis: int = 0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """``[finger_joint_pos, measured total grasp normal force]`` (N, 2)."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  force = total_grasp_force(env, force_sensor_names, normal_axis).unsqueeze(-1)
  return torch.cat([pos, force], dim=-1)


def sanitized_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Previous action, read back from each term's *sanitized* raw action.

  ``mdp.last_action`` returns the action as the policy emitted it, so a single
  non-finite output would be fed straight back in as an observation and poison
  every subsequent step through the recurrent state. Both action terms here
  ``nan_to_num`` their input on the way in, so reading their stored raw actions
  keeps the observation finite even when the fallback path fires.
  """
  return torch.cat(
    [
      env.action_manager.get_term(name).raw_action
      for name in env.action_manager.active_terms
    ],
    dim=-1,
  )


def grasp_phase_flag(
  env: ManagerBasedRlEnv, command_name: str = "teacher"
) -> torch.Tensor:
  """Binary "closure commanded" flag (N, 1). Allowed on the real robot."""
  return _teacher(env, command_name).grasp_flag.unsqueeze(-1)


def total_grasp_force(
  env: ManagerBasedRlEnv, force_sensor_names: tuple[str, ...], normal_axis: int = 0
) -> torch.Tensor:
  """Total normal force across the fingertip FT sensors (N,)."""
  total = torch.zeros(env.num_envs, device=env.device)
  for name in force_sensor_names:
    total = total + env.scene[name].data[:, normal_axis].abs()
  return total


# --- critic (privileged) ------------------------------------------------------


def privileged_teacher(
  env: ManagerBasedRlEnv, command_name: str = "teacher"
) -> torch.Tensor:
  """Everything the actor is denied, for the value function only.

  ``[x_t − x_ee (3), ẋ_t (3), x_ref(s) − x_ee (3), s (1), ṡ (1),
     F_finger_target (1), grasp_state one-hot-ish (1)]`` = 13 dims.
  """
  t = _teacher(env, command_name)
  ee = t.ee_pos_w()
  return torch.cat(
    [
      t.x_t - ee,
      t.xd_t,
      t.x_ref() - ee,
      t.s.unsqueeze(-1),
      t.s_dot.unsqueeze(-1),
      t.finger_force_target.unsqueeze(-1),
      t.grasp_state.float().unsqueeze(-1),
    ],
    dim=-1,
  )


def pull_direction(
  env: ManagerBasedRlEnv, command_name: str = "teacher"
) -> torch.Tensor:
  """Privileged unit pull direction ``u`` (N, 3), zero when idle.

  DIAGNOSTIC ONLY.  The actor is normally proprioception-only and must infer the
  push from joint torques; this hands it the ground-truth direction so we can
  ask whether *observability* is what blocks directional compliance.  If K∥/K⊥
  drops below 1 with this in the actor obs, the deployable fix is a
  proprioception-derived force estimate; if not, the diagonal-K action space is
  the wall.  Not deployable as-is (needs the privileged perturbation state).
  """
  return _teacher(env, command_name).perturbation.direction


def ext_force(env: ManagerBasedRlEnv, command_name: str = "teacher") -> torch.Tensor:
  """Privileged external force ``F_ext`` (N, 3) — auxiliary-loss LABEL only.

  This is *not* an actor input.  It lives in its own observation group that no
  model set consumes, so it flows to the rollout storage untouched and is read
  in ``AuxPPO.update`` as the regression target for the actor's force head.  The
  policy is thereby taught to *estimate* the push from its own proprioceptive
  (joint-torque) history at train time, with a strong per-step gradient, and
  needs no privileged input at deployment — unlike ``pull_direction``, which
  hands the answer in and cannot be deployed.  See ``rl_aux.AuxRNNModel``.
  """
  return _teacher(env, command_name).perturbation.force


def privileged_perturbation(
  env: ManagerBasedRlEnv, command_name: str = "teacher"
) -> torch.Tensor:
  """Perturbation ground truth + schedule lookahead for the critic.

  ``[F_ext (3), u (3), anchor − x_ee (3), log K_h (1), active (1),
     events_remaining (1)]`` = 12 dims.  ``events_remaining`` is the schedule
  term: without it the critic cannot tell an episode that is about to be pushed
  again from one that is over, and those have very different returns.
  """
  t = _teacher(env, command_name)
  p = t.perturbation
  ee = t.ee_pos_w()
  log_kh = (
    torch.log(p.stiffness.clamp(min=1.0))
    / torch.log(torch.tensor(_KH_NORM_REF, device=p.stiffness.device))
  ).unsqueeze(-1)
  return torch.cat(
    [
      p.force,
      p.direction,
      p.anchor_pos - ee,
      log_kh,
      p.active.float().unsqueeze(-1),
      p.events_remaining.float().unsqueeze(-1),
    ],
    dim=-1,
  )
