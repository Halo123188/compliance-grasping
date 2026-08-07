from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
)
from mjlab.utils.lab_api import math as math_utils

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.

  ``asset_cfg`` may name one site or several; several are averaged. Naming the
  two pad sites makes the reaching term measure from between the fingertips
  rather than from a palm site -- a palm site is a point the hand can hold on
  the object while splaying the fingers away from it, which is the loophole
  ``fingertip_proximity`` exists to patch. Averaging first and comparing the
  MIDPOINT is deliberate: the midpoint is what "the hand is at the cube" means,
  while per-pad errors are what "the jaw is closed on it" means, and that second
  job belongs to the pinch term, not here.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].mean(dim=1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def fingertip_proximity(
  env: ManagerBasedRlEnv,
  object_name: str,
  std: float,
  z_std: float = 0.06,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward getting BOTH finger pads onto the object.

  ``staged_position_reward`` measures reaching from a site on the palm, and a
  palm site is a point the hand can place on the object while splaying the
  fingers as far from it as possible. That is exactly what the policy learns to
  do -- splaying costs nothing and keeps the fingertips clear of the
  table-contact penalty. Measured on the 3000-iteration run 45123: grasp_site
  ended up 8.5 mm from the cube centre while the pads sat 81.6 mm away, jaws
  trained wide open (left_1 +0.98), and the cube was never touched at all.

  This term closes that loophole. It averages the two pads' INDIVIDUAL squared
  errors -- not the error of their midpoint, which stays on the jaw centreline
  no matter how wide the jaws are -- so both pads have to arrive, which is only
  possible by closing around the object.

  ``asset_cfg`` must name the ``left_pad``/``right_pad`` SITES. The collision
  geoms' own frame origins are not usable here: a mesh geom carries a compiled
  pos/quat offset that puts its origin 10-26 mm behind the contact face, by an
  amount that varies with the jaw angle (26 mm wide open, 10 mm closed) -- so a
  geom-based version of this reward is itself gameable by opening the jaws,
  which is the very loophole it exists to close.

  ANISOTROPIC, not axis-dropping. Where a pad touches along the finger's length
  has no single right answer: recording the contact points MuJoCo reports while
  the jaws squeeze the cube gives a spread of 0.3 mm across the closing
  direction but 3.9-12.0 mm along the finger, with the two fingers disagreeing
  about the mean by 26 mm. The closing direction is sharply defined; height is
  not.

  The fix for an uncertain quantity is a loose tolerance, NOT dropping it. An
  earlier version scored the horizontal component only, on the reasoning that
  the site's height could then not be got wrong. It could not -- but nothing
  pulled the hand down either, and the policy settled with the jaws correctly
  centred and correctly opened (3.5 mm horizontal error, 45.7 mm separation for
  a 50 mm cube) while hovering 90.5 mm above it. That traded a ~10 mm problem
  for a ~90 mm one. ``z_std`` is deliberately several times the site's own
  uncertainty, so a 10 mm error in where the site sits along the finger is
  immaterial while a 90 mm hover is not.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pads = robot.data.site_pos_w[:, asset_cfg.site_ids]  # [B, P, 3]
  delta = pads - obj.data.root_link_pos_w.unsqueeze(1)
  error = torch.square(delta[..., :2]).sum(-1) / std**2
  error = error + torch.square(delta[..., 2]) / z_std**2
  return torch.exp(-error.mean(-1))


def antipodal_pinch(
  env: ManagerBasedRlEnv,
  object_name: str,
  width: float,
  sep_std: float = 0.02,
  mid_std: float = 0.04,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the pads straddling the object, with the optimum AT the grasp.

  ``fingertip_proximity`` pulls BOTH pads toward the object CENTRE. Two pads
  cannot both be at the centre -- an antipodal grasp puts them ``width`` apart,
  half a width either side -- so that reward is maximised by a jaw gap of ZERO
  and pays for closing tighter than the object. On a 50 mm cube at std=0.05 a
  perfect grasp scores exp(-0.25) = 0.779 against a ceiling of 1.0, and the
  cheapest way to collect the missing 0.22 is to swing one finger past the cube.
  Measured on run 45530: the hand settled with the knuckles jammed together
  0.21 mm apart and the pads still 24 mm apart, off to one side of the cube.

  This factors the grasp into the two things that actually define it, each with
  its maximum exactly where the grasp is:

    separation  the pads are one object-width apart -- not closer, which is the
                loophole above, and not further, which is not a grasp
    midpoint    the point between them is the object's centre, which is what
                makes it ANTIPODAL rather than two pads on the same side

  The product is used, not the sum: a sum lets a policy bank the midpoint term
  while ignoring separation entirely, which is the pose the old reward found.

  Note both terms are symmetric under swapping the pads, so the order of
  ``asset_cfg.site_ids`` does not matter.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pads = robot.data.site_pos_w[:, asset_cfg.site_ids]  # [B, 2, 3]
  assert pads.shape[1] == 2, "antipodal_pinch needs exactly two pad sites"
  left, right = pads[:, 0], pads[:, 1]

  sep = torch.norm(left - right, dim=-1)
  sep_term = torch.exp(-(((sep - width) / sep_std) ** 2))

  mid = 0.5 * (left + right)
  mid_err = torch.square(mid - obj.data.root_link_pos_w).sum(-1)
  mid_term = torch.exp(-mid_err / mid_std**2)

  return sep_term * mid_term


def pad_site_touch(
  env: ManagerBasedRlEnv,
  object_name: str,
  half_extent: float,
  std: float = 0.02,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the two tuned fingertip sites for touching the object's SURFACE.

  Every other grasp term here scores a proxy -- separation, midpoint, jaw angle,
  pad-plane tilt -- and each proxy turned out to be satisfiable without the
  fingertips ever landing on the cube. This scores the thing itself: the
  distance from each pad site to the nearest point on the cube's surface, which
  is zero exactly when the tip is on the cube and grows in every direction away
  from it, with no pose that scores well while missing.

  Distance is the box SDF evaluated in the object's own frame, so it is correct
  for a rotated cube -- a tipped cube is a failed grasp, but the term should
  still report honestly rather than silently measuring an axis-aligned box.

  The product of the two sides is used, not the sum: a sum pays half the reward
  for planting one fingertip and ignoring the other, which is the one-finger
  skew this task has produced repeatedly.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pads = robot.data.site_pos_w[:, asset_cfg.site_ids]  # [B, 2, 3]
  assert pads.shape[1] == 2, "pad_site_touch needs exactly two pad sites"

  # Into the object frame: q* (p - c) q, via the conjugate rotation.
  rel = pads - obj.data.root_link_pos_w.unsqueeze(1)
  quat = obj.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
  local = math_utils.quat_apply_inverse(quat, rel)

  # Exact SDF of a box, outside and inside.
  q = local.abs() - half_extent
  outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
  inside = torch.clamp(q.max(dim=-1).values, max=0.0)
  dist = outside + inside  # [B, 2], negative when the site is inside the cube

  touch = torch.exp(-((dist / std) ** 2))
  return touch[:, 0] * touch[:, 1]


def lift_hold_bonus(
  env: ManagerBasedRlEnv,
  object_name: str,
  height: float,
  hold_time: float,
  table_z: float = 0.0,
  left_sensor: str = "left_cube_contact",
  right_sensor: str = "right_cube_contact",
) -> torch.Tensor:
  """Pay, every step, for HOLDING the object up -- gripped, above the line.

  Success is currently a TERMINATION that carries no reward, and
  ``TerminationTermCfg.time_out`` defaults False, so it lands in ``terminated``
  rather than ``time_outs`` and rsl_rl zeroes the bootstrap value. Measured on
  run 46934: 14.4% of episodes end this way, at a mean step 540 of 1000, against
  a dense return of ~0.056/step -- so succeeding forfeits ~26 of a ~52 return,
  HALF, and is paid nothing for it. Not succeeding strictly dominates. The
  policy reached 26% anyway, i.e. it is succeeding despite the pay cut and has
  not yet found the exploit (hover just under the line, or drop and re-catch to
  reset the counter), which is a ceiling waiting to become a regression.

  This is the other half of the fix: the termination becomes a time_out so the
  value is bootstrapped, and holding pays here instead.

  Paid per step while the condition holds, not once on the transition, so there
  is nothing to farm: dropping the cube stops the money immediately and the
  re-grasp has to climb the same counter again. The grip test is the same
  bilateral-contact one ``bilateral_grasp`` uses -- without it the term would pay
  for balancing the cube on a fingertip or shoving it up a wall.
  """
  obj: Entity = env.scene[object_name]
  above = obj.data.root_link_pos_w[:, 2] > (table_z + height)

  def _mag(name: str) -> torch.Tensor:
    data = env.scene[name].data
    f = data.force
    assert f is not None, f"contact sensor {name} reports no force"
    return torch.norm(f.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=-1)

  gripped = torch.minimum(_mag(left_sensor), _mag(right_sensor)) > 0.1

  key = f"_hold_bonus_{object_name}"
  counter = getattr(env, key, None)
  if counter is None or counter.shape[0] != env.num_envs:
    counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, key, counter)
  ok = above & gripped
  counter[ok] += 1
  counter[~ok] = 0

  hold_steps = max(1, math.ceil(hold_time / env.step_dt))
  return (counter >= hold_steps).float()


def jaw_alignment_gate(
  env: ManagerBasedRlEnv,
  object_name: str,
  std: float = 0.436,
  floor: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squaring the jaw to a cube face, as a MULTIPLIER in [floor, 1].

  Adding an alignment term has now failed in both directions. At weight 1.0
  (run 45657) rolling the wrist is cheap and grasping is hard, so the cheap term
  became the whole return: jaw_alignment hit 0.946 while pad_pinch collapsed
  0.555 -> 0.033 and grasp went to exactly 0. At weight 0.15 the policy simply
  ignored it -- 0.074, i.e. a raw alignment of ~0.49, no better than chance.
  There is no weight that is both worth chasing and not worth hacking, because
  the term is *separable* from the grasp: it can be collected without one.

  Multiplying removes that. Alignment can no longer be banked on its own -- it
  only ever scales a reward that already requires the fingers on the cube -- so
  there is nothing to hack, and it cannot be ignored either because
  misalignment costs real grasp reward.

  ``floor`` is not cosmetic. A bare product is the 0.0000-forever trap this task
  has hit twice (run 45593's antipodal term, run 46789's 15 mm pad_touch): if the
  gate reads ~0 from the start pose it annihilates the reward it multiplies and
  no gradient toward EITHER factor survives. At floor=0.3 a fully misaligned jaw
  still earns 30% of the grasp reward, so the grasp is learnable first and the
  alignment then sharpens it -- worth up to 3.3x.

  Worth stating plainly: the mechanics do NOT need this. Measured, closing on a
  table-resting cube holds 19.75 N at every yaw from 0 to 45 deg, because the
  fingers rotate the cube square as they close. What is untested is whether that
  rotation -- 41 deg of cube yaw and 17 mm of displacement at the worst angle --
  is survivable during a real moving approach rather than a slow static close.
  That is what this arm measures.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pads = robot.data.site_pos_w[:, asset_cfg.site_ids]
  assert pads.shape[1] == 2, "jaw_alignment_gate needs exactly two pad sites"
  axis = pads[:, 1] - pads[:, 0]
  jaw = torch.atan2(axis[:, 1], axis[:, 0])

  q = obj.data.root_link_quat_w
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

  quarter = torch.pi / 2
  d = (jaw - yaw) % quarter
  mis = torch.minimum(d, quarter - d)
  return floor + (1.0 - floor) * torch.exp(-((mis / std) ** 2))


def gated_by_alignment(
  env: ManagerBasedRlEnv,
  inner: Any,
  object_name: str,
  align_std: float = 0.436,
  align_floor: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Scale any existing reward term by the jaw-alignment gate.

  ``inner`` is the RewardTermCfg this wraps; its func and params are called with
  the wrapped term's own semantics and only its magnitude is modulated. Used to
  put the gate on terms that are non-zero early in training -- see the note in
  env_cfgs on why gating only `grasp` leaves the wrist frozen.

  The inner ``asset_cfg`` MUST be substituted, not passed through. The reward
  manager resolves ``SceneEntityCfg`` name lists into index lists for the params
  of each registered term, and it does not recurse into a cfg nested inside
  another term's params -- so ``inner.params["asset_cfg"].site_ids`` is still its
  ``slice(None)`` default, i.e. EVERY site on the robot. Both failure modes were
  observed: `pad_site_touch` asserts on the shape and killed run 47065 outright,
  while `staged_position_reward` merely takes ``.mean(dim=1)`` and silently
  averaged all of the robot's sites instead of the two pad sites for all 3000
  iterations of run 47064, whose result is therefore void. The loud one was the
  lucky case.
  """
  params = dict(inner.params)
  inner_cfg = params.get("asset_cfg")
  if isinstance(inner_cfg, SceneEntityCfg):
    assert inner_cfg.name == asset_cfg.name, (
      f"gated_by_alignment can only fix up an inner asset_cfg on the same entity:"
      f" inner is {inner_cfg.name!r}, gate is {asset_cfg.name!r}"
    )
    # Compared as tuples: resolving turns the configured tuple into a list.
    assert tuple(inner_cfg.site_names or ()) == tuple(asset_cfg.site_names or ()), (
      f"inner term selects sites {inner_cfg.site_names!r} but the gate resolves"
      f" {asset_cfg.site_names!r}; the resolved ids cannot stand in for it"
    )
    params["asset_cfg"] = asset_cfg
  return inner.func(env, **params) * jaw_alignment_gate(
    env, object_name, align_std, align_floor, asset_cfg
  )


def grasp_aligned(
  env: ManagerBasedRlEnv,
  object_name: str,
  left_sensor: str,
  right_sensor: str,
  force_threshold: float = 0.1,
  firm_scale: float = 5.0,
  align_std: float = 0.436,
  align_floor: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``bilateral_grasp`` scaled by the alignment gate. See both for why."""
  return bilateral_grasp(
    env, left_sensor, right_sensor, force_threshold, firm_scale
  ) * jaw_alignment_gate(env, object_name, align_std, align_floor, asset_cfg)


def jaw_face_alignment(
  env: ManagerBasedRlEnv,
  object_name: str,
  std: float = 0.436,  # 25 deg
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward squaring the jaw to a cube FACE instead of closing on a corner.

  Nothing else in the reward stack mentions orientation. The cube's quaternion
  is in the observation, but with no term scoring alignment the policy has no
  reason to roll the wrist to match it -- and measured on run 45530's successor
  (v2_range, 3000 iters, 64 envs) the jaw ends up a median 13.9 deg off a face,
  with 16% of episodes past 30 deg, i.e. nearer a corner than a face. The pads
  are flat and 28 mm wide, so even the median tilt means one pad CORNER bears on
  the cube rather than the pad face: point contact, little friction area, and a
  cube that tips and slides out of the fingers.

  A box has four side normals (+-x, +-y of its own frame) and so repeats every
  90 deg. The error is therefore folded into a quarter turn and reported as
  0..45 deg, where 0 is flat on a face and 45 is dead on a corner.

  ``std`` is 25 deg by design rather than something tighter. Cube yaw is uniform
  over the full circle and the wrist starts at a fixed roll, so the initial
  misalignment is uniform on 0..45 deg -- mean 22.5. At 25 deg that scores 0.44
  at the start and 1.0 when square, which is a climb. At 15 deg the 45 deg end
  scores 1e-4 and the worst half of the episodes would have no usable gradient
  at all, which is how the antipodal term died in run 45593.

  Yaw is taken from the cube's world quaternion, which is exact while the cube
  sits flat and approximate once it has been tipped onto an edge -- acceptable
  here, since a tipped cube is already a failed grasp.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pads = robot.data.site_pos_w[:, asset_cfg.site_ids]
  assert pads.shape[1] == 2, "jaw_face_alignment needs exactly two pad sites"
  axis = pads[:, 1] - pads[:, 0]
  jaw = torch.atan2(axis[:, 1], axis[:, 0])

  q = obj.data.root_link_quat_w
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

  quarter = torch.pi / 2
  d = (jaw - yaw) % quarter
  mis = torch.minimum(d, quarter - d)  # 0 .. pi/4
  return torch.exp(-((mis / std) ** 2))


def mirror_asymmetry_penalty(
  env: ManagerBasedRlEnv,
  joint_pairs: tuple[tuple[str, str], ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise the two fingers doing different things.

  Both fingers hinge about the same local axis while their meshes are mirrored,
  so closing is ``left`` negative and ``right`` positive: a symmetric jaw has
  ``left + right == 0`` for each pair. Run 45530 settled at left_1=-0.745 with
  right_1=+0.080 -- a sum of -0.665, i.e. one finger swung nearly ten times as
  far as the other, which is the skewed wedge seen in the rollout.

  This is the soft alternative to coupling the two fingers to one action. The
  coupling makes asymmetry unrepresentable, which is stronger but bakes in an
  assumption no real gripper honours -- two physical motors are never exactly
  mirrored, so a policy that can only ever command symmetric pairs has nothing
  to fall back on when the hardware is not. A penalty keeps asymmetry reachable
  and merely expensive.

  ``joint_pairs`` is resolved by name rather than relying on the order of
  ``asset_cfg.joint_ids``, so a pairing can never silently transpose.
  """
  robot: Entity = env.scene[asset_cfg.name]
  names = list(robot.joint_names)
  q = robot.data.joint_pos
  total = torch.zeros(q.shape[0], device=q.device)
  for left_name, right_name in joint_pairs:
    total = total + torch.square(
      q[:, names.index(left_name)] + q[:, names.index(right_name)]
    )
  return total


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def multi_cube_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for the target cube selected by MultiCubeLiftingCommand."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = command.target_object_pos()
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def multi_cube_bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian reward for bringing the selected target cube to goal."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  obj_pos_w = command.target_object_pos()
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  return torch.exp(-position_error / std**2)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)


def action_rate_l2_except(
  env: ManagerBasedRlEnv,
  exclude_joints: tuple[str, ...],
) -> torch.Tensor:
  """`action_rate_l2` with the named joints' action dimensions left out.

  The stock penalty sums the squared action delta over ALL dimensions, which is
  correct for joints the task needs to hold still and wrong for a joint the task
  needs to explore. joint7, the wrist roll, is the case in point: `grasp` reads
  0.0000 for the first ~2000 iterations, so during exactly the window in which
  the wrist would have to discover the +-45 deg that squares the jaw, the only
  gradient touching it is this penalty plus `joint_vel_hinge`. Its learned action
  std collapsed to 0.0083 rad (+-0.5 deg), the lowest of any joint, and stayed
  there; corr(cube yaw, joint7) came out +0.123 and -0.024 on two separate runs.

  Exempting a dimension does not reward moving it -- it only stops charging for
  it, leaving the alignment gate free to decide whether moving is worth it.
  """
  active = env.action_manager.active_terms
  assert len(active) == 1, f"expected a single action term, got {active}"
  term = env.action_manager.get_term(active[0])
  names = list(cast(Any, term).target_names)
  missing = set(exclude_joints) - set(names)
  assert not missing, f"not action dimensions: {sorted(missing)}; have {names}"
  keep = [i for i, n in enumerate(names) if n not in exclude_joints]
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta[:, keep]), dim=1)


def bilateral_grasp(
  env: ManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  force_threshold: float = 0.1,
  firm_scale: float = 5.0,
) -> torch.Tensor:
  """Reward a genuine two-finger pinch on the object.

  Pays out only when the LEFT and RIGHT fingertip colliders are BOTH in contact
  with the cube -- i.e. the object is pinched between the fingers, not merely
  bumped or pressed down onto the table (which would show up as one-sided
  contact). A small firmness bonus (the weaker of the two contact forces,
  normalised) rewards squeezing rather than just grazing.

  This is the missing exploration signal for grasping: while the arm is reaching,
  the fingers straddle the cube, and closing them turns on bilateral contact and
  this reward -- which in turn makes the cube controllable so the pre-existing
  lift/bring rewards become climbable.

  ``force_threshold`` is retained only for config compatibility and is unused:
  the payout is now continuous in the contact force rather than gated on it.
  """

  def _force_mag(sensor_name: str) -> torch.Tensor:
    data = env.scene[sensor_name].data
    if data.force is not None:
      return torch.norm(data.force, dim=-1).amax(dim=-1)  # [B]
    assert data.found is not None
    return data.found.any(dim=-1).float()

  lf = _force_mag(left_sensor)
  rf = _force_mag(right_sensor)
  # The WEAKER of the two forces, normalised. This is naturally zero unless both
  # pads are loaded, so it still means "pinched, not shoved" -- but unlike a hard
  # threshold it is continuous from the very first gram of contact, so there is a
  # gradient pulling a graze into a squeeze. The previous version gated at
  # `force_threshold` and then jumped to 0.5, which meant no gradient at all
  # until the policy had already stumbled into a two-sided grip.
  weaker = torch.minimum(lf, rf)
  del force_threshold  # kept in the signature for cfg compatibility
  return (weaker / firm_scale).clamp(max=1.0)


def contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Soft penalty on contact force reported by a ``ContactSensor``.

  Penalizes only the force magnitude above ``force_threshold`` (summed over the
  sensor's slots), so light grazing contact is nearly free while hard slamming
  is expensive. Used to keep the fingertips from mashing into the table when
  grasping an object that sits on it, without terminating the episode (which
  would kill every legitimate low grasp).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3] -- worst substep over the history window.
    mag = torch.norm(data.force_history, dim=-1).amax(dim=-1)  # [B, N]
  else:
    assert data.force is not None
    mag = torch.norm(data.force, dim=-1)  # [B, N]
  excess = (mag - force_threshold).clamp_min(0.0)
  return excess.sum(dim=-1)
