"""Gate the wide-claw domain randomization before spending a GPU-day on it.

Every failure mode this catches is silent. A DR event whose `asset_cfg` matches
nothing does not raise -- it randomizes an empty selection, the field stays at
its single compile-time value, and training proceeds at exactly the confidence
level that made the randomization necessary in the first place. The only way to
know a range is live is to read the per-env model back and see it spread.

Checks, in order:
  1. the actuator split did not change the action space or the joint ordering
  2. every physics DR term actually moved its field, with the observed spread
     printed against the range that was asked for
  3. the cube's inertia is consistent with its randomized mass (pseudo_inertia,
     not body_mass -- the whole reason for using it)
  4. every perception DR term moved, and the depth stream shows range noise,
     speckle dropout, the one-sided occlusion shadow, surface blinding that
     lands on the claw and cube rather than the bench, and a per-episode scale
     error

  uv run python scripts/wide_dr_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp.observations import _global_geom_ids
from mjlab.tasks.registry import load_env_cfg

TEACHER = "Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr"
STUDENT = "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr"
PLAIN = "Mjlab-Grasp-TwoFingerWide-Flexiv-Success"
N = 64

_fails: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
  print(f"  {'OK  ' if ok else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
  if not ok:
    _fails.append(label)


def cam_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  obs = env.observation_manager.compute()
  assert isinstance(obs, dict)
  out = obs["camera"]
  assert isinstance(out, torch.Tensor)
  return out


def _resolved(env: ManagerBasedRlEnv, name: str, geoms: tuple[str, ...]):
  """A SceneEntityCfg resolved by hand, since we bypass the term manager here."""
  cfg = SceneEntityCfg(name, geom_names=geoms)
  cfg.resolve(env.scene)
  return cfg


def spread(t: torch.Tensor) -> tuple[float, float]:
  a = t.detach().float().cpu().numpy()
  return float(a.min()), float(a.max())


def build(task: str, **over) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  for k, v in over.items():
    setattr(cfg, k, v)
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)


def name_ids(model, kind: str, names: list[str]) -> list[int]:
  import mujoco

  objtype = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "camera": mujoco.mjtObj.mjOBJ_CAMERA,
    "joint": mujoco.mjtObj.mjOBJ_JOINT,
    "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
  }[kind]
  out = []
  for n in names:
    i = mujoco.mj_name2id(model, objtype, n)
    if i < 0:
      raise SystemExit(f"no {kind} named {n!r} in the compiled model")
    out.append(i)
  return out


def main() -> None:
  # --- 1. the actuator split is a no-op for control -------------------------
  print("\n[1] actuator split (arm group / finger group) is control-neutral")
  plain = build(PLAIN)
  ref_dim = plain.action_manager.total_action_dim
  ref_joints = list(plain.scene["robot"].joint_names)
  ref_acts = list(plain.scene["robot"].actuator_names)
  ngroups = len(plain.scene["robot"].actuators)
  plain.close()
  check(ngroups == 2, "two actuator groups", f"got {ngroups}")
  check(ref_dim == 11, "action dim", f"{ref_dim}")

  env = build(TEACHER)
  robot = env.scene["robot"]
  check(
    env.action_manager.total_action_dim == ref_dim,
    "action dim unchanged vs the non-DR task",
  )
  check(list(robot.joint_names) == ref_joints, "joint ordering unchanged")
  check(list(robot.actuator_names) == ref_acts, "actuator ordering unchanged")

  m = env.sim.mj_model
  sm = env.sim.model

  # --- 2. physics DR is live ------------------------------------------------
  print("\n[2] physics DR: per-env spread of each randomized field")

  # Entity names are prefixed once the scene attaches them; SceneEntityCfg
  # takes the bare name, the compiled model carries "robot/".
  hand = [
    f"robot/{b}"
    for b in ("hand_base_tf", "left_1_tf", "left_2_tf", "right_1_tf", "right_2_tf")
  ]
  hand_ids = name_ids(m, "body", hand)
  lo, hi = spread(sm.body_mass[:, hand_ids].sum(dim=1))
  nominal = float(m.body_mass[hand_ids].sum())
  check(
    hi / lo > 1.6,
    "hand mass",
    f"total {lo * 1e3:.1f}..{hi * 1e3:.1f} g about {nominal * 1e3:.1f}, "
    f"ratio {hi / lo:.2f} (asked 1.4/0.6 = 2.33)",
  )
  ilo, ihi = spread(sm.body_inertia[:, hand_ids].sum(dim=1).sum(dim=-1))
  check(ihi / ilo > 1.6, "hand inertia moved with it", f"ratio {ihi / ilo:.2f}")

  arm_j = [f"robot/joint{i}" for i in range(1, 8)]
  jnt_ids = name_ids(m, "joint", arm_j)
  dof_ids = [int(m.jnt_dofadr[j]) for j in jnt_ids]
  for field, label in (
    ("dof_damping", "arm damping"),
    ("dof_frictionloss", "arm friction"),
  ):
    v = getattr(sm, field)[:, dof_ids]
    ratio = float((v.max(dim=0).values / v.min(dim=0).values.clamp(min=1e-9)).max())
    check(ratio > 1.3, label, f"max per-dof ratio {ratio:.2f} (asked 1.2/0.8 = 1.50)")

  arm_a = name_ids(m, "actuator", arm_j)
  fin_a = name_ids(
    m,
    "actuator",
    [f"robot/{j}" for j in ("left_1_tf", "left_2_tf", "right_1_tf", "right_2_tf")],
  )
  for ids, label, want in (
    (arm_a, "arm kp", 1.5),
    (fin_a, "finger kp", 2.33),
  ):
    v = sm.actuator_gainprm[:, ids, 0].abs()
    ratio = float((v.max(dim=0).values / v.min(dim=0).values.clamp(min=1e-9)).max())
    check(
      ratio > 1.0 + (want - 1.0) * 0.6, label, f"ratio {ratio:.2f} (asked {want:.2f})"
    )
  # The finger band must be visibly wider than the arm's, which is the entire
  # point of splitting the groups.
  arm_r = float(
    (
      sm.actuator_gainprm[:, arm_a, 0].abs().max(dim=0).values
      / sm.actuator_gainprm[:, arm_a, 0].abs().min(dim=0).values
    ).max()
  )
  fin_r = float(
    (
      sm.actuator_gainprm[:, fin_a, 0].abs().max(dim=0).values
      / sm.actuator_gainprm[:, fin_a, 0].abs().min(dim=0).values
    ).max()
  )
  check(fin_r > arm_r, "finger band wider than arm band", f"{fin_r:.2f} vs {arm_r:.2f}")

  # The bench is checked in WORLD coordinates, not in the model field. The table
  # is a fixed-base entity, and mjlab places fixed entities by writing a mocap
  # pose every reset -- for a mocap body MuJoCo takes xpos from mocap_pos and
  # ignores body_pos entirely. So a body_pos draw can spread perfectly in the
  # model and move nothing at all, which is the exact shape of silent failure
  # this script exists to catch.
  table_id = name_ids(m, "body", ["table/table"])[0]
  lo, hi = spread(sm.body_pos[:, table_id, 2])
  print(f"        (model body_pos spread {lo * 1e3:+.1f}..{hi * 1e3:+.1f} mm)")
  env.reset()
  z = env.sim.data.xpos[:, table_id, 2]
  lo, hi = spread(z)
  check(hi - lo > 0.004, "bench height REACHES the world", f"{(hi - lo) * 1e3:.1f} mm")

  qadr = [int(m.jnt_qposadr[j]) for j in jnt_ids]
  bias = robot.data.encoder_bias
  lo, hi = spread(bias)
  check(hi - lo > 0.004, "encoder bias", f"{lo * 1e3:+.2f}..{hi * 1e3:+.2f} mrad")
  check(
    env.cfg.observations["actor"].terms["joint_pos"].params.get("biased") is True,
    "joint_pos observation reads the biased encoder",
  )

  # Start pose. reset_robot_joints is a reset event, so read qpos not qpos0.
  q = env.sim.data.qpos[:, qadr]
  lo, hi = spread(q - q.mean(dim=0, keepdim=True))
  check(hi - lo > 0.02, "start joint pose", f"{lo * 1e3:+.1f}..{hi * 1e3:+.1f} mrad")

  # --- 3. the cube: mass range, and inertia that agrees with it -------------
  print("\n[3] cube (reset-mode DR, so this is one episode's draw)")
  cube_id = name_ids(m, "body", ["cube/cube"])[0]
  cube_g = name_ids(m, "geom", ["cube/cube"])[0]
  masses, frics = [], []
  for _ in range(12):
    env.reset()
    masses.append(sm.body_mass[:, cube_id].clone())
    frics.append(sm.geom_friction[:, cube_g, 0].clone())
  mass = torch.cat(masses)
  fric = torch.cat(frics)
  lo, hi = spread(mass)
  check(
    lo < 0.045 and hi > 0.12,
    "cube mass",
    f"{lo * 1e3:.1f}..{hi * 1e3:.1f} g (asked 30..150)",
  )
  lo, hi = spread(fric)
  check(
    lo < 0.65 and hi > 1.05,
    "cube slide friction",
    f"{lo:.2f}..{hi:.2f} (asked 0.50..1.20)",
  )

  # A solid cube has I = m*a^2/6 on every axis. If inertia tracked mass, this
  # ratio is the same for every env; if only the mass moved it is not.
  #
  # `a` is read PER ENV from the model, not from the compile-time 50 mm.
  # `dr_cube_scale` draws +-10% and carries the inertia by s^2, so a fixed side
  # makes this check fail by exactly 1.1^2 - 1 = 21% on the largest draw --
  # reporting the size DR as an inertia bug.
  side = 2.0 * sm.geom_size[:, cube_g, 0]
  expect = sm.body_mass[:, cube_id] * side * side / 6.0
  rel = float((sm.body_inertia[:, cube_id, 0] / expect - 1.0).abs().max())
  check(
    rel < 0.02, "cube inertia consistent with its mass", f"max rel err {rel * 100:.2f}%"
  )

  env.close()

  # --- 4. perception DR ------------------------------------------------------
  print("\n[4] perception DR (student env)")
  senv = build(STUDENT)
  sm2 = senv.sim.model
  m2 = senv.sim.mj_model
  cam = name_ids(m2, "camera", ["robot/scene_cam"])[0]
  # PER AXIS. The three components of cam_pos are metres apart from each other,
  # so a min/max over all three reports the camera's nominal offset from the arm
  # base and passes whether or not anything was randomized.
  span = (
    sm2.cam_pos[:, cam, :].max(dim=0).values - sm2.cam_pos[:, cam, :].min(dim=0).values
  )
  check(
    float(span.min()) > 0.004,
    "camera position",
    f"per-axis span {[round(float(v) * 1e3, 2) for v in span]} mm (asked 6.0)",
  )
  q = sm2.cam_quat[:, cam, :]
  check(
    float(q.std(dim=0).max()) > 1e-4,
    "camera orientation",
    f"max std {float(q.std(dim=0).max()):.2e}",
  )
  lo, hi = spread(sm2.cam_fovy[:, cam])
  check(hi > lo, "camera fovy", f"{lo:.4f}..{hi:.4f} deg")

  # The depth terms are exercised by calling the observation function DIRECTLY,
  # one effect at a time, rather than through the observation manager.
  # `ObservationManager.compute()` returns a cached `_obs_buffer` unless asked to
  # update history, so two manager calls hand back the identical tensor and any
  # frame-to-frame comparison reads exactly 0 no matter what the noise is doing.
  # Calling the function isolates the effect from that cache and from every
  # other term, and a separate assertion below confirms the cfg actually wires
  # the parameters in -- checking the function alone would pass on an env where
  # none of this is switched on.
  from mjlab.tasks.manipulation.config.flexiv_two_finger_wide import env_cfgs as ec

  # RESET, THEN STEP, and both are load-bearing.
  #
  # step() first because the camera is only rendered by `sim.sense()`, which
  # runs inside it -- on a freshly built env `sensor.data.depth` is still all
  # zeros, and a frame of zeros clamps to min_depth and reads as a uniform
  # 0.0033 everywhere. Every depth statistic taken off that frame is meaningless
  # while looking entirely plausible.
  #
  # reset() first because a freshly built env has not run its reset events or
  # resampled its command: the arm sits at qpos = 0 and the cube has not been
  # placed. The frame that comes back is a perfectly reasonable-looking picture
  # of the bench with NEITHER THE CLAW NOR THE CUBE IN IT, so anything measured
  # per-object silently measures nothing. Depth noise and the occlusion shadow
  # do not notice -- they work off the table and the wall just as happily --
  # which is exactly why this went unseen until the blinding checks below asked
  # a question that only the claw and the cube could answer.
  senv.reset()
  senv.step(torch.zeros(N, senv.action_manager.total_action_dim, device=senv.device))

  def depth(**kw) -> torch.Tensor:
    return manipulation_mdp.camera_depth(senv, "d435", cutoff_distance=3.0, **kw)

  def zero_pct(t: torch.Tensor) -> float:
    return float((t == 0.0).float().mean()) * 100.0

  clean = depth()
  print(
    f"        clean frame: {zero_pct(clean):.2f}% zero, "
    f"depth {float(clean.min()):.4f}..{float(clean.max()):.4f}, "
    f"{float((clean < 0.02).float().mean()) * 100:.1f}% below 0.02"
  )
  z_noise = zero_pct(depth(range_noise=ec._DEPTH_NOISE))
  z_drop = zero_pct(depth(dropout_prob=ec._DEPTH_DROPOUT))
  shadow_kw = dict(
    shadow_focal_px=ec._DEPTH_SHADOW_FOCAL_PX,
    shadow_baseline=ec._DEPTH_SHADOW_BASELINE,
    shadow_fill=ec._DEPTH_SHADOW_FILL,
    shadow_step=ec._DEPTH_SHADOW_STEP,
  )
  grip_cfg = _resolved(senv, "robot", ec._CLAW_VIS_GEOMS)
  obj_cfg = _resolved(senv, "cube", ("cube",))
  blind_kw = dict(
    blind_gripper_cfg=grip_cfg,
    blind_gripper_prob=ec._DEPTH_BLIND_GRIPPER,
    blind_object_cfg=obj_cfg,
    blind_object_prob=ec._DEPTH_BLIND_OBJECT,
  )
  # Blinding can only fire on pixels the camera actually returns for those
  # geoms, so an empty target is the one way it reports "working" while doing
  # nothing. Counted here, before the effect, so a zero rate can be told apart
  # from a zero target.
  seg0 = senv.scene["d435"].data.segmentation
  assert seg0 is not None, "the camera is not rendering segmentation"
  vis = {}
  for label, c in (("claw", grip_cfg), ("cube", obj_cfg)):
    ids_c = _global_geom_ids(senv, c)
    vis[label] = int((seg0[..., 0].unsqueeze(-1) == ids_c).any(-1).sum())
  cam_cfg = next(
    s
    for s in (senv.cfg.scene.sensors or ())
    if isinstance(s, CameraSensorCfg) and s.name == "d435"
  )
  print(
    f"        sensor data_types {tuple(cam_cfg.data_types)}, "
    f"seg ids present {sorted(torch.unique(seg0[..., 0]).tolist())[:14]}"
  )
  print(
    f"        wanted: claw {_global_geom_ids(senv, grip_cfg).tolist()}, "
    f"cube {_global_geom_ids(senv, obj_cfg).tolist()} -> "
    f"claw {vis['claw']} px, cube {vis['cube']} px"
  )
  check(min(vis.values()) > 0, "blind targets are in frame", str(vis))

  z_shadow = zero_pct(depth(**shadow_kw))
  z_blind = zero_pct(depth(**blind_kw))
  print(
    f"        zero%% by cause: range_noise {z_noise:.2f}  "
    f"dropout {z_drop:.2f}  shadow {z_shadow:.2f}  blind {z_blind:.2f}"
  )
  check(0.3 < z_drop < 1.5, "i.i.d. dropout", f"{z_drop:.2f}% (asked 0.5-1.0)")
  check(z_shadow > 0.05, "occlusion shadow", f"{z_shadow:.2f}% of the frame")

  # The shadow is COUPLED TO THE SCENE, which is the whole difference between
  # it and the random blocks it replaces: every pixel it invalidates has a
  # background -> foreground step within reach to its right, and none of them
  # is anywhere else. Re-derived here from the raw sensor rather than from the
  # observation, so a wiring error cannot make this agree with itself.
  zm = senv.scene["d435"].data.depth
  assert zm is not None
  zm = zm.permute(0, 3, 1, 2).clamp(0.01, 3.0)
  step = (zm[..., :-1] - zm[..., 1:]) > ec._DEPTH_SHADOW_STEP
  reach = torch.zeros_like(zm, dtype=torch.bool)
  for k in range(16):
    reach[..., : step.shape[-1] - k] |= step[..., k:]
  new_invalid = (depth(**shadow_kw) == 0.0) & (clean != 0.0)
  beside = float((new_invalid & reach).sum() / new_invalid.sum().clamp_min(1))
  check(beside > 0.999, "shadow sits beside a depth step", f"{beside * 100:.2f}%")

  # Blinding lands ON the claw and the cube, not on the bench: every blinded
  # pixel must be a pixel that segmentation says belongs to one of those geoms.
  seg = senv.scene["d435"].data.segmentation
  assert seg is not None, "the camera is not rendering segmentation"
  ids = torch.cat([_global_geom_ids(senv, grip_cfg), _global_geom_ids(senv, obj_cfg)])
  target = ((seg[..., 0].unsqueeze(-1) == ids).any(-1)).unsqueeze(1)
  blinded = (depth(**blind_kw) == 0.0) & ~(depth() == 0.0)
  on_target = float((blinded & target).sum() / blinded.sum().clamp_min(1))
  check(on_target > 0.99, "blinding lands on its geoms", f"{on_target * 100:.1f}% on")

  # Range noise: two evaluations of the same rendered frame must differ, on
  # most pixels and by no more than the half-width asked for.
  a, b = depth(range_noise=ec._DEPTH_NOISE), depth(range_noise=ec._DEPTH_NOISE)
  d = (a - b).abs()
  frac_moved = float((d > 1e-6).float().mean())
  check(
    frac_moved > 0.9 and float(d.max()) < 2.1 * ec._DEPTH_NOISE,
    "range noise",
    f"{frac_moved * 100:.1f}% of pixels moved, max delta {float(d.max()):.4f} "
    f"(bound {2 * ec._DEPTH_NOISE:.3f})",
  )

  # Per-episode scale error: constant within an episode, different across envs.
  senv.reset()
  g = depth(scale_err=ec._DEPTH_SCALE_ERR)
  h = depth(scale_err=ec._DEPTH_SCALE_ERR)
  check(
    float((g - h).abs().max()) == 0.0,
    "depth scale error is held for the episode",
    "two evaluations inside one episode are identical",
  )

  # ...and the cfg actually asks for all of it.
  dparams = senv.cfg.observations["camera"].terms["d435_depth"].params
  missing = [
    k
    for k in (
      "range_noise",
      "dropout_prob",
      "scale_err",
      "shadow_focal_px",
      "blind_gripper_cfg",
      "blind_object_cfg",
    )
    if not dparams.get(k)
  ]
  check(
    not missing, "cfg wires the depth DR in", f"missing {missing}" if missing else ""
  )

  senv.close()

  print()
  if _fails:
    print(f"{len(_fails)} CHECK(S) FAILED: {', '.join(_fails)}")
    sys.exit(1)
  print("all DR checks passed")


if __name__ == "__main__":
  np.set_printoptions(precision=4, suppress=True)
  main()
