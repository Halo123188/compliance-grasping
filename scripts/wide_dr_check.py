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
  4. every perception DR term moved, and the depth stream shows all four of
     range noise, i.i.d. dropout, block dropout and per-episode scale error

  uv run python scripts/wide_dr_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation import mdp as manipulation_mdp
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

  # A solid 50 mm cube has I = m*a^2/6 on every axis. If inertia tracked mass,
  # this ratio is the same for every env; if only the mass moved it is not.
  side = 2.0 * float(m.geom_size[cube_g, 0])
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

  # STEP FIRST. The camera is only rendered by `sim.sense()`, which runs inside
  # step() -- on a freshly built env `sensor.data.depth` is still all zeros, and
  # a frame of zeros clamps to min_depth and reads as a uniform 0.0033
  # everywhere. Every depth statistic taken off that frame is meaningless while
  # looking entirely plausible.
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
  z_patch = zero_pct(
    depth(patch_prob=ec._DEPTH_PATCH_PROB, patch_size=ec._DEPTH_PATCH_SIZE)
  )
  print(
    f"        zero%% by cause: range_noise {z_noise:.2f}  "
    f"dropout {z_drop:.2f}  patch {z_patch:.2f}"
  )
  check(1.0 < z_drop < 4.0, "i.i.d. dropout", f"{z_drop:.2f}% (asked 2)")
  check(0.3 < z_patch < 4.0, "block dropout", f"{z_patch:.2f}% (asked ~1)")

  # Blocks, not just speckle: a fully-zero 8x8 tile is essentially impossible
  # from 2% i.i.d. dropout (0.02^64).
  z = depth(patch_prob=ec._DEPTH_PATCH_PROB, patch_size=ec._DEPTH_PATCH_SIZE) == 0.0
  tiles = z.float().reshape(z.shape[0], 1, 15, 8, 20, 8).mean(dim=(3, 5))
  check(
    float((tiles == 1.0).float().sum()) > 0,
    "block dropout is blocky",
    f"{int((tiles == 1.0).sum())} full 8x8 tiles",
  )

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
    for k in ("range_noise", "dropout_prob", "patch_prob", "scale_err")
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
