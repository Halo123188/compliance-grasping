"""P2 gate for the wide-claw task: does it build, and can it grasp?

Three checks, in the order that makes a failure attributable:

  A. NAMES     every site, geom pattern, body and camera the reward stack
               references resolves on the compiled scene. A contact sensor whose
               pattern matches nothing reports "no contact" rather than raising,
               so this is the difference between a wiring bug and a learning
               problem -- and they look identical from a reward curve.

  B. REWARDS   every term is read at three poses: the hover the episode resets
               to, the pre-grasp, and the closed grip. A term that is 0.0000 at
               ALL THREE is unreachable, which is the failure that has cost this
               project four training runs.

  C. GRASP     a scripted hover -> pre-grasp -> close -> lift, driven THROUGH the
               action interface (so action scale, default offset and clipping are
               all exercised) rather than by writing ctrl. Reports peak rise and
               whether the grip survives it.

  uv run python scripts/wide_check.py
"""

from __future__ import annotations

import re
import sys

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  ARM_BASE_Z,
  TWOFINGER_ARM_PREGRASP,
  twofinger_arm_spec,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.env_cfgs import (
  ARM_JOINTS,
  CUBE_REGION,
  FINGER_JOINTS,
  FINGERTIP_GEOMS,
  GRASP_SITE,
  LEFT_FINGERTIP_GEOMS,
  LIFT_HEIGHT,
  PAD_SITES,
  PALM_BODY,
  RIGHT_FINGERTIP_GEOMS,
)
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (
  CUBE_HALF,
  RESTING_Z,
  TABLE_X,
  WORK_SURFACE_Z,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api import math as math_utils

TASK = sys.argv[1] if len(sys.argv) > 1 else "Mjlab-Grasp-TwoFingerWide-Flexiv"
N, DEV = 64, "cuda:0"
FAILURES: list[str] = []


def ok(cond: bool, msg: str) -> None:
  print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
  if not cond:
    FAILURES.append(msg)


# --------------------------------------------------------------------- A. names
def check_names() -> None:
  print("=" * 78)
  print("A. NAME RESOLUTION  (compiled robot spec)")
  print("=" * 78)
  m = twofinger_arm_spec().compile()
  sites = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)}
  bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
  geoms = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
  cams = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)}
  joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}

  for s in (*PAD_SITES, GRASP_SITE):
    ok(s in sites, f"site {s!r} exists")
  ok(PALM_BODY in bodies, f"palm body {PALM_BODY!r} exists")
  ok("scene_cam" in cams, "camera 'scene_cam' exists")
  for j in (*ARM_JOINTS, *FINGER_JOINTS):
    ok(j in joints, f"joint {j!r} exists")
  # The COUNT is not the invariant -- it is a property of the collision
  # representation, and hardcoding the fitted boxes' 4-per-link made both of
  # these fail under CG_HAND_COLLISION=coacd_t0.2 for no reason (right_2 is 3
  # hulls there, not 4). What has to hold is that each side resolves to at least
  # one geom and that the combined pattern is exactly their union: a pattern
  # matching NOTHING is the silent failure this check exists for, since a
  # contact sensor then reports no contact rather than raising.
  hit = {
    label: [g for g in geoms if g and re.fullmatch(pat, g)]
    for label, pat in (
      ("both", FINGERTIP_GEOMS),
      ("left", LEFT_FINGERTIP_GEOMS),
      ("right", RIGHT_FINGERTIP_GEOMS),
    )
  }
  for side in ("left", "right"):
    ok(len(hit[side]) >= 1, f"{side} fingertip pattern matches {len(hit[side])} geoms")
  ok(
    set(hit["both"]) == set(hit["left"]) | set(hit["right"]),
    f"FINGERTIP_GEOMS is exactly left+right ({len(hit['both'])} geoms:"
    f" {len(hit['left'])} left, {len(hit['right'])} right)",
  )

  # The spawn box has to sit on the bench with the cube fully supported.
  x0, x1 = CUBE_REGION["x"]
  y0, y1 = CUBE_REGION["y"]
  print(
    f"\n  spawn box  x [{x0:.3f}, {x1:.3f}]  y [{y0:.3f}, {y1:.3f}]  "
    f"cube centre z {RESTING_Z:.3f}  work surface {WORK_SURFACE_Z:.3f}"
  )
  from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (
    _TABLE_X_SPAN,
    _TABLE_Y_SPAN,
  )

  ok(
    x1 + CUBE_HALF < _TABLE_X_SPAN[1] and x0 - CUBE_HALF > _TABLE_X_SPAN[0],
    f"cube stays on the bench in x (table spans {_TABLE_X_SPAN[0]:.3f}"
    f" to {_TABLE_X_SPAN[1]:.3f})",
  )
  ok(
    y1 + CUBE_HALF < _TABLE_Y_SPAN[1],
    f"cube stays on the bench in y (half-width {_TABLE_Y_SPAN[1]:.3f})",
  )


# ------------------------------------------------------------- action plumbing
def action_for(env, targets: dict[str, float]) -> torch.Tensor:
  """The action that commands `targets`, inverting default + scale * action."""
  term = env.action_manager.get_term("joint_pos")
  names = list(term._target_names)
  default, scale = term.offset, term.scale
  assert isinstance(default, torch.Tensor), "use_default_offset should give a tensor"
  tgt = default.clone()
  for i, n in enumerate(names):
    if n in targets:
      tgt[:, i] = targets[n]
  return ((tgt - default) / scale).clamp(-1.0, 1.0)


def _grip_geometry(env) -> None:
  """WHERE on the cube the scripted grip lands, split into its two directions.

  A reward that asks for a pose the hardware cannot reach behaves exactly like
  a reward that is wired wrong: it reads a constant and never moves. This is the
  reachability reference for `contact_face_centring` -- the scripted grip is the
  best pose this claw is known to achieve, so whatever it scores here is the
  practical floor, and a kernel has to be sized against THAT rather than against
  a geometric ideal of zero.

  Printed as the two independent parts, because they are not equally reachable:
  the vertical one is bounded by the claw itself (the gripping face sits 23-33 mm
  above the fingertip, so on a 50 mm cube the grip "lands 9.3 mm high" against a
  rigid surface), while the horizontal one is free -- nothing stops the jaw
  landing on the middle of a face except squaring up to it.
  """
  try:
    sensors = {s: env.scene[f"{s}_cube_pose"] for s in ("left", "right")}
  except Exception:
    return  # arm without the face gate; these sensors only exist with it
  obj = env.scene["cube"]
  print("    contact, in the cube's frame:")
  for side, sensor in sensors.items():
    data = sensor.data
    if data.force is None or data.pos is None:
      continue
    mag = torch.norm(data.force, dim=-1)
    idx = mag.argmax(dim=1)
    best = mag.gather(1, idx.view(-1, 1)).squeeze(1)
    # PER ENV, not on the batch mean. An env whose finger is not touching
    # reports a stale slot position, and one of those among 64 is enough to drag
    # the average past the cube's own half-edge -- which is exactly how this
    # first printed "44.8 mm from the centre-line" on a 25 mm half-cube.
    live = best > 0.05
    if not bool(live.any()):
      print(f"      {side:<5}  no contact")
      continue
    pos = data.pos.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    rel = pos - obj.data.root_link_pos_w
    local = math_utils.quat_apply_inverse(obj.data.root_link_quat_w, rel).abs()
    face = local.argmax(dim=-1, keepdim=True)
    in_face = local.scatter(-1, face, 0.0)[live]
    vert = float(in_face[:, 2].median()) * 1e3
    horiz = float(in_face[:, :2].amax(dim=-1).median()) * 1e3
    print(
      f"      {side:<5}  {horiz:5.1f} mm from the face's centre-line,"
      f" {vert:5.1f} mm above its middle"
      f"   ({int(live.sum())}/{len(live)} envs touching)"
    )


def reward_snapshot(env, label: str) -> dict[str, float]:
  """Per-term reward at the CURRENT state, unscaled by dt.

  Read off `_step_reward`, which `compute()` fills with `value * weight` for
  every term on the step just taken -- so this is what the term is worth right
  now, not an episode average that a long hover would dilute.
  """
  names = env.reward_manager.active_terms
  step = env.reward_manager._step_reward.mean(dim=0)
  vals = {n: float(step[i]) for i, n in enumerate(names)}
  print(f"\n  --- {label} ---")
  for k, v in sorted(vals.items()):
    flag = "  <-- ZERO" if abs(v) < 1e-9 else ""
    print(f"    {k:28s} {v:+12.5f}{flag}")
  return vals


# ------------------------------------------------------- a straight-up lift
def solve_lift_ladder(heights: list[float]) -> list[dict[str, float]]:
  """Arm poses that raise `grasp_site` by each height with the tool UNROTATED.

  Interpolating pre-grasp -> hover in JOINT space does not do this. Those two
  poses differ in joint5/6/7 as well as in the shoulder, so the straight line
  between them rolls and pitches the tool as it rises -- and measured on run
  55048 the cube shears out of the jaw at +45 mm, with the pads then closing to
  the empty-jaw 55 mm. The lift is not the thing that failed there; the path was.

  Damped least squares on the site's 6-DOF error, run on a CPU copy of the same
  model, position tracked to the target and orientation pinned to whatever the
  pre-grasp pose already had.
  """
  arm = twofinger_arm_spec()
  for b in arm.worldbody.bodies:
    if b.name == "base":
      b.pos = [0.0, 0.0, ARM_BASE_Z]
  m = arm.compile()
  d = mujoco.MjData(m)

  jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
  adr = [m.jnt_qposadr[i] for i in jid]
  dof = [m.jnt_dofadr[i] for i in jid]
  lo = np.array([m.jnt_range[i][0] for i in jid])
  hi = np.array([m.jnt_range[i][1] for i in jid])
  sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)

  q = np.array([TWOFINGER_ARM_PREGRASP[j] for j in ARM_JOINTS])
  d.qpos[adr] = q
  mujoco.mj_forward(m, d)
  p0 = d.site_xpos[sid].copy()
  quat0 = np.empty(4)
  mujoco.mju_mat2Quat(quat0, d.site_xmat[sid].copy())

  jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  out = []
  for h in heights:
    target = p0 + np.array([0.0, 0.0, h])
    for _ in range(300):
      d.qpos[adr] = q
      mujoco.mj_forward(m, d)
      e_pos = target - d.site_xpos[sid]
      quat = np.empty(4)
      mujoco.mju_mat2Quat(quat, d.site_xmat[sid].copy())
      dq_quat, e_rot = np.empty(4), np.zeros(3)
      mujoco.mju_negQuat(dq_quat, quat)
      mujoco.mju_mulQuat(dq_quat, quat0, dq_quat)
      mujoco.mju_quat2Vel(e_rot, dq_quat, 1.0)
      err = np.concatenate([e_pos, e_rot])
      if np.linalg.norm(e_pos) < 1e-5 and np.linalg.norm(e_rot) < 1e-4:
        break
      mujoco.mj_jacSite(m, d, jp, jr, sid)
      jac = np.vstack([jp[:, dof], jr[:, dof]])
      lam = 1e-3
      dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6), err)
      q = np.clip(q + 0.5 * dq, lo, hi)
    out.append(dict(zip(ARM_JOINTS, q, strict=True)))
  return out


def main() -> None:
  check_names()

  print()
  print("=" * 78)
  print(f"B/C. ENV  {TASK}")
  print("=" * 78)
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  # LiftingCommand RESPAWNS THE CUBE when it resamples (commands.py
  # _resample_command writes a fresh root pose), and play mode resamples every
  # 5 s. That is correct for training -- it is how the goal and the object get
  # re-drawn within a long episode -- and it silently ruins an open-loop script:
  # run 55054 carried the cube to +42.9 mm with a steady 12.4 N per pad, then at
  # t = 5.08 s the cube teleported back to the bench and the jaw closed on air,
  # which reads exactly like a grip that failed under load. Nothing here should
  # move the cube except the claw.
  cfg.commands["lift_height"].resampling_time_range = (1e9, 1e9)
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  print(f"  built: {N} envs, action dim {env.action_manager.total_action_dim}")

  robot, cube = env.scene["robot"], env.scene["cube"]
  sn = list(robot.site_names)
  li, ri = sn.index(PAD_SITES[0]), sn.index(PAD_SITES[1])

  torch.manual_seed(0)
  env.reset()
  zero = torch.zeros(N, env.action_manager.total_action_dim, device=DEV)

  def pin_cube() -> None:
    """Put every cube at the nominal grasp pose, square to the jaw.

    The reset event spawns the cube uniformly over +-80/+-100 mm with the yaw
    uniform on the whole circle, which is right for training and wrong for this
    check: a fixed scripted pose then grasps the cube only by luck, and the
    result reads as "the claw cannot grip" when what actually happened is that
    the cube was 90 mm away and turned 40 degrees. Run 55044 failed exactly this
    way -- jaw closing to 63.9 mm on nothing, 0% lifted.
    """
    pose = torch.zeros(N, 7, device=DEV)
    pose[:, :3] = env.scene.env_origins + torch.tensor(
      [TABLE_X, 0.0, RESTING_Z], device=DEV
    )
    pose[:, 3] = 1.0
    cube.write_root_link_pose_to_sim(pose)
    cube.write_root_link_velocity_to_sim(torch.zeros(N, 6, device=DEV))

  def run(action, steps: int):
    for _ in range(steps):
      env.step(action)

  def cube_h() -> torch.Tensor:
    return (
      cube.data.root_link_pos_w[:, 2]
      - env.scene.env_origins[:, 2]
      - WORK_SURFACE_Z
      - CUBE_HALF
    )

  gsi = sn.index(GRASP_SITE)

  def grasp_site_z_mm() -> float:
    return (
      float((robot.data.site_pos_w[:, gsi, 2] - env.scene.env_origins[:, 2]).mean())
      * 1e3
    )

  def grip_force_n() -> tuple[float, float]:
    """Net contact force on each fingertip from the cube, in newtons.

    The decisive measurement when a carried cube slips: it separates "the jaw
    opened" (pad separation grows) from "the jaw held but the cube slid through
    it" (separation constant, force low) from "the arm never got there".
    """
    out = []
    for name in ("left_cube_contact", "right_cube_contact"):
      f = env.scene.sensors[name].data.force
      out.append(float(torch.norm(f.reshape(N, -1, 3), dim=-1).sum(-1).mean()))
    return out[0], out[1]

  def pad_sep_mm() -> float:
    p = robot.data.site_pos_w[:, [li, ri]]
    return float(torch.norm(p[:, 0] - p[:, 1], dim=-1).mean()) * 1e3

  stages: list[tuple[str, dict[str, float]]] = []

  # --- hover ---------------------------------------------------------------
  pin_cube()
  run(zero, 30)
  stages.append(("hover (reset pose, zero action)", reward_snapshot(env, "hover")))
  print(
    f"    pad separation {pad_sep_mm():.1f} mm   "
    f"cube rise {float(cube_h().mean()) * 1e3:+.1f} mm"
  )

  # --- pre-grasp -----------------------------------------------------------
  pre = {k: v for k, v in TWOFINGER_ARM_PREGRASP.items() if k in ARM_JOINTS}
  a_pre = action_for(env, pre)
  pin_cube()
  run(a_pre, 60)
  stages.append(("pre-grasp", reward_snapshot(env, "pre-grasp")))
  print(
    f"    pad separation {pad_sep_mm():.1f} mm   "
    f"cube rise {float(cube_h().mean()) * 1e3:+.1f} mm"
  )

  # --- close ---------------------------------------------------------------
  # Index the finger dims BY NAME. `preserve_order` is False, so the action
  # order is the model's actuator order, not the order they were configured in.
  names = list(env.action_manager.get_term("joint_pos")._target_names)
  fdim = [names.index(j) for j in FINGER_JOINTS]
  a_close = a_pre.clone()
  # Proximals to full close, distals left straight -- this claw grips flush at
  # distal 0 and curling here would only spend fingertip clearance.
  #
  # THE SIGNS ARE OPPOSITE, and this is the whole reason the finger limits are
  # given per-side and mirrored rather than as one symmetric range. Closing is
  # left-NEGATIVE and right-POSITIVE: the left proximal runs +0.2746 -> -0.0754
  # and the right runs -0.2746 -> +0.0754. Driving both to -1 swings the right
  # finger WIDE (to -0.6246) while the left closes, which reads as a jaw that
  # refuses to shut -- pad separation going UP under a full close command.
  a_close[:, fdim[0]] = -1.0  # left_1_tf  ->  -0.0754
  a_close[:, fdim[2]] = +1.0  # right_1_tf ->  +0.0754
  run(a_close, 60)
  stages.append(("closed grip", reward_snapshot(env, "closed grip")))
  sep_grip = pad_sep_mm()
  _fl, _fr = grip_force_n()
  print(f"    grip force  left {_fl:.2f} N   right {_fr:.2f} N")
  print(
    f"    pad separation {sep_grip:.1f} mm   "
    f"cube rise {float(cube_h().mean()) * 1e3:+.1f} mm"
  )
  _grip_geometry(env)

  # --- press: does the surface-contact penalty actually fire? -------------
  # The one pattern that had to change and could not be checked by name: the
  # bench's work surface is `foam_*` bodies, not `table`, so the inherited
  # secondary pattern "table" would have matched a real body and reported no
  # contact for ever. Drive the open jaw down into the foam and read the term.
  press_pose = solve_lift_ladder([-0.045])[0]
  a_press = action_for(env, press_pose)
  run(a_press, 60)
  press = reward_snapshot(env, "fingertips pressed into the foam")
  print(f"    fingertip z clearance {grasp_site_z_mm():.1f} mm (grasp site)")

  # --- lift ----------------------------------------------------------------
  pin_cube()
  run(a_pre, 40)
  run(a_close, 60)
  # A straight-up carry, orientation held: the thing the task actually scores.
  heights = list(np.linspace(0.0, 0.16, 33))
  ladder = solve_lift_ladder(heights)
  peak = torch.full((N,), -1e9, device=DEV)
  trace = []
  for h, tgt in zip(heights, ladder, strict=True):
    a = action_for(env, tgt)
    a[:, fdim] = a_close[:, fdim]
    for _ in range(8):
      env.step(a)
    peak = torch.maximum(peak, cube_h())
    fl, fr = grip_force_n()
    trace.append(
      (h * 1e3, float(cube_h().mean()) * 1e3, grasp_site_z_mm(), pad_sep_mm(), fl, fr)
    )
  print("\n  --- lift trace ---")
  print(
    f"    {'cmd mm':>7} {'cube mm':>8} {'site z mm':>10} {'pad mm':>7}"
    f" {'F left N':>9} {'F right N':>10}"
  )
  for cc, hh, zz, ss, fl, fr in trace[::2]:
    print(f"    {cc:7.1f} {hh:8.1f} {zz:10.1f} {ss:7.1f} {fl:9.2f} {fr:10.2f}")

  rise = float(peak.mean()) * 1e3
  held = float((peak > LIFT_HEIGHT).float().mean())
  print(f"    peak rise (mean over {N} envs)  {rise:+.1f} mm")
  print(f"    envs reaching {LIFT_HEIGHT * 1e3:.0f} mm      {held:.1%}")

  # --- solver headroom -----------------------------------------------------
  # njmax and nconmax are JIT-kernel parameters here, not just buffers, so they
  # are tuned down to what compiles -- which makes "is it still enough?" a thing
  # that has to be measured rather than assumed.
  # nconmax and njmax are PER WORLD (sim.py), while `nacon` is the global total
  # across all worlds and `nefc` is already per world. Comparing the raw total
  # against a per-world limit reads as a 70% overflow when the true figure is 4
  # contacts of 150.
  peak_con = int(env.sim.data.nacon.max()) / N
  peak_efc = int(env.sim.data.nefc.max())
  print("\n  --- solver headroom ---")
  print(f"    peak contacts   {peak_con:7.1f}/world of nconmax {cfg.sim.nconmax}")
  print(f"    peak efc rows   {peak_efc:7d}/world of njmax   {cfg.sim.njmax}")

  # --- verdict -------------------------------------------------------------
  print()
  print("=" * 78)
  print("VERDICT")
  print("=" * 78)
  # Only terms that SHOULD pay out somewhere in this sequence are required to.
  # `action_rate_l2` and `joint_vel_hinge` charge for MOVING and every snapshot
  # here is taken at a settled pose under a held action, so a zero is correct.
  # `joint_pos_limits` charges for the arm nearing a hardware limit, which no
  # pose in the task should -- a non-zero there would be the finding.
  stages.append(("pressed into the foam", press))
  required = ("lift", "lift_precise", "pad_touch", "grasp", "fingertip_table_contact")
  all_terms = set().union(*(set(s[1]) for s in stages))
  for t in sorted(all_terms):
    vals = [s[1].get(t, 0.0) for s in stages]
    dead = all(abs(v) < 1e-9 for v in vals)
    if t in required:
      ok(not dead, f"reward {t!r} pays out at some pose")
    else:
      print(
        f"  ----  {t!r} is zero at every static pose (expected)"
        if dead
        else f"  ----  {t!r} non-zero"
      )
  ok(peak_con < cfg.sim.nconmax, "contact buffer has headroom")
  ok(peak_efc < cfg.sim.njmax, "constraint buffer has headroom")
  ok(sep_grip < 80.0, f"jaw actually closed on the cube ({sep_grip:.1f} mm)")
  ok(rise > LIFT_HEIGHT * 1e3, f"scripted lift clears the bar ({rise:+.1f} mm)")

  env.close()
  print()
  if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
      print("  -", f)
    sys.exit(1)
  print("ALL CHECKS PASSED")


if __name__ == "__main__":
  main()
