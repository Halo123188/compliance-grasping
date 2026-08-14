"""Is a wider cube spawn box still visible, and still reachable?

  uv run python scripts/wide_spawn_gate.py [X_MM] [Y_MM]

Two P0 checks that have to pass BEFORE a wider spawn range is trained on,
because both fail silently -- as a policy that is merely worse, with nothing
raising:

  CAMERA   every corner of the cube, at every spawn pose, must be inside the
           D435i's frame AND beyond its 200 mm minimum range. The current box
           was swept against exactly this and clears it by 5.3 mm, which is the
           tightest constraint in the scene; it is what pins TABLE_X at 0.46.
           Widening toward the arm spends that margin first.

  REACH    the arm must be able to put the grasp site over every spawn pose
           within the action envelope. `default + scale * action` means the
           action needed is (q_target - q_default)/scale, and anything past
           ~1.5 sigma is a pose Gaussian exploration never visits -- the bug
           that cost this project four training runs, whose symptom is a reward
           term reading 0.0000 for a whole run rather than any kind of error.

Reports the worst case over a grid of the proposed box, so a PASS means every
corner passes, not the average.
"""

import sys

import mujoco
import numpy as np

from mjlab.tasks.manipulation.config.flexiv_two_finger_wide import V11_ALIGN_KWARGS
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.env_cfgs import (
  GRASP_SITE,
  flexiv_two_finger_grasp_env_cfg,
)
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (
  CUBE_HALF,
  RESTING_Z,
  TABLE_X,
)

# Read inside `main`, not at import: `feasibility_map` is imported by the eval
# plotting, and parsing argv at module scope made THAT script's first argument
# get read as a box half-width.
MIN_RANGE = 0.200  # D435i minimum depth, metres
GRID = 9
ARM_SCALE = 0.40  # joints 1-6, from env_cfgs
SIGMA_LIMIT = 1.5


def _corners(centre: np.ndarray, half: float) -> np.ndarray:
  s = np.array([-1.0, 1.0])
  return np.array(
    [centre + half * np.array([a, b, c]) for a in s for b in s for c in s]
  )


def feasibility_map(
  xs: np.ndarray, ys: np.ndarray, half: float = CUBE_HALF
) -> dict[str, np.ndarray]:
  """Per-pose camera and reach feasibility over a grid, as [len(xs), len(ys)].

  The same three quantities `main` reduces to a worst case, kept per pose so a
  caller can DRAW the boundary rather than be told whether one box clears it:

    range   distance to the nearest cube corner, m. Under `MIN_RANGE` the D435i
            returns nothing there, so the student is blind, not bad.
    frame   worst cube corner in normalised image coords; >= 1 is out of frame.
    sigma   the largest arm action, in units of the exploration std, needed to
            put the grasp site on the cube. NaN where the IK does not converge,
            which is the honest reading of "the arm cannot get there at all".

  Exported because an evaluation that sweeps OUTSIDE the trained spawn box is
  measuring two different things at once -- the policy, and whether the object
  was visible and reachable in the first place -- and they have to be separable
  in the figure. One implementation, so the gate and the map cannot disagree.
  """
  m, d, base, q_home, sid, arm_ids, cam = _scene()
  cam_pos, cam_mat, tx, ty = cam
  out = {k: np.full((len(xs), len(ys)), np.nan) for k in ("range", "frame", "sigma")}
  for i, x in enumerate(xs):
    for j, y in enumerate(ys):
      centre = np.array([x, y, RESTING_Z])
      local = (_corners(centre, half) - cam_pos) @ cam_mat
      depth = -local[:, 2]
      out["range"][i, j] = depth.min()
      u = np.abs(local[:, 0] / np.maximum(depth, 1e-6)) / tx
      v = np.abs(local[:, 1] / np.maximum(depth, 1e-6)) / ty
      out["frame"][i, j] = max(u.max(), v.max())
      out["sigma"][i, j] = _reach_sigma(m, d, base, q_home, sid, arm_ids, centre)
  return out


def _scene():
  cfg = flexiv_two_finger_grasp_env_cfg(**V11_ALIGN_KWARGS)
  spec = cfg.scene.entities["robot"].spec_fn()
  m = spec.compile()
  d = mujoco.MjData(m)
  home = cfg.scene.entities["robot"].init_state.joint_pos or {}
  for name, val in home.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
      d.qpos[m.jnt_qposadr[jid]] = val
  q_home = d.qpos.copy()
  mujoco.mj_forward(m, d)
  base = np.array(cfg.scene.entities["robot"].init_state.pos)
  cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
  assert cid >= 0, "scene_cam not found"
  cam_pos = d.cam_xpos[cid].copy() + base
  cam_mat = d.cam_xmat[cid].reshape(3, 3).copy()
  ty = np.tan(np.radians(m.cam_fovy[cid]) / 2.0)
  sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)
  arm_ids = [
    (
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}"),
      m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")],
      m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")],
    )
    for j in range(1, 7)
  ]
  return m, d, base, q_home, sid, arm_ids, (cam_pos, cam_mat, ty * (160.0 / 120.0), ty)


def _reach_sigma(m, d, base, q_home, sid, arm_ids, centre: np.ndarray) -> float:
  """Damped least squares onto the cube centre; NaN if it does not converge."""
  target = centre - base
  d.qpos[:] = q_home
  mujoco.mj_forward(m, d)
  for _ in range(80):
    err = target - d.site_xpos[sid]
    if np.linalg.norm(err) < 1e-4:
      break
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, sid)
    j6 = jac[:, [dof for _, _, dof in arm_ids]]
    dq = j6.T @ np.linalg.solve(j6 @ j6.T + 1e-4 * np.eye(3), err)
    for (_jid, adr, _), step in zip(arm_ids, dq, strict=True):
      d.qpos[adr] += float(np.clip(step, -0.1, 0.1))
    mujoco.mj_forward(m, d)
  if np.linalg.norm(target - d.site_xpos[sid]) > 2e-3:
    return float("nan")
  return max(abs(d.qpos[adr] - q_home[adr]) / ARM_SCALE for _, adr, _ in arm_ids)


def main() -> None:
  X_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 110.0
  Y_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 140.0
  m, _d, _base, _q, _sid, _arm, cam = _scene()
  cam_pos = cam[0]
  cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
  print(f"scene_cam at {np.round(cam_pos, 4)}  fovy {m.cam_fovy[cid]:.1f} deg")
  print(
    f"proposed spawn: x = {TABLE_X * 1e3:.0f} +- {X_MM:.0f} mm,  y = +- {Y_MM:.0f} mm\n"
  )

  xs = np.linspace(TABLE_X - X_MM / 1e3, TABLE_X + X_MM / 1e3, GRID)
  ys = np.linspace(-Y_MM / 1e3, Y_MM / 1e3, GRID)
  fm = feasibility_map(xs, ys)

  def _where(a: np.ndarray, pick) -> tuple[float, float]:
    i, j = np.unravel_index(pick(a), a.shape)
    return xs[i], ys[j]

  worst_range = float(np.nanmin(fm["range"]))
  at_range = _where(fm["range"], np.nanargmin)
  worst_frame = float(np.nanmax(fm["frame"]))
  at_frame = _where(fm["frame"], np.nanargmax)
  # NaN is "the IK never got there", which is not a small sigma -- nanmax would
  # silently report the worst REACHABLE pose as the worst pose.
  unreachable = int(np.isnan(fm["sigma"]).sum())
  worst_sigma = float(np.nanmax(fm["sigma"]))
  at_sigma = _where(fm["sigma"], np.nanargmax)

  ok = True
  print(f"{'check':<26}{'worst':>12}   where (mm)        verdict")
  r_ok = worst_range > MIN_RANGE
  ok &= r_ok
  print(
    f"{'nearest cube corner':<26}{worst_range * 1e3:9.1f} mm"
    f"   ({at_range[0] * 1e3:.0f}, {at_range[1] * 1e3:+.0f})"
    f"      {'PASS' if r_ok else 'FAIL'}  (min range {MIN_RANGE * 1e3:.0f} mm)"
  )
  f_ok = worst_frame < 1.0
  ok &= f_ok
  print(
    f"{'frame coord (1.0 = edge)':<26}{worst_frame:9.3f}   "
    f"   ({at_frame[0] * 1e3:.0f}, {at_frame[1] * 1e3:+.0f})"
    f"      {'PASS' if f_ok else 'FAIL'}"
  )
  s_ok = worst_sigma < SIGMA_LIMIT and unreachable == 0
  ok &= s_ok
  print(
    f"{'worst arm action (sigma)':<26}{worst_sigma:9.2f}   "
    f"   ({at_sigma[0] * 1e3:.0f}, {at_sigma[1] * 1e3:+.0f})"
    f"      {'PASS' if s_ok else 'FAIL'}  (limit {SIGMA_LIMIT},"
    f" {unreachable} pose(s) unreachable)"
  )
  print(f"\n{'ALL SPAWN CHECKS PASSED' if ok else 'SPAWN GATE FAILED'}")
  if not ok:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
