"""Render `scene_cam` depth exactly as the student sees it, for a real-vs-sim diff.

This is the reference half of `deploy/live_view.py`. It needs mjlab and a GPU,
so it runs on the workstation and writes an npz that the deploy venv -- which
deliberately has neither -- can load.

  uv run python scripts/render_sim_depth.py sim_ref.npz --cube-xy 0.52 0.10

The output is the same (120,160) metric depth `deploy/perception.py` produces
from the real D435, through the same recipe: render 640x480 (4:3, which is the
centre-crop of the modelled 848x480 sensor) and area-average by 4. Comparing
anything else compares two different fields of view.

The arm defaults to the policy's home pose, because that is where a run starts.
Pass `--joints` (read from `states().q`) to match any other pose, and
`--no-cube` when the bench does not have one. `CG_FOAM_H=0` removes the foam,
matching a bench with it taken off.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Before `import mujoco`: the classic renderer needs a GL platform chosen at
# import time, and on a headless workstation there is no X display to find one
# from. Set rather than required, because a run that dies in MjrContext reports
# it as "no OpenGL context", which does not sound like a missing env var.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  ARM_BASE_Z,
  TWOFINGER_ARM_HOME,
  twofinger_arm_spec,
)
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (
  CG_FOAM_H,
  RESTING_Z,
  TABLE_X,
  get_cube_spec,
  get_table_spec,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_depth import RENDER_WH, render_observation  # noqa: E402

OUT_HW = (120, 160)


def build(
  cube_xy: tuple[float, float] | None,
  arm_joints: np.ndarray | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The scene, optionally without the cube and at an arbitrary arm pose.

  `cube_xy=None` leaves the cube out entirely, which is what to use when the
  bench does not have one -- a cube the real scene lacks is a 200-pixel
  disagreement sitting in the middle of the comparison.

  `arm_joints` overrides the seven arm angles; the fingers stay at the policy's
  home. Read them off the robot (`states().q`) rather than assuming, because
  the whole point of the comparison is that both sides are in the same pose.
  """
  arm = twofinger_arm_spec()
  for b in arm.worldbody.bodies:
    if b.name == "base":
      b.pos = [0.0, 0.0, ARM_BASE_Z]
  arm.attach(get_table_spec(), prefix="", frame=arm.worldbody.add_frame())
  if cube_xy is not None:
    arm.attach(get_cube_spec(), prefix="", frame=arm.worldbody.add_frame())
  m = arm.compile()
  d = mujoco.MjData(m)
  joint_pos = TWOFINGER_ARM_HOME.joint_pos
  assert joint_pos is not None, "TWOFINGER_ARM_HOME must pin every joint"
  for name, q in joint_pos.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qpos[m.jnt_qposadr[jid]] = q
  if arm_joints is not None:
    for i, q in enumerate(np.asarray(arm_joints, float)[:7]):
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
      d.qpos[m.jnt_qposadr[jid]] = q
  if cube_xy is not None:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    adr = m.jnt_qposadr[m.body_jntadr[bid]]
    d.qpos[adr : adr + 7] = [*cube_xy, RESTING_Z, 1.0, 0.0, 0.0, 0.0]
  mujoco.mj_forward(m, d)
  return m, d


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("out", help="path for the .npz reference")
  ap.add_argument(
    "--cube-xy",
    type=float,
    nargs=2,
    default=(TABLE_X, 0.0),
    metavar=("X", "Y"),
    help="cube centre in the WORLD frame (= base frame + 0.365 in z)",
  )
  ap.add_argument(
    "--no-cube", action="store_true", help="render the bench without the cube"
  )
  ap.add_argument(
    "--joints",
    default=None,
    help=(
      "seven arm angles in rad: a comma-separated list, or the path to a .npy. "
      "Defaults to the policy's home pose. Read them off the robot -- the "
      "comparison is only meaningful with both sides in the same pose."
    ),
  )
  args = ap.parse_args()

  joints = None
  if args.joints:
    joints = (
      np.load(args.joints)
      if args.joints.endswith(".npy")
      else np.array([float(x) for x in args.joints.split(",")])
    )
    if joints.shape[0] < 7:
      print(f"!! --joints needs seven values, got {joints.shape[0]}")
      return 1
  cube = None if args.no_cube else (float(args.cube_xy[0]), float(args.cube_xy[1]))
  m, d = build(cube, joints)
  w, h = RENDER_WH
  renderer = mujoco.Renderer(m, height=h, width=w)
  renderer.enable_depth_rendering()
  depth = render_observation(renderer, d)
  renderer.close()
  np.savez_compressed(
    args.out,
    depth_m=depth,
    cube_xy=np.asarray(args.cube_xy if cube else []),
    arm_joints=np.asarray(joints if joints is not None else []),
    # Stamped so a viewer can print what the reference assumes. Comparing a
    # live frame against a reference rendered for a different bench is the
    # easiest mistake to make here and the hardest to notice: every difference
    # looks like a sim-to-real gap.
    foam_h=np.float32(CG_FOAM_H),
  )
  valid = depth > 0
  print(f"rendered {OUT_HW}, {100 * valid.mean():.1f}% valid -> {args.out}")
  if valid.any():
    v = depth[valid]
    print(
      f"  range p05 {np.percentile(v, 5):.3f}  p50 {np.percentile(v, 50):.3f}  "
      f"p95 {np.percentile(v, 95):.3f} m"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
