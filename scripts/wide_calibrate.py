"""Measure the WIDE claw against the foam bench, on the compiled model.

Every constant the reward stack needs that changed with the gripper, read off
the physics rather than carried over:

  1. aperture table      pad-marker and pad-SITE separation vs the proximal
                         angle, and what curling the distal does to both
  2. PAD_SEP_AT_GRASP    site separation while actually gripping the cube --
                         driven onto it and stopped by it, not commanded into
                         free space. Those are different numbers.
  3. hover geometry      where the reset pose puts the pads relative to the cube,
                         and how much the fingertips clear the foam by
  4. sigma rule          (target - default) / scale for every pose the task needs
                         to reach. Past ~1.5 sigma Gaussian exploration never
                         gets there and the reward reads exactly 0.0000 forever.

  uv run python scripts/wide_calibrate.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  ARM_BASE_Z,
  FINGER_OPEN_ANGLE,
  TWOFINGER_ARM_HOME,
  TWOFINGER_ARM_PREGRASP,
  aperture_mm,
  open_angle_for,
  twofinger_arm_spec,
)
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (
  CUBE_HALF,
  RESTING_Z,
  TABLE_X,
  WORK_SURFACE_Z,
  get_cube_spec,
  get_table_spec,
)

FINGERS = ("left_1_tf", "left_2_tf", "right_1_tf", "right_2_tf")


def build() -> tuple[mujoco.MjModel, mujoco.MjData]:
  arm = twofinger_arm_spec()
  for b in arm.worldbody.bodies:
    if b.name == "base":
      b.pos = [0.0, 0.0, ARM_BASE_Z]
  arm.attach(get_table_spec(), prefix="", frame=arm.worldbody.add_frame())
  arm.attach(get_cube_spec(), prefix="", frame=arm.worldbody.add_frame())
  m = arm.compile()
  return m, mujoco.MjData(m)


def set_arm(m, d, pose: dict[str, float]) -> None:
  for name, q in pose.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qpos[m.jnt_qposadr[jid]] = q
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
      d.ctrl[aid] = q


def put_cube(m, d, xyz, yaw: float = 0.0) -> None:
  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
  adr = m.jnt_qposadr[jid]
  d.qpos[adr : adr + 3] = xyz
  d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def cube_pos(m, d):
  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
  return d.qpos[m.jnt_qposadr[jid] : m.jnt_qposadr[jid] + 3].copy()


def sites(m, d):
  return {
    n: d.site(n).xpos.copy() for n in ("left_pad_tf", "right_pad_tf", "grasp_site_tf")
  }


def markers(m, d):
  return {n: d.body(n).xpos.copy() for n in ("left_pad_tf", "right_pad_tf")}


def tip_z(m, d) -> float:
  """Lowest point of either distal collider -- what actually hits the foam."""
  lo = np.inf
  for i in range(m.ngeom):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    if n and ("left_2_col" in n or "right_2_col" in n):
      lo = min(lo, d.geom_xpos[i][2] - m.geom_rbound[i])
  return float(lo)


def main() -> None:
  m, d = build()
  np.set_printoptions(precision=4, suppress=True)

  # ---------------------------------------------------------------- 1. aperture
  print("=" * 78)
  print("1. APERTURE  (arm at HOME, cube removed; kinematics only)")
  print("=" * 78)
  put_cube(m, d, [2.0, 2.0, 2.0])  # out of the way
  set_arm(m, d, TWOFINGER_ARM_HOME.joint_pos)
  print(
    f"{'prox q':>8} {'dist q':>8} {'marker sep':>11} {'site sep':>10} "
    f"{'site z':>9} {'marker z':>9} {'tip z':>8} {'tip-foam':>9}"
  )
  rows = []
  for dist in (0.0, -0.3, -0.6):
    for prox in (-0.10, 0.0, 0.10, 0.1276, 0.20, 0.2746, 0.35, 0.50):
      set_arm(
        m,
        d,
        {
          "left_1_tf": prox,
          "right_1_tf": -prox,
          "left_2_tf": dist,
          "right_2_tf": -dist,
        },
      )
      mujoco.mj_forward(m, d)
      s, b = sites(m, d), markers(m, d)
      ssep = np.linalg.norm(s["left_pad_tf"] - s["right_pad_tf"]) * 1e3
      bsep = np.linalg.norm(b["left_pad_tf"] - b["right_pad_tf"]) * 1e3
      sz = (s["left_pad_tf"][2] + s["right_pad_tf"][2]) / 2
      bz = (b["left_pad_tf"][2] + b["right_pad_tf"][2]) / 2
      tz = tip_z(m, d)
      rows.append((prox, dist, bsep, ssep))
      print(
        f"{prox:8.4f} {dist:8.2f} {bsep:11.1f} {ssep:10.1f} "
        f"{sz * 1e3:9.1f} {bz * 1e3:9.1f} {tz * 1e3:8.1f} "
        f"{(tz - WORK_SURFACE_Z) * 1e3:9.1f}"
      )
  print(
    f"\naperture_mm() model says: q=0 -> {aperture_mm(0.0):.1f} mm, "
    f"q={FINGER_OPEN_ANGLE} -> {aperture_mm(FINGER_OPEN_ANGLE):.1f} mm"
  )

  # ------------------------------------------------------- 2. grip on the cube
  print()
  print("=" * 78)
  print("2. GRIP ON THE CUBE  (pre-grasp pose, close under the servo, settle)")
  print("=" * 78)
  best = None
  for close_cmd in (-0.10, -0.0754, -0.05, 0.0, 0.05):
    mujoco.mj_resetData(m, d)
    set_arm(m, d, TWOFINGER_ARM_PREGRASP)
    put_cube(m, d, [TABLE_X, 0.0, RESTING_Z])
    mujoco.mj_forward(m, d)
    for _ in range(400):  # settle at the pre-grasp
      mujoco.mj_step(m, d)
    for name, sgn in (("left_1_tf", +1), ("right_1_tf", -1)):
      d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = sgn * close_cmd
    for _ in range(600):  # close and settle against the cube
      mujoco.mj_step(m, d)
    s, b = sites(m, d), markers(m, d)
    ssep = np.linalg.norm(s["left_pad_tf"] - s["right_pad_tf"]) * 1e3
    bsep = np.linalg.norm(b["left_pad_tf"] - b["right_pad_tf"]) * 1e3
    q = d.qpos[
      m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_1_tf")]
    ]
    c = cube_pos(m, d)
    mid = (s["left_pad_tf"] + s["right_pad_tf"]) / 2
    print(
      f"  cmd q={close_cmd:+.4f} -> settled q={q:+.4f}  "
      f"site sep {ssep:6.2f} mm  marker sep {bsep:6.2f} mm"
    )
    print(
      f"      cube at {c * 1e3} mm, pad-mid - cube = "
      f"{(mid - c) * 1e3} mm, grasp_site - cube = "
      f"{(s['grasp_site_tf'] - c) * 1e3} mm"
    )
    if close_cmd == -0.0754:
      best = (ssep, bsep, q, mid - c)
  assert best is not None
  print(f"\n  PAD_SEP_AT_GRASP = {best[0] / 1e3:.4f}  ({best[0]:.2f} mm)")
  print(
    f"  each site is {(best[0] - 2 * CUBE_HALF * 1e3) / 2:.2f} mm "
    f"outboard of the cube face"
  )

  # ---------------------------------------------------- 3. the hover reset pose
  print()
  print("=" * 78)
  print("3. HOVER RESET POSE  (settle from HOME, cube at the spawn centre)")
  print("=" * 78)
  mujoco.mj_resetData(m, d)
  set_arm(m, d, TWOFINGER_ARM_HOME.joint_pos)
  put_cube(m, d, [TABLE_X, 0.0, RESTING_Z])
  for _ in range(800):
    mujoco.mj_step(m, d)
  s = sites(m, d)
  c = cube_pos(m, d)
  mid = (s["left_pad_tf"] + s["right_pad_tf"]) / 2
  ssep = np.linalg.norm(s["left_pad_tf"] - s["right_pad_tf"]) * 1e3
  print(f"  pad-site separation      {ssep:8.2f} mm")
  print(f"  pad midpoint - cube      {(mid - c) * 1e3} mm")
  print(f"  grasp_site - cube        {(s['grasp_site_tf'] - c) * 1e3} mm")
  print(f"  fingertip clears foam by {(tip_z(m, d) - WORK_SURFACE_Z) * 1e3:8.2f} mm")
  print(
    f"  cube settled at z        {c[2] * 1e3:8.2f} mm "
    f"(RESTING_Z = {RESTING_Z * 1e3:.1f})"
  )

  # ------------------------------------------------------------- 4. sigma rule
  print()
  print("=" * 78)
  print("4. ACTION REACH  (target - default) / scale;  >1.5 sigma is unreachable")
  print("=" * 78)
  hand_scale = 0.35
  arm_scale, wrist_scale = 0.40, 1.1
  grip_q = best[2]
  targets = [
    ("left_1_tf close-to-flush", open_angle_for(50.1), FINGER_OPEN_ANGLE, hand_scale),
    ("left_1_tf grip (settled)", grip_q, FINGER_OPEN_ANGLE, hand_scale),
    ("left_1_tf full close 40mm", open_angle_for(40.0), FINGER_OPEN_ANGLE, hand_scale),
    ("left_2_tf curl -0.3", -0.3, 0.0, hand_scale),
    ("left_2_tf curl -0.6", -0.6, 0.0, hand_scale),
  ]
  for name, tgt, dflt, sc in targets:
    print(
      f"  {name:28s} target {tgt:+.4f}  default {dflt:+.4f}  "
      f"scale {sc:.2f}  ->  {(tgt - dflt) / sc:+.2f} sigma"
    )
  for jn in ("joint2", "joint3", "joint4", "joint5", "joint6", "joint7"):
    h = TWOFINGER_ARM_HOME.joint_pos[jn]
    p = TWOFINGER_ARM_PREGRASP[jn]
    sc = wrist_scale if jn == "joint7" else arm_scale
    print(
      f"  {jn} hover->pregrasp        target {p:+.4f}  default {h:+.4f}  "
      f"scale {sc:.2f}  ->  {(p - h) / sc:+.2f} sigma"
    )


if __name__ == "__main__":
  main()
