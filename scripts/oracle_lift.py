"""Oracle test: can a scripted, perfect grasp lift the cube at all?

The learned policies reliably close a firm two-finger grip and then never lift.
Before shaping the reward yet again, this checks whether the *mechanism* can do
it: descend onto the cube, close the fingers, raise the arm, and see whether the
cube comes with it. Runs in vanilla MuJoCo on the exported scene (CPU).
"""

import mujoco
import numpy as np

XML = "/home/yiboc/compliance-grasping/exports/two_finger_grasp_scene/scene.xml"
ARM = [f"robot/joint{i}" for i in range(1, 8)]
FINGERS = ["robot/left_1", "robot/left_2", "robot/right_1", "robot/right_2"]
TABLE_TOP, CUBE_HALF = 0.40, 0.015

m = mujoco.MjModel.from_xml_path(XML)
m.opt.timestep = 0.005
m.opt.iterations, m.opt.ls_iterations = 20, 30
d = mujoco.MjData(m)

jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM + FINGERS}
qadr = {n: m.jnt_qposadr[j] for n, j in jid.items()}
vadr = {n: m.jnt_dofadr[j] for n, j in jid.items()}
aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM + FINGERS}
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot/grasp_site")
cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")
gL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_2_col")
gR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/right_2_col")
cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube/cube_geom")

print("finger actuators:")
for n in FINGERS:
  a = aid[n]
  print(
    f"  {n:16} gainprm={m.actuator_gainprm[a][0]:7.2f} "
    f"biasprm={m.actuator_biasprm[a][:3]} ctrlrange={m.actuator_ctrlrange[a]} "
    f"forcerange={m.actuator_forcerange[a]}"
  )

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)  # xpos is stale until forward kinematics runs
HOVER = np.array([d.qpos[qadr[n]] for n in ARM])
CUBE_XY = d.xpos[cube_bid][:2].copy()
assert np.linalg.norm(CUBE_XY) > 0.1, f"cube position looks unset: {CUBE_XY}"


def ik(target_xyz, q0):
  """Solve arm joints so grasp_site hits target, keeping tool orientation."""
  q = q0.copy()
  for n, v in zip(ARM, q, strict=False):
    d.qpos[qadr[n]] = v
  mujoco.mj_forward(m, d)
  r_ref = d.site_xmat[sid].reshape(3, 3).copy()
  va = [vadr[n] for n in ARM]
  for _ in range(400):
    for n, v in zip(ARM, q, strict=False):
      d.qpos[qadr[n]] = v
    mujoco.mj_forward(m, d)
    p, rc = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
    qt, aa = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(qt, (r_ref @ rc.T).flatten())
    mujoco.mju_quat2Vel(aa, qt, 1.0)
    err = np.concatenate([target_xyz - p, aa])
    if np.linalg.norm(err) < 1e-5:
      break
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    jac = np.vstack([jp[:, va], jr[:, va]])
    q = q + 0.5 * jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
  return q


GRASP_Q = ik(np.array([CUBE_XY[0], CUBE_XY[1], TABLE_TOP + CUBE_HALF]), HOVER)


def drive(arm_q, close, steps):
  for n, v in zip(ARM, arm_q, strict=False):
    d.ctrl[aid[n]] = v
  # left/right proximal joints are mirrored; distal joints stay at 0.
  d.ctrl[aid["robot/left_1"]] = -close
  d.ctrl[aid["robot/right_1"]] = close
  for _ in range(steps):
    mujoco.mj_step(m, d)


def grip_force():
  """Total normal force between either fingertip and the cube."""
  tot = 0.0
  for i in range(d.ncon):
    c = d.contact[i]
    pair = {c.geom1, c.geom2}
    if cube_gid in pair and (gL in pair or gR in pair):
      f = np.zeros(6)
      mujoco.mj_contactForce(m, d, i, f)
      tot += abs(f[0])
  return tot


print(f"\ncube at {CUBE_XY.round(4)}, grasp pose IK'd; hover->grasp descent")
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = m.key_ctrl[0]
drive(GRASP_Q, 0.0, 600)
gap = np.linalg.norm(d.geom_xpos[gL] - d.geom_xpos[gR])
print(f"  at cube, fingers open: tip gap = {gap * 1000:.1f} mm (cube is 30 mm)")

print("\nclosing sweep (holding at the cube):")
best = None
for close in (0.10, 0.20, 0.30, 0.38, 0.42):
  drive(GRASP_Q, close, 300)
  gap = np.linalg.norm(d.geom_xpos[gL] - d.geom_xpos[gR])
  f = grip_force()
  print(f"  close={close:.2f} -> tip gap {gap * 1000:5.1f} mm, grip force {f:6.2f} N")
  if best is None or f > best[1]:
    best = (close, f)

print(f"\nbest closure = {best[0]:.2f} ({best[1]:.2f} N). Now lifting:")
for close in (best[0],):
  mujoco.mj_resetDataKeyframe(m, d, 0)
  d.ctrl[:] = m.key_ctrl[0]
  drive(GRASP_Q, 0.0, 600)  # descend, open
  drive(GRASP_Q, close, 400)  # close on the cube
  z_grasped = d.xpos[cube_bid][2]
  drive(HOVER, close, 1200)  # raise back to the hover pose
  z_lifted = d.xpos[cube_bid][2]
  ee = d.site_xpos[sid][2]
  print(
    f"  cube z after close = {z_grasped:.4f}  (resting {TABLE_TOP + CUBE_HALF:.4f})"
  )
  print(f"  cube z after raise = {z_lifted:.4f}   grasp_site z = {ee:.4f}")
  print(f"  -> cube lifted {(z_lifted - TABLE_TOP - CUBE_HALF) * 1000:+.1f} mm")
  held = z_lifted > TABLE_TOP + CUBE_HALF + 0.02
  print(
    f"  -> {'HELD: the mechanism CAN lift' if held else 'DROPPED: cube slipped out'}"
  )
