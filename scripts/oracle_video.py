"""Scripted (oracle) grasp of the 50 mm cube, rendered to video.

Descend with the jaws open, close on the cube, lift. The arm is driven
kinematically so the demo tests the *gripper*, not the arm's tracking. Finger
joints are dynamic (position actuators), so the grasp itself is real physics.

  uv run python scripts/oracle_video.py OUT.mp4
"""

import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

XML = "/home/yiboc/compliance-grasping/exports/two_finger_grasp_scene/scene.xml"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/yiboc/oracle_grasp.mp4"
TABLE, CH = 0.40, 0.025  # table top, cube half-extent (50 mm cube)
JAW_OPEN, JAW_GRIP = 0.60, 0.00  # left_1: 72.8 mm open; 0.10 -> 40 mm nominal,
# i.e. ~10 mm of interference on the 50 mm cube so the fingers actually squeeze
# (at 0.28 the nominal gap is 49.5 mm -- 0.5 mm of interference, no grip at all).
FPS, EVERY = 30, 6  # render every 6th step (dt=0.005 -> 33 Hz)

m = mujoco.MjModel.from_xml_path(XML)
m.opt.timestep = 0.005
m.opt.iterations, m.opt.ls_iterations = 20, 30
d = mujoco.MjData(m)

ARM = [f"robot/joint{i}" for i in range(1, 8)]
name2id = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
J, A, OBJ = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj
qadr = {
  n: m.jnt_qposadr[name2id(J, n)] for n in ARM + ["robot/left_1", "robot/right_1"]
}
aid = {n: name2id(A, n) for n in ARM + ["robot/left_1", "robot/right_1"]}
ARM_Q = [qadr[n] for n in ARM]
ARM_V = [m.jnt_dofadr[name2id(J, n)] for n in ARM]
sid = name2id(OBJ.mjOBJ_SITE, "robot/grasp_site")
cube_b = name2id(OBJ.mjOBJ_BODY, "cube/cube")
cube_q = m.jnt_qposadr[name2id(J, "cube/cube_joint")]
gL, gR = (
  name2id(OBJ.mjOBJ_GEOM, "robot/left_2_col"),
  name2id(OBJ.mjOBJ_GEOM, "robot/right_2_col"),
)
cube_g = name2id(OBJ.mjOBJ_GEOM, "cube/cube_geom")

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
CXY = d.xpos[cube_b][:2].copy()
HOVER = np.array([d.qpos[qadr[n]] for n in ARM])


def ik(target, q0, R_target=None):
  """6-DOF IK on the grasp site.

  R_target aligns the *jaw* as well as the position. The jaw closes along the
  site's local +X (measured: closing axis is [1,0,0] in the palm frame), so
  R_target = identity puts the jaw perpendicular to the cube's yaw=0 faces.
  Without this the wrist sat ~45 deg off and the pads met the cube's corner
  edge-on instead of flat on a face.
  """
  q = q0.copy()
  for n, v in zip(ARM, q, strict=False):
    d.qpos[qadr[n]] = v
  mujoco.mj_forward(m, d)
  ref = np.eye(3) if R_target is None else R_target
  for _ in range(600):
    for n, v in zip(ARM, q, strict=False):
      d.qpos[qadr[n]] = v
    mujoco.mj_forward(m, d)
    p, rc = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
    qt, aa = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(qt, (ref @ rc.T).flatten())
    mujoco.mju_quat2Vel(aa, qt, 1.0)
    e = np.concatenate([target - p, aa])
    if np.linalg.norm(e) < 1e-6:
      break
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    jac = np.vstack([jp[:, ARM_V], jr[:, ARM_V]])
    q = q + 0.4 * jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
  return q


# Demo-only: stiffen the arm servos ~25x. The default Rizon position gains sag
# several cm under gravity, and the alternative -- prescribing arm qpos directly
# -- cannot work here: teleporting positions imposes motion that friction (which
# only resists *velocity*) can never oppose, so a kinematic arm slides through
# the contact and leaves the object behind no matter how hard the jaws squeeze.
# The GRASP is untouched real physics; only the arm's tracking authority changes.
for _n in ARM:
  _a = aid[_n]
  m.actuator_gainprm[_a][0] *= 25.0
  m.actuator_biasprm[_a][1] *= 25.0
  m.actuator_biasprm[_a][2] *= 5.0
  m.actuator_forcerange[_a] *= 20.0

CUBE_C = TABLE + CH
ABOVE = ik(np.array([CXY[0], CXY[1], CUBE_C + 0.08]), HOVER)
AT = ik(np.array([CXY[0], CXY[1], CUBE_C]), ABOVE)
UP = ik(np.array([CXY[0], CXY[1], CUBE_C + 0.15]), ABOVE)

for n, v in zip(ARM, AT, strict=False):
  d.qpos[qadr[n]] = v
mujoco.mj_forward(m, d)
jaw_w = d.geom_xpos[gR] - d.geom_xpos[gL]
jaw_w /= np.linalg.norm(jaw_w)
print(
  f"jaw axis in WORLD after alignment = {jaw_w.round(4)}  "
  f"({np.degrees(np.arctan2(jaw_w[1], jaw_w[0])):+.1f} deg from +X; "
  f"0 or 90 means flat on a cube face)"
)
print(f"grasp_site at AT = {d.site_xpos[sid].round(4)} (cube centre z={CUBE_C:.3f})")

mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[ARM_Q] = ABOVE
d.qpos[qadr["robot/left_1"]], d.qpos[qadr["robot/right_1"]] = JAW_OPEN, -JAW_OPEN
d.qpos[cube_q : cube_q + 3] = [CXY[0], CXY[1], CUBE_C]
d.qpos[cube_q + 3 : cube_q + 7] = [1, 0, 0, 0]
d.qvel[:] = 0
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, height=540, width=960)
cam = mujoco.MjvCamera()
cam.lookat[:] = [CXY[0], CXY[1], CUBE_C + 0.05]
cam.distance, cam.elevation, cam.azimuth = 0.75, -18, 140
frames, log = [], []


def cart_path(z0, z1, k=24):
  """IK a straight vertical Cartesian path, so tool ORIENTATION is held fixed.

  Interpolating in joint space between two IK'd endpoints does not preserve
  orientation -- the gripper tilts mid-stroke and shears the cube out of the
  pads. Solving IK at each waypoint keeps the jaw flat on the cube face the
  whole way.
  """
  poses, seed = [], ABOVE
  for z in np.linspace(z0, z1, k):
    seed = ik(np.array([CXY[0], CXY[1], z]), seed)
    poses.append(seed.copy())
  return poses


def phase(poses, ja, jb, steps_per, tag, kin=True):
  """kin=True prescribes arm qpos -- millimetre-accurate positioning, but such a
  body cannot carry anything: friction resists velocity, and a teleport imposes
  position, so the pads slide through the contact. Use it to place and close the
  jaws, then hand off to the (stiffened) servos with kin=False for the lift,
  where real force transmission is what matters.
  """
  if not kin:  # bumpless transfer: hold where we actually are
    for _k, nm in enumerate(ARM):
      d.ctrl[aid[nm]] = d.qpos[qadr[nm]]
  n = len(poses)
  for j in range(n - 1):
    qa, qb = poses[j], poses[j + 1]
    for i in range(steps_per):
      t = (i + 1) / steps_per
      q = qa + t * (qb - qa)
      if kin:
        d.qpos[ARM_Q] = q
        d.qvel[ARM_V] = (qb - qa) / (steps_per * m.opt.timestep)
      for k, nm in enumerate(ARM):
        d.ctrl[aid[nm]] = q[k]
      frac = (j * steps_per + i + 1) / ((n - 1) * steps_per)
      jaw = ja + frac * (jb - ja)
      d.ctrl[aid["robot/left_1"]], d.ctrl[aid["robot/right_1"]] = jaw, -jaw
      mujoco.mj_step(m, d)
      if (j * steps_per + i) % EVERY == 0:
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
  f = 0.0
  for j in range(d.ncon):
    pr = {d.contact[j].geom1, d.contact[j].geom2}
    if cube_g in pr and (gL in pr or gR in pr):
      fv = np.zeros(6)
      mujoco.mj_contactForce(m, d, j, fv)
      f += abs(fv[0])
  log.append(
    f"{tag:10} cube_z={d.xpos[cube_b][2]:.4f}  site_z={d.site_xpos[sid][2]:.4f}  "
    f"gripF={f:6.2f} N  left_1={d.qpos[qadr['robot/left_1']]:+.3f}"
  )


HOLD = [AT, AT]
phase([ABOVE, ABOVE], JAW_OPEN, JAW_OPEN, 150, "settle")
phase(cart_path(CUBE_C + 0.08, CUBE_C), JAW_OPEN, JAW_OPEN, 30, "descend")
phase(HOLD, JAW_OPEN, JAW_GRIP, 500, "close")
phase(HOLD, JAW_GRIP, JAW_GRIP, 250, "grip")
phase(cart_path(CUBE_C, CUBE_C + 0.15), JAW_GRIP, JAW_GRIP, 60, "LIFT", kin=False)
phase([UP, UP], JAW_GRIP, JAW_GRIP, 500, "hold", kin=False)

print("\n".join(log))
lifted = (d.xpos[cube_b][2] - CUBE_C) * 1000
print(
  f"\ncube lifted {lifted:+.1f} mm  -> "
  f"{'SUCCESS: oracle grasp holds' if lifted > 50 else 'FAILED'}"
)
imageio.mimwrite(OUT, frames, fps=FPS, quality=8)
print(f"wrote {len(frames)} frames -> {OUT}")
