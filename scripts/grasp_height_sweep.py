"""Where can this hand actually grasp a 50 mm cube sitting on the table?

The arm is pinned kinematically, so this isolates the GRIPPER: at each approach
height (and optional distal curl) the jaws close on a cube resting on the table
and we measure the pad-vs-cube force, whether the fingertips are mashing the
table, how far the closing jaws shove the cube, and how much upward pull the
resulting grasp survives before the cube escapes.

Lifting is deliberately NOT tested here -- a kinematically prescribed arm cannot
carry anything (friction resists velocity; a teleport imposes position), so the
pull-out force is the honest measure of grasp strength.

  uv run python scripts/grasp_height_sweep.py [--render OUT.png]
"""

import sys

import mujoco
import numpy as np

XML = "/home/yiboc/compliance-grasping/exports/two_finger_grasp_scene/scene.xml"
TABLE, CH = 0.40, 0.025  # table top, cube half-extent
CX, CY = 0.4501, 0.0  # cube spawn (directly under the hover pose)
JAW_OPEN = 0.40

m = mujoco.MjModel.from_xml_path(XML)
m.opt.timestep = 0.005
m.opt.iterations, m.opt.ls_iterations = 20, 30
d = mujoco.MjData(m)

nid = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
J, A, MJ = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj
ARM = [f"robot/joint{i}" for i in range(1, 8)]
FING = ["robot/left_1", "robot/left_2", "robot/right_1", "robot/right_2"]
qadr = {n: m.jnt_qposadr[nid(J, n)] for n in ARM + FING}
aid = {n: nid(A, n) for n in ARM + FING}
ARM_Q = [qadr[n] for n in ARM]
ARM_V = [m.jnt_dofadr[nid(J, n)] for n in ARM]
sid = nid(MJ.mjOBJ_SITE, "robot/grasp_site")
cube_b = nid(MJ.mjOBJ_BODY, "cube/cube")
cube_q = m.jnt_qposadr[nid(J, "cube/cube_joint")]
cube_g = nid(MJ.mjOBJ_GEOM, "cube/cube_geom")
table_g = nid(MJ.mjOBJ_GEOM, "table/table_top")
gL, gR = nid(MJ.mjOBJ_GEOM, "robot/left_2_col"), nid(MJ.mjOBJ_GEOM, "robot/right_2_col")

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
HOVER = d.qpos[ARM_Q].copy()


def ik(target, q0):
  """6-DOF IK on the grasp site, holding the tool square to the cube faces."""
  q = q0.copy()
  for _ in range(600):
    d.qpos[ARM_Q] = q
    mujoco.mj_forward(m, d)
    p = d.site_xpos[sid].copy()
    rc = d.site_xmat[sid].reshape(3, 3).copy()
    qt, aa = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(qt, rc.T.flatten())
    mujoco.mju_quat2Vel(aa, qt, 1.0)
    e = np.concatenate([target - p, aa])
    if np.linalg.norm(e) < 1e-6:
      break
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    jac = np.vstack([jp[:, ARM_V], jr[:, ARM_V]])
    q = q + 0.4 * jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
  return q


def cforce(ga, gb_set):
  f = 0.0
  for i in range(d.ncon):
    p = {d.contact[i].geom1, d.contact[i].geom2}
    if ga in p and p & gb_set:
      fv = np.zeros(6)
      mujoco.mj_contactForce(m, d, i, fv)
      f += abs(fv[0])
  return f


def close_on_cube(site_z, close, distal):
  q = ik(np.array([CX, CY, site_z]), HOVER)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  d.qpos[ARM_Q] = q
  d.qpos[qadr["robot/left_1"]] = JAW_OPEN
  d.qpos[qadr["robot/right_1"]] = -JAW_OPEN
  d.qpos[cube_q : cube_q + 3] = [CX, CY, TABLE + CH]
  d.qpos[cube_q + 3 : cube_q + 7] = [1, 0, 0, 0]
  d.qvel[:] = 0
  mujoco.mj_forward(m, d)
  for i in range(700):
    d.qpos[ARM_Q], d.qvel[ARM_V] = q, 0
    for k, n in enumerate(ARM):
      d.ctrl[aid[n]] = q[k]
    t = min(1.0, (i + 1) / 350)
    a, b = JAW_OPEN + t * (close - JAW_OPEN), t * distal
    d.ctrl[aid["robot/left_1"]], d.ctrl[aid["robot/right_1"]] = a, -a
    d.ctrl[aid["robot/left_2"]], d.ctrl[aid["robot/right_2"]] = b, -b
    mujoco.mj_step(m, d)
  return q


def pull_out(q):
  """Ramp an upward force on the cube (weight already cancelled) until it slips."""
  z0 = d.xpos[cube_b][2]
  slip = None
  for i in range(4000):
    d.qpos[ARM_Q], d.qvel[ARM_V] = q, 0
    for k, n in enumerate(ARM):
      d.ctrl[aid[n]] = q[k]
    d.xfrc_applied[cube_b][2] = i * 0.004 + 0.05 * 9.81
    mujoco.mj_step(m, d)
    if abs(d.xpos[cube_b][2] - z0) > 0.02:
      slip = i * 0.004
      break
  d.xfrc_applied[cube_b][:] = 0
  return slip


CASES = [
  (TABLE + CH + dz, c, dst)
  for dz in (-0.005, 0.0, 0.008, 0.015, 0.025)
  for c, dst in ((0.05, 0.0), (0.00, -0.35), (0.00, -0.50))
]

render_to = None
if "--render" in sys.argv:
  render_to = sys.argv[sys.argv.index("--render") + 1]
  m.vis.global_.offheight, m.vis.global_.offwidth = 480, 640
  renderer = mujoco.Renderer(m, height=480, width=640)
  cam = mujoco.MjvCamera()
  cam.lookat[:] = [CX, CY, TABLE + CH]
  cam.distance, cam.elevation = 0.22, -8

print(f"cube centre z = {TABLE + CH:.3f}, table top {TABLE:.3f}, cube weight 0.49 N")
print(
  f"{'site_z':>7} {'rel':>7} {'close':>6} {'distal':>7} "
  f"{'gripF':>8} {'tip-tbl':>8} {'shove':>7} {'slip(N)':>8}"
)
rows = []
for z, c, dst in CASES:
  q = close_on_cube(z, c, dst)
  gf = cforce(cube_g, {gL, gR})
  tf = cforce(table_g, {gL, gR})
  shove = np.linalg.norm(d.xpos[cube_b][:2] - [CX, CY]) * 1000
  if render_to:
    tiles = []
    for az in (140, 90, 50):
      cam.azimuth = az
      renderer.update_scene(d, camera=cam)
      tiles.append(renderer.render())
    rows.append(np.concatenate(tiles, axis=1))
  sl = pull_out(q)
  ss = f"{sl:8.2f}" if sl is not None else "   >16.0"
  print(
    f"{z:7.3f} {(z - TABLE - CH) * 1000:+6.0f}m {c:6.2f} {dst:7.2f} "
    f"{gf:8.2f} {tf:8.2f} {shove:6.1f}m{ss}"
  )

if render_to:
  import imageio.v2 as imageio

  imageio.imwrite(render_to, np.concatenate(rows, axis=0))
  print(f"wrote {render_to}")
