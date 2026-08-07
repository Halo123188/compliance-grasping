"""Reusable tabletop grasping scene: Flexiv + two-finger hand bolted to a table,
a cube to grasp, and a wrist-adjacent Intel RealSense D435 RGB-D camera.

Shared by both grasp tasks (state-based and vision-based). Build the scene with
``build_scene()``; place / randomise the cube via ``set_cube``.

Real-world dims and sensor:
  * Table: 32 in (width, y) x 48 in (length, x). The arm sits at the back edge;
    the workspace extends forward (+x). The arm cannot reach the whole 48 in, so
    cube placement is limited to CUBE_REGION (a reachable rectangle).
  * Camera: Intel RealSense D435, mounted right in front of the arm base, ~18 cm
    high, pitched 45 deg down, looking forward over the workspace.
    D435 (from Intel product spec):
      depth FOV 87 x 58 deg, depth res up to 1280x720, min-Z ~0.105 m, range ~0.3-3 m
      RGB   FOV 69.4 x 42.5 deg, RGB res up to 1920x1080
    We model one aligned RGB-D camera at the depth vertical FOV (58 deg); at 16:9
    that gives ~87 deg horizontal, matching the depth module.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand import constants as C

IN = 0.0254  # metres per inch

# ── Table (top at z=0) ───────────────────────────────────────────────────────
TABLE_LEN_X = 48 * IN  # 1.219 m, front-back (arm reaches along +x)
TABLE_WID_Y = 32 * IN  # 0.813 m, left-right
TABLE_THICK = 0.2
TABLE_BACK_X = -0.10  # back edge sits 10 cm behind the arm base at x=0

# ── Reachable cube-placement region on the table (Flexiv 4S reach ~0.9 m) ─────
CUBE_REGION = dict(x=(0.32, 0.60), y=(-0.28, 0.28))  # metres, on the table top
CUBE_HALF = 0.022  # 44 mm cube

# ── Task spec shared by both grasp envs ──────────────────────────────────────
# Success = lift the cube >= LIFT_HEIGHT above the table and hold for HOLD_S.
# Action = 6-DOF Cartesian EE delta (position + orientation, resolved by IK) plus
# the gripper's four finger joints driven directly.
LIFT_HEIGHT = 0.10  # m above the table top (z=0)
HOLD_S = 1.0  # s the cube must stay above LIFT_HEIGHT to count as success

# ── D435 camera ──────────────────────────────────────────────────────────────
D435 = dict(
  pos=(0.09, 0.0, 0.18),  # hard up against the arm base, 18 cm high
  pitch_deg=45.0,  # looking forward and 45 deg down
  depth_fovy=58.0,  # vertical FOV of the depth module
  rgb_fovy=42.5,  # vertical FOV of the RGB sensor
  res=(1280, 720),  # native depth resolution (16:9)
  min_z=0.105,  # min depth distance (m); closer reads as invalid
  max_z=3.0,  # usable range (m)
)


def _cam_quat(pitch_deg: float) -> list[float]:
  """Quat for a camera in front of the arm looking +x and ``pitch`` deg down."""
  p = np.radians(pitch_deg)
  # view dir forward-down = (cos p, 0, -sin p); camera looks along -z.
  fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
  right = np.array([0.0, -1.0, 0.0])
  up = np.cross(right, fwd)
  z_out = -fwd
  R = np.array([right, up, z_out]).T.flatten()  # columns = cam axes
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, R)
  return q.tolist()


def build_scene(cube_pos: tuple[float, float, float] | None = None) -> mujoco.MjSpec:
  """Flexiv + two-finger hand on a 32x48 in table, a cube, and a D435 camera."""
  spec = C.get_spec()
  spec.visual.global_.offwidth = max(1280, D435["res"][0])
  spec.visual.global_.offheight = max(960, D435["res"][1])
  wb = spec.worldbody

  # table
  cx = TABLE_BACK_X + TABLE_LEN_X / 2
  t = wb.add_body(name="table", pos=[cx, 0, -TABLE_THICK])
  g = t.add_geom()
  g.type = mujoco.mjtGeom.mjGEOM_BOX
  g.size = [TABLE_LEN_X / 2, TABLE_WID_Y / 2, TABLE_THICK]
  g.rgba = [0.55, 0.45, 0.35, 1.0]

  # ground under the table
  fg = wb.add_geom()
  fg.type = mujoco.mjtGeom.mjGEOM_PLANE
  fg.size = [4, 4, 0.1]
  fg.pos = [0, 0, -2 * TABLE_THICK]
  fg.rgba = [0.2, 0.21, 0.23, 1]

  # cube (free body) on the table
  cx0, cy0 = 0.45, 0.0
  if cube_pos is not None:
    cx0, cy0 = cube_pos[0], cube_pos[1]
  cb = wb.add_body(name="cube", pos=[cx0, cy0, CUBE_HALF])
  cb.add_freejoint()
  cg = cb.add_geom()
  cg.type = mujoco.mjtGeom.mjGEOM_BOX
  cg.size = [CUBE_HALF, CUBE_HALF, CUBE_HALF]
  cg.rgba = [0.15, 0.7, 0.2, 1]
  cg.mass = 0.05

  # D435 camera (one aligned RGB-D cam at the depth vertical FOV) + a visible body
  q = _cam_quat(D435["pitch_deg"])
  cam = wb.add_camera()
  cam.name = "d435"
  cam.pos = list(D435["pos"])
  cam.quat = q
  cam.fovy = D435["depth_fovy"]
  camb = wb.add_body(name="d435_body", pos=list(D435["pos"]), quat=q)
  gc = camb.add_geom()
  gc.type = mujoco.mjtGeom.mjGEOM_BOX
  gc.size = [0.045, 0.0125, 0.0125]  # D435 is ~90x25x25 mm
  gc.rgba = [0.08, 0.08, 0.09, 1]
  gl = camb.add_geom()
  gl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
  gl.size = [0.008, 0.006, 0]
  gl.pos = [0, 0, -0.0135]
  gl.rgba = [0.1, 0.15, 0.28, 1]

  lt = wb.add_light(pos=[0.3, -0.8, 1.5])
  lt.dir = [-0.1, 0.4, -1]
  return spec


def ik_hover(m, d, target, n=300):
  """Pose the 7 arm joints so grasp_site reaches ``target`` pointing down."""
  ja = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k): k for k in range(m.njnt)}
  arm = [f"joint{i}" for i in range(1, 8)]
  dof = [m.jnt_dofadr[ja[n]] for n in arm]
  qad = [m.jnt_qposadr[ja[n]] for n in arm]
  gs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
  bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  d.qpos[m.jnt_qposadr[ja["joint2"]]] = 0.5
  d.qpos[m.jnt_qposadr[ja["joint4"]]] = 1.3
  for _ in range(n):
    mujoco.mj_forward(m, d)
    ep = np.asarray(target) - d.site_xpos[gs]
    R = d.xmat[bl].reshape(3, 3)
    eo = np.cross(-R[:, 2], [0, 0, -1.0])
    if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(eo) < 2e-2:
      break
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, gs)
    J = np.concatenate([jp[:, dof], jr[:, dof]], 0)
    dq = J.T @ np.linalg.solve(
      J @ J.T + 1e-3 * np.eye(6), np.concatenate([ep, 0.5 * eo])
    )
    for k, qa in enumerate(qad):
      lo, hi = m.jnt_range[ja[arm[k]]]
      d.qpos[qa] = np.clip(d.qpos[qa] + 0.5 * dq[k], lo, hi)
  mujoco.mj_forward(m, d)
