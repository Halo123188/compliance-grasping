"""Turntable view of the gripper's pad sites, and where contact ACTUALLY lands.

Two jobs:

  1. Render the hand alone (no cube, no table) rotating about Z, so the pad
     sites can be checked against the real fingertip geometry from every angle.
  2. Derive the pad site position from physics instead of by eye: close the jaws
     on the cube and average the contact points MuJoCo reports on each fingertip,
     expressed in that fingertip link's own frame. That is the definition of
     "where this finger touches things", so it beats reading a centroid off the
     mesh, which depends on an arbitrary threshold for what counts as the face.

  uv run python scripts/show_pad_sites.py
"""

import imageio.v2 as imageio
import mujoco
import numpy as np

XML = "/home/yiboc/compliance-grasping/exports/two_finger_grasp_scene/scene.xml"
OUT = "/work/yiboc/pad_sites.png"

m = mujoco.MjModel.from_xml_path(XML)
m.opt.timestep = 0.005
m.opt.iterations, m.opt.ls_iterations = 20, 30
d = mujoco.MjData(m)
nid = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
MJ = mujoco.mjtObj
qa = {
  n: m.jnt_qposadr[nid(MJ.mjOBJ_JOINT, n)]
  for n in ("robot/left_1", "robot/right_1", "robot/left_2", "robot/right_2")
}
cube_b = nid(MJ.mjOBJ_BODY, "cube/cube")
cube_q = m.jnt_qposadr[nid(MJ.mjOBJ_JOINT, "cube/cube_joint")]
sL, sR = nid(MJ.mjOBJ_SITE, "robot/left_pad"), nid(MJ.mjOBJ_SITE, "robot/right_pad")
sG = nid(MJ.mjOBJ_SITE, "robot/grasp_site")
gL, gR = nid(MJ.mjOBJ_GEOM, "robot/left_2_col"), nid(MJ.mjOBJ_GEOM, "robot/right_2_col")
cube_g = nid(MJ.mjOBJ_GEOM, "cube/cube_geom")
bL, bR = nid(MJ.mjOBJ_BODY, "robot/left_2"), nid(MJ.mjOBJ_BODY, "robot/right_2")

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

# ---- 1. Contact-derived pad location ---------------------------------------
ARM = [f"robot/joint{i}" for i in range(1, 8)]
ARM_Q = [m.jnt_qposadr[nid(MJ.mjOBJ_JOINT, n)] for n in ARM]
ARM_V = [m.jnt_dofadr[nid(MJ.mjOBJ_JOINT, n)] for n in ARM]
aid = {n: nid(MJ.mjOBJ_ACTUATOR, n) for n in ARM + list(qa)}
HOVER = d.qpos[ARM_Q].copy()

hits = {gL: [], gR: []}
d.qpos[cube_q : cube_q + 3] = (d.site_xpos[sL] + d.site_xpos[sR]) / 2
d.qpos[cube_q + 3 : cube_q + 7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)
for i in range(900):
  d.qpos[ARM_Q], d.qvel[ARM_V] = HOVER, 0
  for k, n in enumerate(ARM):
    d.ctrl[aid[n]] = HOVER[k]
  t = min(1.0, i / 400)
  a = 0.70 + t * (0.00 - 0.70)
  d.ctrl[aid["robot/left_1"]], d.ctrl[aid["robot/right_1"]] = a, -a
  d.ctrl[aid["robot/left_2"]], d.ctrl[aid["robot/right_2"]] = -0.5 * t, 0.5 * t
  d.xfrc_applied[cube_b][2] = 0.05 * 9.81  # hold it up until the jaws bite
  mujoco.mj_step(m, d)
  for c in range(d.ncon):
    con = d.contact[c]
    pr = {con.geom1, con.geom2}
    if cube_g not in pr:
      continue
    for g, b in ((gL, bL), (gR, bR)):
      if g in pr:
        hits[g].append(d.xmat[b].reshape(3, 3).T @ (con.pos - d.xpos[b]))
d.xfrc_applied[:] = 0

print("contact-derived pad centre, in the fingertip LINK frame:")
for g, name, site in ((gL, "left_2 ", sL), (gR, "right_2", sR)):
  if not hits[g]:
    print(f"  {name}: no contacts recorded")
    continue
  P = np.array(hits[g])
  c, sd = P.mean(0), P.std(0)
  cur = m.site_pos[site]
  print(
    f"  {name}: n={len(P):4d}  contact centre = "
    f"({c[0]:+.5f}, {c[1]:+.5f}, {c[2]:+.5f})   "
    f"spread 1sd = ({sd[0] * 1000:.1f}, {sd[1] * 1000:.1f}, {sd[2] * 1000:.1f}) mm"
  )
  print(
    f"           current site   = ({cur[0]:+.5f}, {cur[1]:+.5f}, {cur[2]:+.5f})"
    f"   off by {np.linalg.norm(c - cur) * 1000:.1f} mm"
    f"  (dz {(c[2] - cur[2]) * 1000:+.1f} mm)"
  )

# ---- 2. Turntable render, hand only -----------------------------------------
for g in range(m.ngeom):
  name = mujoco.mj_id2name(m, MJ.mjOBJ_GEOM, g) or ""
  if not name.startswith("robot/"):
    m.geom_rgba[g] = [0, 0, 0, 0]  # hide cube, table, terrain
for _s in range(m.nsite):
  m.site_rgba[_s] = [0, 0, 0, 0]
for s, rgba in ((sL, [0.15, 0.55, 1.0, 1.0]), (sR, [1.0, 0.2, 0.2, 1.0])):
  m.site_rgba[s] = rgba
  m.site_size[s] = [0.0035, 0, 0]
m.site_rgba[sG] = [0.45, 0.45, 0.45, 0.8]
m.site_size[sG] = [0.003, 0, 0]
for g in range(m.ngeom):
  if m.geom_rgba[g][3] == 0:
    continue
  if m.geom_group[g] == 1:
    m.geom_rgba[g] = [0.70, 0.73, 0.78, 0.32]
  elif m.geom_group[g] == 3:
    m.geom_rgba[g] = [0.95, 0.42, 0.22, 0.22]

opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = 1
opt.geomgroup[3] = 1
opt.sitegroup[:] = 1

m.vis.global_.offheight, m.vis.global_.offwidth = 520, 520
r = mujoco.Renderer(m, height=520, width=520)
cam = mujoco.MjvCamera()

mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[qa["robot/left_1"]], d.qpos[qa["robot/right_1"]] = 0.10, -0.10
d.qpos[qa["robot/left_2"]] = d.qpos[qa["robot/right_2"]] = 0.0
mujoco.mj_forward(m, d)
cam.lookat[:] = (d.site_xpos[sL] + d.site_xpos[sR]) / 2
cam.distance, cam.elevation = 0.155, -4

tiles = ([], [])
for k, az in enumerate(range(0, 360, 45)):
  cam.azimuth = az
  r.update_scene(d, camera=cam, scene_option=opt)
  tiles[k // 4].append(r.render())

imageio.imwrite(OUT, np.concatenate([np.concatenate(t, axis=1) for t in tiles], axis=0))
print(f"\nwrote {OUT}  (turntable about Z, 45 deg steps, jaws at grip 0.10)")
