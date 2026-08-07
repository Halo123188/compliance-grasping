"""Does the gripper occlude the cube in the D435 view?

Renders the camera image with the cube pinned at the corners of its spawn
region, at two arm heights: the 10 cm hover start pose, and descended so the
fingertips straddle the cube (the worst case for self-occlusion).
"""

import imageio.v2 as imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  CUBE_HALF,
  TABLE_H,
)
from mjlab.tasks.registry import load_env_cfg

OUT = "/work/yiboc"

# Corners of the trimmed cube spawn region (see CUBE_REGION in env_cfgs).
CUBE_XY = [
  ("near", 0.35, 0.0),
  ("center", 0.475, 0.0),
  ("far", 0.60, 0.0),
  ("near-left", 0.35, 0.20),
  ("near-right", 0.35, -0.20),
  ("far-left", 0.60, 0.20),
]

# Arm poses: the hover start, and an IK'd descent to grasp height at (0.45, 0).
POSES = {
  "hover": [-0.1104, -0.4672, 0.2889, 2.3750, -0.4049, 1.2378, 0.6380],
  "grasp": None,  # solved below
}

cfg = load_env_cfg("Mjlab-Grasp-TwoFinger-Flexiv", play=True)
cfg.scene.num_envs = 2
cfg.terminations = {}
# MuJoCo derives the horizontal FOV from fovy AND the viewport aspect, so this
# must match the policy camera's 128x72 (16:9) or the render shows a narrower
# cone than the D435 actually has.
cfg.viewer.width, cfg.viewer.height = 960, 540
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
env.reset()

robot = env.scene["robot"]
cube = env.scene["cube"]
r = env._offline_renderer
origin = env.scene.env_origins[0].cpu().numpy()


def solve_grasp_pose() -> list[float]:
  """IK the grasp_site down onto the cube at (0.45, 0)."""
  import mujoco

  from mjlab.asset_zoo.robots.two_finger_hand import constants as c

  m = c.get_spec().compile()
  d = mujoco.MjData(m)
  arm = [f"joint{i}" for i in range(1, 8)]
  jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm]
  qad = [m.jnt_qposadr[j] for j in jid]
  vad = [m.jnt_dofadr[j] for j in jid]
  lo = np.array([m.jnt_range[j][0] for j in jid])
  hi = np.array([m.jnt_range[j][1] for j in jid])
  sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
  q = np.array(POSES["hover"])

  def fk(q):
    d.qpos[qad] = q
    mujoco.mj_forward(m, d)
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

  _, r_ref = fk(q)
  target = np.array([0.45, 0.0, CUBE_HALF])  # grasp_site at cube mid-height
  for _ in range(300):
    p, rc = fk(q)
    qt = np.zeros(4)
    aa = np.zeros(3)
    mujoco.mju_mat2Quat(qt, (r_ref @ rc.T).flatten())
    mujoco.mju_quat2Vel(aa, qt, 1.0)
    err = np.concatenate([target - p, aa])
    if np.linalg.norm(err) < 1e-5:
      break
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    jac = np.vstack([jp[:, vad], jr[:, vad]])
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
    q = np.clip(q + 0.5 * dq, lo + 1e-3, hi - 1e-3)
  print("grasp pose grasp_site =", np.round(fk(q)[0], 4))
  return q.tolist()


POSES["grasp"] = solve_grasp_pose()

nj = len(robot.joint_names)
tiles: dict[str, list[np.ndarray]] = {}
for pose_name, arm_q in POSES.items():
  row = []
  for label, cx, cy in CUBE_XY:
    q = torch.zeros((env.num_envs, nj), device=env.device)
    q[:, :7] = torch.tensor(arm_q, device=env.device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))

    state = cube.data.default_root_state.clone()
    state[:, 0:3] = (
      torch.tensor([cx, cy, TABLE_H + CUBE_HALF], device=env.device)
      + env.scene.env_origins
    )
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state)
    env.sim.forward()

    r.update(env.sim.data, camera="robot/d435")
    row.append(r.render())
    print(f"{pose_name:6} cube@{label}")
  tiles[pose_name] = row

for pose_name, row in tiles.items():
  h = np.concatenate(
    [np.concatenate(row[:3], axis=1), np.concatenate(row[3:], axis=1)], axis=0
  )
  imageio.imwrite(f"{OUT}/occl_{pose_name}.png", h)
  print(f"wrote occl_{pose_name}.png  (order: {[c[0] for c in CUBE_XY]})")
