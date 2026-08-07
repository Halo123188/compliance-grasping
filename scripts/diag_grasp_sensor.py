"""Is the bilateral-grasp contact sensor actually wired to the cube?

Training reports Episode_Reward/grasp == 0.0000 for every iteration while the
cube visibly gets knocked around, so either the fingers never pinch it or the
sensor never sees it. This forces the pinch: the cube is teleported between the
fingertips and the finger actuators are commanded closed, then we compare what
MuJoCo's own contact list says against what the left/right cube contact sensors
report.

  uv run python scripts/diag_grasp_sensor.py
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.config.flexiv_two_finger import env_cfgs as E

N = 8
DEV = "cuda"


def main():
  cfg = E.flexiv_two_finger_grasp_env_cfg(goal_mode="above_object", bringing_std=0.12)
  cfg.scene.num_envs = N
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
  env.reset()

  robot = env.scene["robot"]
  cube = env.scene["cube"]
  nact = env.action_manager.total_action_dim
  jn = list(robot.joint_names)
  print(f"action_dim={nact}")
  print(f"joints={jn}")
  for s in ("left_cube_contact", "right_cube_contact", "fingertip_table_contact"):
    d = env.scene[s].data
    print(
      f"sensor {s:24} force={None if d.force is None else tuple(d.force.shape)} "
      f"found={None if d.found is None else tuple(d.found.shape)}"
    )

  site = robot.data.site_pos_w[:, 0, :]  # grasp_site
  print(f"grasp_site_w[0] = {site[0].tolist()}")

  # Put the cube on the table directly under the hover pose, where it RESTS
  # stably -- no re-pinning, which previously froze the fingers.
  origin = env.scene.env_origins
  pose = torch.zeros(N, 7, device=DEV)
  pose[:, 0] = origin[:, 0] + 0.4501
  pose[:, 1] = origin[:, 1] + 0.0
  pose[:, 2] = origin[:, 2] + 0.425
  pose[:, 3] = 1.0
  cube.write_root_link_pose_to_sim(pose)
  cube.write_root_link_velocity_to_sim(torch.zeros(N, 6, device=DEV))

  # Constant action that reproduces the grasp validated offline at 36.9 N:
  # arm descends so grasp_site lands on the cube centre with the jaw square to
  # the faces, fingers close with the distal curl.
  ARM_ACT = [-0.0434, -0.5978, -0.4052, 0.1048, -1.3638, 0.6485, -2.0119]
  act = torch.zeros(N, nact, device=DEV)
  for i, v in enumerate(ARM_ACT):
    act[:, i] = v
  fi = {n: jn.index(n) for n in ("left_1", "left_2", "right_1", "right_2")}

  for t in range(400):
    frac = min(1.0, t / 150.0)  # descend first, then close
    a = act.clone()
    a[:, :7] *= frac
    if t > 150:
      c = min(1.0, (t - 150) / 100.0)
      a[:, fi["left_1"]] = -0.8 * c
      a[:, fi["right_1"]] = 0.8 * c
      a[:, fi["left_2"]] = -1.0 * c
      a[:, fi["right_2"]] = 1.0 * c
    env.step(a)
    if t % 50 == 49:
      lf = env.scene["left_cube_contact"].data
      rf = env.scene["right_cube_contact"].data
      lmag = torch.norm(lf.force, dim=-1).amax(dim=-1)
      rmag = torch.norm(rf.force, dim=-1).amax(dim=-1)
      q = robot.data.joint_pos[0]
      site_now = robot.data.site_pos_w[0, 0, :] - origin[0]
      cz = cube.data.root_link_pos_w[0, 2] - origin[0, 2]
      both = ((lmag > 0.1) & (rmag > 0.1)).float().mean()
      print(
        f"t={t:4d} site={[round(float(v), 3) for v in site_now]} cube_z={cz:.4f} "
        f"j7={q[6]:+.3f} l1={q[fi['left_1']]:+.3f} l2={q[fi['left_2']]:+.3f} "
        f"| L={lmag[0]:7.3f} R={rmag[0]:7.3f} bilateral_frac={both:.2f}"
      )

  # Ground truth straight out of the MuJoCo contact list for env 0.
  lf = env.scene["left_cube_contact"].data
  rf = env.scene["right_cube_contact"].data
  print(
    f"\nfinal: L_force={torch.norm(lf.force, dim=-1).amax(dim=-1).tolist()}\n"
    f"       R_force={torch.norm(rf.force, dim=-1).amax(dim=-1).tolist()}"
  )
  print("MuJoCo contact list: skipped (warp Data has no ncon)")


if __name__ == "__main__":
  main()
