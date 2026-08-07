"""Pinpoint the NaN source in the two-finger grasp env by forcing contact.

Each env gets a *sustained* random action vector (held every step), which drives
the arm/fingers into extreme configs -> ground/limit/cube contact -- the regime
the hover-jitter diagnostic never reached. On the first non-finite state we dump
the previous step's fastest joints and largest contact forces to localize it.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.config.flexiv_two_finger import env_cfgs as E

N = 512
DEV = "cuda"


def main():
  cfg = E.flexiv_two_finger_grasp_env_cfg()
  cfg.scene.num_envs = N
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
  env.reset()
  nact = env.action_manager.total_action_dim
  robot = env.scene["robot"]
  jnames = robot.joint_names
  print(f"num_envs={N} action_dim={nact} joints={jnames}")

  g = torch.Generator(device=DEV).manual_seed(0)
  # A spread of sustained biases: some gentle, some slammed to the rails.
  bias = torch.randn(N, nact, generator=g, device=DEV)
  bias *= torch.linspace(0.2, 3.0, N, device=DEV).unsqueeze(1)

  prev = {}
  for t in range(800):
    act = bias + 0.1 * torch.randn(N, nact, generator=g, device=DEV)
    obs, *_ = env.step(act)
    qvel = robot.data.joint_vel
    qpos = robot.data.joint_pos
    actor = obs["actor"] if isinstance(obs, dict) else obs

    finite = (
      torch.isfinite(qvel).all()
      and torch.isfinite(qpos).all()
      and torch.isfinite(actor).all()
    )
    if not finite:
      print(f"\n*** NaN at step {t} ***")
      badenv = (~torch.isfinite(qvel)).any(dim=1) | (~torch.isfinite(qpos)).any(dim=1)
      badenv |= (~torch.isfinite(actor)).any(dim=1)
      ei = int(torch.nonzero(badenv)[0].item())
      print(f"first bad env = {ei}, its sustained bias = {bias[ei].tolist()}")
      if "qvel" in prev:
        pv = prev["qvel"][ei]
        pp = prev["qpos"][ei]
        print("PREV-step joint state of that env:")
        for j, nm in enumerate(jnames):
          print(f"  {nm:8} pos={pp[j]:+.3f} vel={pv[j]:+8.2f}")
        print(f"PREV-step |qvel|max over all envs = {prev['qvmax']:.1f}")
      return
    prev = {
      "qvel": qvel.clone(),
      "qpos": qpos.clone(),
      "qvmax": qvel.abs().max().item(),
    }
    if t % 50 == 0:
      cube_v = env.scene["cube"].data.root_link_lin_vel_w.abs().max().item()
      print(
        f"step {t:4d}: |qvel|max={qvel.abs().max().item():8.1f} "
        f"|cube_v|max={cube_v:7.2f}"
      )
  print("survived 800 steps with sustained extreme actions -- no NaN")


if __name__ == "__main__":
  main()
