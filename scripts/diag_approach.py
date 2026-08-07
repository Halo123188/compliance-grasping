"""Does the trained policy fail to APPROACH, or approach and miss with the pads?

Both look identical in the logs (grasp reward 0.0000, cube untouched), but they
need opposite fixes -- reward shaping vs. gripper geometry. This rolls out a
checkpoint and reports, per episode, the closest the grasp_site and each finger
pad ever got to the cube, plus the jaw opening at that moment.

  uv run python scripts/diag_approach.py TASK CKPT
"""

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, STEPS, DEV = 64, 400, "cuda:0"

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device=DEV)

agent_cfg = load_rl_cfg(TASK)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=DEV)
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

robot, cube = env.scene["robot"], env.scene["cube"]
jn = list(robot.joint_names)
li, ri = jn.index("left_1"), jn.index("right_1")
l2i, r2i = jn.index("left_2"), jn.index("right_2")
snames = list(robot.site_names)
si = snames.index("grasp_site")
print(f"sites={snames}  grasp_site index={si}")
gnames = list(robot.geom_names)
gL = gnames.index("left_2_col")
gR = gnames.index("right_2_col")

obs = wrapped.reset()[0]
best_site = torch.full((N,), 1e9, device=DEV)
best_pad = torch.full((N,), 1e9, device=DEV)
jaw_at_best = torch.zeros(N, device=DEV)
best_mid = torch.full((N,), 1e9, device=DEV)
sep_at_best = torch.zeros(N, device=DEV)
dz_at_best = torch.zeros(N, device=DEV)
dist_at_best = torch.zeros(N, device=DEV)
for _ in range(STEPS):
  with torch.inference_mode():
    act = policy(obs)
  obs, _, _, _ = wrapped.step(act)
  c = cube.data.root_link_pos_w
  site = robot.data.site_pos_w[:, si, :]
  pl = robot.data.geom_pos_w[:, gL, :]
  pr = robot.data.geom_pos_w[:, gR, :]
  ds = torch.norm(site - c, dim=-1)
  dp = torch.minimum(torch.norm(pl - c, dim=-1), torch.norm(pr - c, dim=-1))
  best_site = torch.minimum(best_site, ds)
  upd = dp < best_pad
  jaw_at_best = torch.where(upd, robot.data.joint_pos[:, li], jaw_at_best)
  dist_at_best = torch.where(upd, ds, dist_at_best)
  best_pad = torch.minimum(best_pad, dp)
  sep_now = torch.norm((pl - pr)[:, :2], dim=-1)
  hz = ((pl + pr) / 2 - c)[:, 2]
  upd2 = torch.norm(((pl + pr) / 2 - c)[:, :2], dim=-1) < best_mid
  best_mid = torch.minimum(best_mid, torch.norm(((pl + pr) / 2 - c)[:, :2], dim=-1))
  sep_at_best = torch.where(upd2, sep_now, sep_at_best)
  dz_at_best = torch.where(upd2, hz, dz_at_best)

q = robot.data.joint_pos
print(f"\ncheckpoint: {CKPT}")
print(f"envs={N} steps={STEPS}   (cube half-extent 25 mm)")
print(
  f"closest grasp_site->cube : median {best_site.median() * 1000:6.1f} mm   "
  f"min {best_site.min() * 1000:6.1f} mm   max {best_site.max() * 1000:6.1f} mm"
)
print(
  f"closest  finger pad->cube: median {best_pad.median() * 1000:6.1f} mm   "
  f"min {best_pad.min() * 1000:6.1f} mm   max {best_pad.max() * 1000:6.1f} mm"
)
print(
  f"jaw angle left_1 at the closest-pad moment: "
  f"median {jaw_at_best.median():+.3f}  (hover default +0.70, grips at ~+0.10)"
)
print(
  f"final joint angles (median): left_1={q[:, li].median():+.3f} "
  f"left_2={q[:, l2i].median():+.3f} right_1={q[:, ri].median():+.3f} "
  f"right_2={q[:, r2i].median():+.3f}"
)
print(
  f"\nat the moment the pad MIDPOINT is horizontally closest to the cube:"
  f"\n  horizontal midpoint error : median {best_mid.median() * 1000:6.1f} mm"
  f"\n  pad separation            : median {sep_at_best.median() * 1000:6.1f} mm"
  f"   (a real grasp needs ~50 mm; ~0 means closed in mid-air)"
  f"\n  pad midpoint height - cube: median {dz_at_best.median() * 1000:+6.1f} mm"
  f"   (0 = level with the cube centre)"
)
print(
  "\nreading: site close but pad far  -> geometry (pads are not where the site is)"
  "\n         both far                -> approach never happens (reward shaping)"
)
