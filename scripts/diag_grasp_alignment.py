"""Is the jaw squared to a cube FACE, or closing on an EDGE? And at what height?

  uv run python scripts/diag_grasp_alignment.py TASK CKPT

Two things this settles, both of which look identical in the reward logs:

  alignment  Cube yaw is uniform on [-pi, pi] and the wrist has to roll to match
             it. A box has four side normals, +-x and +-y of its own frame, so
             the jaw is square to a face when the closing axis is parallel to
             one of them. Misalignment is reported folded into 0..45 deg: 0 is
             flat on a face, 45 is dead on a corner. A corner grip is point
             contact -- almost no friction area -- which is what a cube that
             tips and slides out of the fingers looks like.

  height     Where the pads sit relative to the cube's centre. Gripping above
             the centre puts a moment on the cube even when the grip itself is
             fine.

Both are reported at the moment of closest approach AND at the end of the
episode, because they are not the same instant: the hand can arrive square and
then rotate while closing.
"""

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, STEPS, DEV = 64, 300, "cuda:0"

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
sn = list(robot.site_names)
li, ri = sn.index("left_pad"), sn.index("right_pad")


def misalign_deg(pl: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
  """Angle between the jaw's closing axis and the nearest cube side normal."""
  axis = pr - pl
  axis = axis[:, :2] / axis[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-9)
  # Cube yaw from its world rotation matrix; a box repeats every 90 deg, so the
  # nearest normal is found by folding the angle difference into a quarter turn.
  m = cube.data.root_link_quat_w
  w, x, y, z = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  jaw = torch.atan2(axis[:, 1], axis[:, 0])
  d = (jaw - yaw) % (torch.pi / 2)
  return torch.rad2deg(torch.minimum(d, torch.pi / 2 - d))


obs = wrapped.reset()[0]
best_d = torch.full((N,), 1e9, device=DEV)
mis_at_best = torch.zeros(N, device=DEV)
dz_at_best = torch.zeros(N, device=DEV)
for _ in range(STEPS):
  with torch.inference_mode():
    act = policy(obs)
  obs, _, _, _ = wrapped.step(act)
  pl = robot.data.site_pos_w[:, li]
  pr = robot.data.site_pos_w[:, ri]
  c = cube.data.root_link_pos_w
  mid = 0.5 * (pl + pr)
  d = torch.norm(mid - c, dim=-1)
  upd = d < best_d
  best_d = torch.minimum(best_d, d)
  mis_at_best = torch.where(upd, misalign_deg(pl, pr), mis_at_best)
  dz_at_best = torch.where(upd, (mid - c)[:, 2], dz_at_best)

pl = robot.data.site_pos_w[:, li]
pr = robot.data.site_pos_w[:, ri]
c = cube.data.root_link_pos_w
mis_end = misalign_deg(pl, pr)
dz_end = (0.5 * (pl + pr) - c)[:, 2]


def q(t: torch.Tensor, s: float = 1.0) -> str:
  v = (t * s).sort().values
  return (
    f"median {v[N // 2]:+7.2f}   p10 {v[N // 10]:+7.2f}   p90 {v[9 * N // 10]:+7.2f}"
  )


print(f"\ncheckpoint: {CKPT}")
print(f"envs={N} steps={STEPS}\n")
print("jaw-vs-face misalignment (deg, 0 = flat on a face, 45 = on a corner)")
print(f"  at closest approach : {q(mis_at_best)}")
print(f"  at end of episode   : {q(mis_end)}")
print(
  f"  fraction past 30 deg (nearer a corner than a face): "
  f"{(mis_end > 30).float().mean():.2f}"
)
print("\npad midpoint height minus cube centre (mm, + = gripping above centre)")
print(f"  at closest approach : {q(dz_at_best, 1000)}")
print(f"  at end of episode   : {q(dz_end, 1000)}")
print(f"\npad midpoint distance to cube centre at closest (mm): {q(best_d, 1000)}")
