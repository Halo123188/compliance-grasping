"""Deployment evaluation: success, near-misses, and what joint7 is doing.

  uv run python scripts/eval_deploy.py NAME=TASK:CKPT [NAME=TASK:CKPT ...]

Handles both PPO and distillation checkpoints. On a distillation task the
student is scored by default; append ":teacher" to score the frozen teacher
instead, which is how you check that the distillation env feeds the teacher the
same observation it was trained on:

  NAME=Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth:/path/v11_align.pt:teacher

Two metrics rather than one, because the training log's `episode_success` has
twice now been the wrong number to steer by:

  success    peak >= 100 mm AND held there 1 s -- the task's own bar
  near-miss  lifted but topped out in [90, 100) mm without ever succeeding

The second exists because the failure analysis of run 46984 found every failure
was a short lift with a median peak of 91.1 mm -- 9 mm under the bar, with the
grasp never lost. A change that converts near-misses into successes and a change
that converts never-lifted into near-misses both look like nothing on the success
number alone, and they are not the same kind of progress.

Also reports joint7. The wrist roll can supply the +-45 deg that squaring the jaw
to an arbitrary cube yaw needs, its limit is +-3.0543 rad (+-175 deg) so the
range is not the constraint, and on run 46984 it sat at +19.1 deg with a 4.3 deg
spread and corr(cube yaw, joint7) = -0.089. Whether any of these variants
unfroze it is the question the alignment work was really asking.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import geometry_for  # noqa: E402

N, STEPS, DEV = 128, 300, "cuda:0"
HOLD = 50
NEAR_BAND = 0.010  # a near-miss tops out within 10 mm below the bar


def _load_policy(runner, ckpt: str, role: str):
  """Load `ckpt` into `runner` and return the network that should act.

  Which state dict to pull depends on how the checkpoint was produced, not on
  the task: a distillation run can be scored on either of its two models, and a
  PPO checkpoint is a valid teacher for a distillation task.
  """
  if role == "teacher":
    runner.load(ckpt, load_cfg={"teacher": True, "iteration": False}, map_location=DEV)
    runner.alg.eval_mode()
    return runner.alg.teacher
  is_distilled = "student_state_dict" in torch.load(
    ckpt, map_location="cpu", weights_only=False
  )
  load_cfg = {"student": True} if is_distilled else {"actor": True}
  runner.load(ckpt, load_cfg=load_cfg, strict=True, map_location=DEV)
  return runner.get_inference_policy(device=DEV)


def evaluate(task: str, ckpt: str, role: str = "policy"):
  # Scene constants come from the TASK, not from a fixed import: the wide-claw
  # bench measures cube height from the top of 50 mm of foam, not from the table.
  geo = geometry_for(task)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  policy = _load_policy(runner, ckpt, role)

  robot, cube = env.scene["robot"], env.scene["cube"]
  j7 = list(robot.joint_names).index("joint7")
  sn = list(robot.site_names)
  li, ri = sn.index(geo.pad_sites[0]), sn.index(geo.pad_sites[1])

  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  Hs, J7, YAW = [], [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    Hs.append(
      (
        cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - geo.surface_z
      ).clone()
    )
    J7.append(robot.data.joint_pos[:, j7].clone())
    pads = robot.data.site_pos_w[:, [li, ri]]
    ax = pads[:, 1] - pads[:, 0]
    jaw = torch.atan2(ax[:, 1], ax[:, 0])
    q = cube.data.root_link_quat_w
    cy = torch.atan2(
      2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    YAW.append(torch.stack([jaw, cy], -1).clone())
  H = torch.stack(Hs).cpu()
  J = torch.stack(J7).cpu()
  Y = torch.stack(YAW).cpu()
  env.close()

  peak = H.max(0).values
  above = H > geo.lift_height
  run = torch.zeros(N, dtype=torch.long)
  cur = torch.zeros(N, dtype=torch.long)
  for t in range(STEPS):
    cur = torch.where(above[t], cur + 1, torch.zeros_like(cur))
    run = torch.maximum(run, cur)
  succ = run >= HOLD
  near = (~succ) & (peak >= geo.lift_height - NEAR_BAND)
  half = slice(STEPS // 2, None)
  j7s = J[half].mean(0).numpy()
  cys = Y[half, :, 1].mean(0).numpy()
  fold = lambda a: np.degrees(a % (np.pi / 2))  # noqa: E731
  corr = float(np.corrcoef(fold(cys), fold(j7s))[0, 1])
  jerr = np.degrees(Y[half, :, 0] - Y[half, :, 1]).numpy() % 90
  jerr = np.minimum(jerr, 90 - jerr).mean(0)
  return dict(
    succ=float(succ.float().mean()),
    near=float(near.float().mean()),
    dead=float(((~succ) & (peak < geo.lift_height - NEAR_BAND)).float().mean()),
    peak_med=float(peak.median()),
    peak_fail=float(peak[~succ].median()) if (~succ).any() else float("nan"),
    j7_mean=float(np.degrees(j7s).mean()),
    j7_std=float(np.degrees(j7s).std()),
    j7_rng=(float(np.degrees(j7s).min()), float(np.degrees(j7s).max())),
    corr=corr,
    yaw_err=float(jerr.mean()),
    yaw_err_s=float(jerr[succ.numpy()].mean()) if succ.any() else float("nan"),
    yaw_err_f=float(jerr[~succ.numpy()].mean()) if (~succ).any() else float("nan"),
  )


runs = []
for spec in sys.argv[1:]:
  name, rest = spec.split("=", 1)
  task, ckpt = rest.split(":", 1)
  role = "policy"
  if ckpt.endswith((":teacher", ":student")):
    ckpt, role = ckpt.rsplit(":", 1)
    role = "teacher" if role == "teacher" else "policy"
  runs.append((name, evaluate(task, ckpt, role)))
  _bar = geometry_for(task).lift_height

print(
  f"\n{N} envs, {STEPS} steps, deterministic policy. Bar"
  f" {_bar * 1000:.0f} mm held {HOLD / 50:.0f} s; near-miss = peak within"
  f" {NEAR_BAND * 1000:.0f} mm below the bar without succeeding.\n"
)
print(
  f"{'run':>12} {'success':>9} {'near-miss':>10} {'neither':>9}"
  f" {'median peak':>12} {'peak|fail':>10}"
)
for n, r in runs:
  print(
    f"{n:>12} {r['succ'] * 100:8.1f}% {r['near'] * 100:9.1f}%"
    f" {r['dead'] * 100:8.1f}% {r['peak_med'] * 1000:11.1f}mm"
    f" {r['peak_fail'] * 1000:9.1f}mm"
  )

print(
  f"\n{'run':>12} {'joint7 mean':>12} {'std':>7} {'range':>18}"
  f" {'corr(yaw,j7)':>13} {'jaw err':>8} {'err|ok':>7} {'err|fail':>9}"
)
for n, r in runs:
  print(
    f"{n:>12} {r['j7_mean']:+11.1f}d {r['j7_std']:6.1f}d"
    f" {r['j7_rng'][0]:+8.1f}..{r['j7_rng'][1]:+7.1f}"
    f" {r['corr']:+13.3f} {r['yaw_err']:7.1f}d {r['yaw_err_s']:6.1f}d"
    f" {r['yaw_err_f']:8.1f}d"
  )
print("\njoint7 limit is +-175 deg; squaring the jaw needs at most +-45 deg.")
print("corr ~ +1 means the wrist tracks cube yaw; ~0 means it ignores it.")
