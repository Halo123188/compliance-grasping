"""Render a direct-torque compliance policy under a scripted human push.

The impedance demo (``render_demo.py``) shows a commanded stiffness ``K``; a
direct-torque policy has none, so this variant shows what the torque policy
actually does:

  Left: the MuJoCo scene with the teacher's debug markers —
    green  = teacher target x_t,  yellow = un-yielded reference x_ref(s),
    blue   = human hand anchor,   orange arrow = applied force F_ext.
  Watch the arm get dragged off x_ref along the push (the yield), then return
  and resume when the hand lets go.

  Right, per step —
    INPUT    joint torque tau (7)         — the only channel carrying the push.
    SENSED   true F_ext vs the policy's own force estimate (aux head), so you can
             see it *infer* the push from proprioception (train-time label, gone
             at deploy). Omitted for a plain-torque policy with no head.
    COMPLY   physical k_par = |F_ext| / yield_along_push (N/m) and the yield in
             cm — low k_par / big yield == soft along the push.

  MUJOCO_GL=egl uv run python -m \
    mjlab.tasks.compliance_tracking.scripts.render_torque \
    --task Mjlab-ComplianceTracking-StageA-TorqueAux-Flexiv \
    --checkpoint logs/.../model_4999.pt --out compliance_torqueaux.mp4
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import torch
import tyro
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def _resolve_ckpt(path: str) -> Path:
  p = Path(path)
  if p.is_dir():
    ckpts = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    assert ckpts, f"no model_*.pt in {p}"
    return ckpts[-1]
  return p


def _barh(ax, labels, vals, lo, hi, colors, title, fmt="{:.0f}", logx=False):
  y = np.arange(len(labels))
  ax.barh(y, np.clip(vals, lo, hi), color=colors)
  ax.set_xlim(lo, hi)
  if logx:
    ax.set_xscale("log")
  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=8)
  ax.invert_yaxis()
  ax.set_title(title, fontsize=9, loc="left")
  ax.tick_params(labelsize=7)
  for yi, v in zip(y, vals, strict=False):
    ax.text(
      hi, yi, "  " + fmt.format(v), va="center", ha="left", fontsize=7, clip_on=False
    )


def _panel(canvas, fig, axes, info, height, has_est):
  for ax in axes:
    ax.clear()
  ax_tau, ax_sense, ax_k = axes[0], axes[1], axes[2]

  _barh(
    ax_tau,
    [f"t{i + 1}" for i in range(7)],
    info["tau"],
    -30,
    30,
    ["#55a868"] * 7,
    "INPUT  joint torque tau (Nm)  — senses the push",
    fmt="{:+.1f}",
  )

  if has_est:
    ft, fe = info["f_true"], info["f_est"]
    vals = [ft[0], fe[0], ft[1], fe[1], ft[2], fe[2]]
    cols = ["#8899aa", "#00b3b3"] * 3
    _barh(
      ax_sense,
      ["Fx true", "Fx est", "Fy true", "Fy est", "Fz true", "Fz est"],
      vals,
      -40,
      40,
      cols,
      "SENSED  true push (grey) vs policy estimate (cyan) — N",
      fmt="{:+.0f}",
    )
  else:
    ft = info["f_true"]
    _barh(
      ax_sense,
      ["Fx", "Fy", "Fz"],
      ft,
      -40,
      40,
      ["#8899aa"] * 3,
      "PUSH  true F_ext (N)",
      fmt="{:+.0f}",
    )

  kpar = info["kpar"]
  if kpar is not None:
    col = "#4c72b0" if kpar < 400 else "#c44e52"
    _barh(
      ax_k,
      ["k_par"],
      [kpar],
      50,
      2000,
      [col],
      f"COMPLY  k_par along push (N/m)   yield {info['yield'] * 100:.1f} cm",
      logx=True,
    )
  else:
    ax_k.text(
      0.5,
      0.5,
      "(no push this step)",
      ha="center",
      va="center",
      fontsize=9,
      transform=ax_k.transAxes,
    )
    ax_k.set_title("COMPLY  k_par along push", fontsize=9, loc="left")
    ax_k.set_xticks([])
    ax_k.set_yticks([])

  kp = "  —" if kpar is None else f"{kpar:.0f} N/m"
  fig.suptitle(
    f"step {info['t']:3d}   {info['phase']:6s}   s={info['s']:.2f}   "
    f"dev={info['dev'] * 100:4.1f}cm   |F_ext|={info['f']:4.1f}N   k_par={kp}",
    fontsize=10,
    x=0.02,
    ha="left",
  )
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
  img = Image.fromarray(buf).resize(
    (int(buf.shape[1] * height / buf.shape[0]), height), Image.BILINEAR
  )
  return np.asarray(img)


def main(
  task: str,
  checkpoint: str,
  out: str = "compliance_torque_demo.mp4",
  steps: int = 420,
  fps: int = 40,
  device: str = "cuda:0",
  seed: int = 0,
  onset_s: float = 0.35,
  displacement: float = 0.20,
  hold_s: float = 2.4,
  stiffness: float = 800.0,
) -> None:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.viewer.width = 720
  cfg.viewer.height = 640
  cfg.seed = seed

  pert = cfg.commands["teacher"].perturbation
  pert.p_no_perturbation = 0.0
  pert.num_events_range = (1, 1)
  pert.onset_s_range = (onset_s, onset_s)
  pert.displacement_range = (displacement, displacement)
  pert.hold_time_range = (hold_s, hold_s)
  pert.ramp_time_range = (0.5, 0.5)
  pert.stiffness_range = (stiffness, stiffness)
  force_limit = float(pert.force_limit)

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  teacher = env.command_manager.get_term("teacher")
  assert isinstance(teacher, TeacherCommand)

  ckpt = _resolve_ckpt(checkpoint)
  print(f"loading {ckpt}")
  agent_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  raw_actor = runner.alg._raw_actor
  has_est = hasattr(raw_actor, "predict_force")
  print(f"force-estimation head: {'yes' if has_est else 'no'}")
  obs = wrapped.get_observations()

  fig = plt.figure(figsize=(4.6, 6.4), dpi=110)
  axes = tuple(fig.add_subplot(3, 1, i + 1) for i in range(3))
  canvas = FigureCanvasAgg(fig)

  frames: list[np.ndarray] = []
  for t in range(steps):
    with torch.no_grad():
      action = policy(obs)
      f_est = (
        raw_actor.predict_force()[0].cpu().numpy() * force_limit if has_est else None
      )
    obs, _, dones, _ = wrapped.step(action)
    if hasattr(policy, "reset"):
      policy.reset(dones)

    scene = env.render()
    if scene is None:
      continue
    height = scene.shape[0]

    u = teacher.perturbation.direction[0]
    pushing = bool(teacher.perturbation.active[0].item())
    dev_vec = teacher.ee_pos_w()[0] - teacher.x_ref()[0]
    f = float(torch.norm(teacher.perturbation.force[0]).item())
    yld = float((dev_vec * u).sum().item())
    kpar = f / max(yld, 2e-3) if (pushing and f > 5.0 and yld > 2e-3) else None
    info = {
      "t": t,
      "phase": "PUSHED" if pushing else "free",
      "s": float(teacher.s[0].item()),
      "dev": float(teacher.deviation[0].item()),
      "f": f,
      "kpar": kpar,
      "yield": max(yld, 0.0),
      "tau": obs["actor"][0, 14:21].cpu().numpy(),
      "f_true": teacher.perturbation.force[0].cpu().numpy(),
      "f_est": f_est,
    }
    frames.append(
      np.concatenate([scene, _panel(canvas, fig, axes, info, height, has_est)], axis=1)
    )

  print(f"writing {out}  ({len(frames)} frames @ {fps} fps)")
  mediapy.write_video(out, frames, fps=fps)
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
