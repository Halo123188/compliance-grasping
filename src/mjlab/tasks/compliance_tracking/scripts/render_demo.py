"""Render a trained compliance-tracking policy under a scripted human push.

Left: the MuJoCo scene. The teacher's debug markers render offline —
  green sphere  = teacher target x_t (what the student tracks)
  yellow sphere = un-yielded reference x_ref(s) (frozen while pushed)
  blue sphere   = human hand anchor
  orange arrow  = applied external force F_ext
so you can watch the arm get dragged off x_ref, the reference freeze, and the
arm return and resume when the hand lets go.

Right: what the policy senses and commands this step —
  INPUT   joint torque tau (7)      — the ONLY channel carrying the push
  OUTPUT  stiffness K (x/y/z)       — blue=soft, red=stiff (log 50..2000 N/m)
  DERIVED K_par vs K_perp about the pull — the compliance question. Genuine
          compliance is K_par < K_perp (soft along the push); this policy does
          the opposite, and the two bars show it directly.

The push is scripted (not sampled) so the demo is deterministic: one strong,
sustained pull partway through the reach.

  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance_tracking.scripts.render_demo \
    --stage A \
    --checkpoint logs/rsl_rl/compliance_tracking_flexiv_a/<run>/model_4999.pt \
    --out compliance_track_A.mp4
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

from mjlab.asset_zoo.robots.flexiv_three_hand.constants import FT_NORMAL_AXIS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.compliance_tracking.config.flexiv.env_cfg import FORCE_SENSORS
from mjlab.tasks.compliance_tracking.mdp.observations import total_grasp_force
from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_TASK = {
  "A": "Mjlab-ComplianceTracking-StageA-Flexiv",
  "B": "Mjlab-ComplianceTracking-StageB-Flexiv",
  "C": "Mjlab-ComplianceTracking-StageC-Flexiv",
}


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


def _panel(canvas, fig, axes, info, height):
  # 4 axes for Stage B/C (a grasp-force row), 3 for Stage A.
  if len(axes) == 4:
    ax_k, ax_kpp, ax_grip, ax_tau = axes
  else:
    ax_k, ax_kpp, ax_tau = axes
    ax_grip = None
  for ax in axes:
    ax.clear()

  k = info["k"]
  kc = plt.cm.coolwarm((np.log(k) - np.log(50)) / (np.log(2000) - np.log(50)))
  _barh(
    ax_k,
    ["Kx", "Ky", "Kz"],
    k,
    50,
    2000,
    kc,
    "OUTPUT  stiffness K (N/m)   blue=soft red=stiff",
    logx=True,
  )

  if info["kpar"] is not None:
    vals = [info["kpar"], info["kperp"]]
    # Red the one that is *larger* — we want K_par (top) to be the smaller/blue.
    cols = [
      "#c44e52" if info["kpar"] >= info["kperp"] else "#4c72b0",
      "#4c72b0" if info["kpar"] >= info["kperp"] else "#c44e52",
    ]
    _barh(
      ax_kpp,
      ["K∥ (along push)", "K⊥ (across)"],
      vals,
      50,
      2000,
      cols,
      "DERIVED  compliance: want K∥ < K⊥",
      logx=True,
    )
  else:
    ax_kpp.text(
      0.5,
      0.5,
      "(no push this step)",
      ha="center",
      va="center",
      fontsize=9,
      transform=ax_kpp.transAxes,
    )
    ax_kpp.set_title("DERIVED  compliance: want K∥ < K⊥", fontsize=9, loc="left")
    ax_kpp.set_xticks([])
    ax_kpp.set_yticks([])

  if ax_grip is not None:
    # Grasp force held vs the scripted target, with the target as a marker line.
    fg, ft = info["fgrip"], info["ftarget"]
    col = "#55a868" if abs(fg - ft) < 4.0 else "#dd8452"
    _barh(
      ax_grip,
      ["F_grip"],
      [fg],
      0,
      40,
      [col],
      f"GRASP  measured force vs target {ft:.0f} N",
      fmt="{:.1f}",
    )
    ax_grip.axvline(ft, color="k", ls="--", lw=1.2)

  tau = info["tau"]
  _barh(
    ax_tau,
    [f"t{i + 1}" for i in range(7)],
    tau,
    -30,
    30,
    ["#55a868"] * 7,
    "INPUT  joint torque tau (Nm)  — senses the push",
    fmt="{:+.1f}",
  )

  ratio = (
    "  —" if info["kpar"] is None else f"{info['kpar'] / max(info['kperp'], 1e-6):.2f}"
  )
  grip = (
    "" if ax_grip is None else f"   F_grip={info['fgrip']:4.1f}/{info['ftarget']:.0f}N"
  )
  fig.suptitle(
    f"step {info['t']:3d}   {info['phase']:9s}   s={info['s']:.2f}   "
    f"dev={info['dev'] * 100:4.1f}cm   |F_ext|={info['f']:4.1f}N   K∥/K⊥={ratio}{grip}",
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
  checkpoint: str,
  stage: str = "A",
  out: str = "compliance_track_demo.mp4",
  steps: int = 420,
  fps: int = 40,
  device: str = "cpu",
  seed: int = 0,
  onset_s: float = 0.35,
  displacement: float = 0.18,
  hold_s: float = 2.2,
  stiffness: float = 600.0,
  dump_png: tuple[int, ...] = (),
) -> None:
  task = _TASK[stage]
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.viewer.width = 720
  cfg.viewer.height = 640
  cfg.seed = seed

  # Script one strong, sustained pull partway through the reach: deterministic
  # so the demo always shows the push/yield/return story rather than a sampled
  # (and possibly empty) schedule.
  pert = cfg.commands["teacher"].perturbation
  pert.p_no_perturbation = 0.0
  pert.num_events_range = (1, 1)
  pert.onset_s_range = (onset_s, onset_s)
  pert.displacement_range = (displacement, displacement)
  pert.hold_time_range = (hold_s, hold_s)
  pert.ramp_time_range = (0.5, 0.5)
  pert.stiffness_range = (stiffness, stiffness)

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  teacher = env.command_manager.get_term("teacher")
  assert isinstance(teacher, TeacherCommand)
  impedance = env.action_manager.get_term("impedance")

  with_object = stage in ("B", "C")

  ckpt = _resolve_ckpt(checkpoint)
  print(f"loading {ckpt}")
  agent_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  obs = wrapped.get_observations()

  nrow = 4 if with_object else 3
  fig = plt.figure(figsize=(4.6, 6.4), dpi=110)
  axes = tuple(fig.add_subplot(nrow, 1, i + 1) for i in range(nrow))
  canvas = FigureCanvasAgg(fig)

  frames: list[np.ndarray] = []
  for t in range(steps):
    with torch.no_grad():
      action = policy(obs)
    obs, _, dones, _ = wrapped.step(action)
    if hasattr(policy, "reset"):
      policy.reset(dones)

    scene = env.render()
    if scene is None:
      continue
    height = scene.shape[0]

    k = impedance.stiffness[0].cpu().numpy()
    u = teacher.perturbation.direction[0].cpu().numpy()
    pushing = bool(teacher.perturbation.active[0].item())
    if pushing and np.linalg.norm(u) > 1e-6:
      kpar = float((u**2 * k).sum())
      kperp = float((k.sum() - kpar) / 2.0)
    else:
      kpar = kperp = None
    f = float(torch.norm(teacher.perturbation.force[0]).item())
    phase = "PUSHED" if pushing else "free"
    info = {
      "t": t,
      "phase": phase,
      "s": float(teacher.s[0].item()),
      "dev": float(teacher.deviation[0].item()),
      "f": f,
      "k": k,
      "kpar": kpar,
      "kperp": kperp,
      "tau": obs["actor"][0, 14:21].cpu().numpy(),
    }
    if with_object:
      info["fgrip"] = float(
        total_grasp_force(env, FORCE_SENSORS, FT_NORMAL_AXIS)[0].item()
      )
      info["ftarget"] = float(teacher.finger_force_target[0].item())
    panel = _panel(canvas, fig, axes, info, height)
    frames.append(np.concatenate([scene, panel], axis=1))

  print(f"writing {out}  ({len(frames)} frames @ {fps} fps)")
  mediapy.write_video(out, frames, fps=fps)

  if dump_png:
    # PNG snapshots so the render can be spot-checked without ffmpeg.
    stem = Path(out).with_suffix("")
    for i in dump_png:
      if 0 <= i < len(frames):
        Image.fromarray(frames[i]).save(f"{stem}_frame{i:04d}.png")
        print(f"  wrote {stem}_frame{i:04d}.png")
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
