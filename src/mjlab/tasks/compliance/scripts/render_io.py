"""Render a compliance rollout with a live policy-I/O dashboard.

Left half: the MuJoCo scene (arm, green goal sphere, cyan hand-target sphere,
orange human-force arrow).  Right half: what the policy sees and decides this
step —

  INPUTS   joint torque tau (7)  -> how it senses the human push
           goal error (3)
  OUTPUTS  Cartesian stiffness K (3 world axes, 50..2000 N/m)  <- soft vs stiff
           delta x_ref (3, +/-3 cm)
  DERIVED  K_par / K_perp relative to the push direction  (<1 = arbitration)
           |F_ext|, distance-to-goal, human phase

Run (CPU, headless):
  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.render_io \
    --checkpoint logs/rsl_rl/compliance_reach_flexiv/<run>/model_2499.pt \
    --out io_demo.mp4
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

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_TASK = "Mjlab-Compliance-Reach-Flexiv"
_PHASE = {0: "idle", 1: "reaching", 2: "PUSHING", 3: "released"}


def _resolve_ckpt(path: str) -> Path:
  p = Path(path)
  if p.is_dir():
    ckpts = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    assert ckpts, f"no model_*.pt in {p}"
    return ckpts[-1]
  return p


def _bar(ax, labels, vals, lo, hi, colors, title, fmt="{:.0f}"):
  y = np.arange(len(labels))
  ax.barh(y, np.clip(vals, lo, hi), color=colors)
  ax.set_xlim(lo, hi)
  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=8)
  ax.invert_yaxis()
  ax.set_title(title, fontsize=9, loc="left")
  ax.tick_params(labelsize=7)
  for yi, v in zip(y, vals, strict=False):
    ax.text(
      hi, yi, "  " + fmt.format(v), va="center", ha="left", fontsize=7, clip_on=False
    )


def _panel(canvas, fig, axes, info, H):
  (ax_k, ax_dx, ax_tau, ax_ge) = axes
  for ax in axes:
    ax.clear()

  if info.get("mode") == "torque":
    # Torque action: the two OUTPUT panels show the commanded joint torque and
    # the raw policy action (both 7-dim) instead of stiffness / delta x_ref.
    tc = info["torque_cmd"]
    _bar(
      ax_k,
      [f"c{i + 1}" for i in range(7)],
      tc,
      -130,
      130,
      ["#8172b3"] * 7,
      "OUTPUT  commanded torque (Nm)",
      fmt="{:+.0f}",
    )
    ra = info["raw_action"]
    _bar(
      ax_dx,
      [f"a{i + 1}" for i in range(7)],
      ra,
      -1,
      1,
      ["#4c72b0"] * 7,
      "OUTPUT  raw action [-1,1]",
      fmt="{:+.2f}",
    )
  else:
    k = info["k"]
    # Soft = blue, stiff = red (log-scaled position in [50, 2000]).
    kc = plt.cm.coolwarm((np.log(k) - np.log(50)) / (np.log(2000) - np.log(50)))
    _bar(ax_k, ["Kx", "Ky", "Kz"], k, 50, 2000, kc, "OUTPUT  stiffness K (N/m)")
    ax_k.set_xscale("log")

    dx = info["dx"] * 100.0  # cm
    _bar(
      ax_dx,
      ["dx", "dy", "dz"],
      dx,
      -3,
      3,
      ["#4c72b0"] * 3,
      "OUTPUT  delta x_ref (cm)",
      fmt="{:+.2f}",
    )
  tau = info["tau"]
  _bar(
    ax_tau,
    [f"t{i + 1}" for i in range(7)],
    tau,
    -30,
    30,
    ["#55a868"] * 7,
    "INPUT  joint torque tau (Nm)",
    fmt="{:+.1f}",
  )
  ge = info["ge"] * 100.0
  _bar(
    ax_ge,
    ["ex", "ey", "ez"],
    ge,
    -40,
    40,
    ["#c44e52"] * 3,
    "INPUT  goal error (cm)",
    fmt="{:+.1f}",
  )

  ratio = "  —" if info["ratio"] is None else f"{info['ratio']:.2f}"
  fig.suptitle(
    f"step {info['t']:3d}   phase={info['phase']:8s}   "
    f"dist={info['dist'] * 100:5.1f} cm   |F_ext|={info['f']:5.1f} N   "
    f"K∥/K⊥={ratio}",
    fontsize=10,
    x=0.02,
    ha="left",
  )
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
  # Resize panel height to match the MuJoCo frame height H.
  from PIL import Image

  img = Image.fromarray(buf).resize(
    (int(buf.shape[1] * H / buf.shape[0]), H), Image.BILINEAR
  )
  return np.asarray(img)


def main(
  checkpoint: str,
  out: str = "io_demo.mp4",
  task: str = _TASK,
  steps: int = 350,
  fps: int = 40,
  device: str = "cpu",
  seed: int = 0,
  d_lo: float = 0.12,
  d_hi: float = 0.20,
  f_max: float | None = None,
  second_push_prob: float = 0.8,
  second_delay_lo: float | None = None,
  second_delay_hi: float | None = None,
  grab_body: str | None = None,
) -> None:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.viewer.width = 720
  cfg.viewer.height = 640
  cfg.seed = seed
  # Scenario overrides (applied before the env builds so per-env sampling picks
  # them up): pin f_max / grab body / repeat-pull rate to script one demo.
  if f_max is not None:
    cfg.events["human_disturbance"].params["f_max_range"] = (f_max, f_max)
  if grab_body is not None:
    cfg.events["human_disturbance"].params["grab_bodies"] = (grab_body,)
  if second_delay_lo is not None and second_delay_hi is not None:
    cfg.events["human_disturbance"].params["second_delay_range"] = (
      second_delay_lo,
      second_delay_hi,
    )
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")

  human = env.event_manager.get_term_cfg("human_disturbance").func
  human.d_range = (d_lo, d_hi)  # dramatic push for the demo
  human._push_time_range = (0.6, 1.2)
  human._second_push_prob = second_push_prob
  reach = env.command_manager.get_term("reach")
  # Impedance task exposes a term named "impedance"; the direct-torque variants
  # expose "torque".  Render adapts the two OUTPUT panels to whichever is present.
  act_name = "impedance" if "impedance" in env.action_manager.active_terms else "torque"
  act_term = env.action_manager.get_term(act_name)
  is_torque = act_name == "torque"

  ckpt = _resolve_ckpt(checkpoint)
  print(f"loading {ckpt}")
  agent_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  obs = wrapped.get_observations()

  fig = plt.figure(figsize=(4.6, 6.4), dpi=110)
  axes = (
    fig.add_subplot(4, 1, 1),
    fig.add_subplot(4, 1, 2),
    fig.add_subplot(4, 1, 3),
    fig.add_subplot(4, 1, 4),
  )
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
    H = scene.shape[0]

    u = human.push_dir()[0].cpu().numpy()
    pushing = bool(human.is_pushing()[0].item())
    ratio = None
    if not is_torque:
      k = act_term.stiffness[0].cpu().numpy()
      if pushing and np.linalg.norm(u) > 1e-6:
        k_par = float((u**2 * k).sum())
        k_perp = float((k.sum() - k_par) / 2.0)
        ratio = k_par / max(k_perp, 1e-6)
    f = float(torch.norm(human.external_force()[0]).item())
    dist = float(torch.norm(reach.command[0] - reach.ee_pos_w()[0]).item())
    tau = obs["actor"][0, 14:21].cpu().numpy()
    ge = obs["actor"][0, 27:30].cpu().numpy()

    info = {
      "t": t,
      "phase": _PHASE[int(human._phase[0].item())],
      "dist": dist,
      "f": f,
      "ratio": ratio,
      "tau": tau,
      "ge": ge,
    }
    if is_torque:
      info["mode"] = "torque"
      info["torque_cmd"] = act_term.processed_torque[0].cpu().numpy()
      info["raw_action"] = act_term.raw_action[0].cpu().numpy()
    else:
      info["k"] = act_term.stiffness[0].cpu().numpy()
      info["dx"] = act_term._delta_pos[0].cpu().numpy()
    panel = _panel(canvas, fig, axes, info, H)
    frames.append(np.concatenate([scene, panel], axis=1))

  env.close()
  mediapy.write_video(out, frames, fps=fps)
  print(f"wrote {len(frames)} frames -> {out}")


if __name__ == "__main__":
  tyro.cli(main)
