"""Render a Stage-1 compliance rollout to video, with the requested overlays.

Draws, per frame (via the env's debug visualizers):
  * the reach goal ``x_g``            — green sphere,
  * the human hand target ``x_hand``  — cyan sphere,
  * the human force ``F_ext``         — orange arrow at the wrist.

With ``--analytic`` the policy is zero action (pure impedance). With
``--checkpoint`` a trained policy drives the arm, so the video shows learned
reach + yield-under-disturbance behaviour.

Run:
  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.render_demo --analytic
  MUJOCO_GL=egl uv run python -m mjlab.tasks.compliance.scripts.render_demo \
    --checkpoint logs/rsl_rl/compliance_reach_flexiv/<run> --out trained.mp4
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mediapy
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_TASK = "Mjlab-Compliance-Reach-Flexiv"


def _resolve_ckpt(path: str) -> Path:
  p = Path(path)
  if p.is_dir():
    ckpts = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    assert ckpts, f"no model_*.pt in {p}"
    return ckpts[-1]
  return p


def main(
  out: str = "compliance_reach_demo.mp4",
  checkpoint: str | None = None,
  analytic: bool = False,
  task: str = _TASK,
  steps: int = 700,
  fps: int = 50,
  device: str = "cpu",
  big_push: bool = True,
) -> None:
  if checkpoint is None and not analytic:
    raise SystemExit("Pass --analytic or --checkpoint <path>.")

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.viewer.width = 960
  cfg.viewer.height = 720
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")

  # Make the disturbance dramatic and frequent so it reads well on video.
  human = env.event_manager.get_term_cfg("human_disturbance").func
  if big_push:
    human.d_range = (0.12, 0.20)
    human._push_time_range = (0.6, 1.2)
    human._second_push_prob = 0.8

  if analytic:
    wrapped = None
    obs = None

    def act(_obs):
      return torch.zeros(1, env.action_manager.total_action_dim, device=device)

    def reset_policy(_d):
      pass

    env.reset()
  else:
    ckpt = _resolve_ckpt(checkpoint)  # type: ignore[arg-type]
    print(f"loading {ckpt}")
    agent_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    obs = wrapped.get_observations()

    def act(o):
      with torch.no_grad():
        return policy(o)

    def reset_policy(d):
      if hasattr(policy, "reset"):
        policy.reset(d)

  frames: list[np.ndarray] = []
  for _ in range(steps):
    action = act(obs)
    if wrapped is None:
      env.step(action)
    else:
      obs, _, dones, _ = wrapped.step(action)
      reset_policy(dones)
    frame = env.render()
    if frame is not None:
      frames.append(frame)

  env.close()
  mediapy.write_video(out, frames, fps=fps)
  print(f"wrote {len(frames)} frames -> {out}")


if __name__ == "__main__":
  tyro.cli(main)
