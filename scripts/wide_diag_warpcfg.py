"""Two warp compiler knobs against the render-megakernel NVRTC failure.

"unable to obtain mapped memory" is an mmap failure inside NVRTC, not a
statement about the scene. Two things make NVRTC's arena blow up: precompiled
headers (a large mapped blob) and loop unrolling.
"""

import os
import subprocess
import sys

CHILD = """
import warp as wp, sys
tweak = sys.argv[2]
if tweak == "no_pch":
    wp.config.use_precompiled_headers = False
elif tweak == "unroll1":
    wp.config.max_unroll = 1
elif tweak == "both":
    wp.config.use_precompiled_headers = False
    wp.config.max_unroll = 1
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(sys.argv[1], play=True)
cfg.scene.num_envs = 4
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
env.step(torch.zeros(4, env.action_manager.total_action_dim, device="cuda:0"))
env.close()
print("RENDER_OK")
"""
TASK = "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success"
for tweak in ("none", "no_pch", "unroll1", "both"):
  p = subprocess.run(
    [sys.executable, "-c", CHILD, TASK, tweak],
    env={**os.environ, "MUJOCO_GL": "egl"},
    capture_output=True,
    text=True,
    timeout=2400,
  )
  good = "RENDER_OK" in p.stdout
  print(f"{'OK  ' if good else 'FAIL'}  {tweak}", flush=True)
  if not good:
    for ln in (p.stdout + p.stderr).splitlines():
      if "Catastrophic" in ln or "NVRTC" in ln:
        print("        ", ln.strip()[:120])
