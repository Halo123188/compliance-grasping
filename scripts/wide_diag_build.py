"""Which part of the new scene breaks the warp narrowphase kernel build?

`_primitive_narrowphase` is JIT-specialised on the set of primitive geom-type
pairs present in the model, so a scene that adds a new pair type compiles a new
kernel -- and the new one dies with NVRTC "unable to obtain mapped memory".
This walks the scene up from the old task to the new one, one addition at a
time, and reports the first configuration that fails.

  uv run python scripts/wide_diag_build.py
"""

from __future__ import annotations

import os
import subprocess
import sys

CASES = [
  ("old task (control)", {}, "Mjlab-Grasp-TwoFinger-Flexiv"),
  ("wide, no foam no wall", {"CG_FOAM_H": "0", "CG_WALL": "0"}, None),
  ("wide, foam, no wall", {"CG_FOAM_H": "0.05", "CG_WALL": "0"}, None),
  ("wide, no foam, wall", {"CG_FOAM_H": "0", "CG_WALL": "1"}, None),
  ("wide, full", {"CG_FOAM_H": "0.05", "CG_WALL": "1"}, None),
]

CHILD = """
import sys, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(sys.argv[1], play=True)
cfg.scene.num_envs = 8
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
env.step(torch.zeros(8, env.action_manager.total_action_dim, device="cuda:0"))
env.close()
print("BUILD_OK")
"""


def main() -> None:
  for label, envvars, task in CASES:
    task = task or "Mjlab-Grasp-TwoFingerWide-Flexiv"
    e = {**os.environ, **envvars, "MUJOCO_GL": "egl"}
    p = subprocess.run(
      [sys.executable, "-c", CHILD, task],
      env=e,
      capture_output=True,
      text=True,
      timeout=900,
    )
    good = "BUILD_OK" in p.stdout
    print(f"{'OK  ' if good else 'FAIL'}  {label:26s}  {task}")
    if not good:
      tail = [
        ln
        for ln in (p.stdout + p.stderr).splitlines()
        if "error" in ln.lower() or "Exception" in ln
      ]
      for ln in tail[:6]:
        print("        ", ln.strip()[:150])


if __name__ == "__main__":
  main()
