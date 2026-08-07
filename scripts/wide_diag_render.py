"""Which camera setting lets the render megakernel compile?

`render.__locals__._render_megakernel` is JIT-specialised on the RenderContext's
static flags (textures, skybox, backface culling, the set of geom types that can
be hit) exactly the way `primitive_narrowphase` is specialised on collision pair
types -- and on this repo's mujoco-warp it dies the same way, with NVRTC's
"unable to obtain mapped memory". Each case below drops one static branch.

  uv run python scripts/wide_diag_render.py
"""

from __future__ import annotations

import os
import subprocess
import sys

TASK = "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success"
OLD = "Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth"

CHILD = """
import sys, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task, tweak = sys.argv[1], sys.argv[2]
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 4
cfg.terminations = {}
for s in cfg.scene.sensors or ():
  if type(s).__name__ == "CameraSensorCfg":
    if tweak.startswith("g"):
      s.enabled_geom_groups = tuple(int(c) for c in tweak[1:])
    elif tweak == "no_shadow_tex":
      s.use_textures = False
      s.use_shadows = False
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
obs = env.step(torch.zeros(4, env.action_manager.total_action_dim, device="cuda:0"))
env.close()
print("RENDER_OK")
"""

CASES = [
  (OLD, "g012", "OLD claw distill, groups (0,1,2)  <- isolates scene from groups"),
  (TASK, "g12", "wide distill, groups (1,2)  visuals only"),
  (TASK, "g01", "wide distill, groups (0,1)"),
  (TASK, "g02", "wide distill, groups (0,2)"),
  (TASK, "g0", "wide distill, group (0,)  bench + cube only"),
  (TASK, "g1", "wide distill, group (1,)  claw/plate/mount visuals only"),
  (TASK, "g2", "wide distill, group (2,)  arm visuals only"),
]


def main() -> None:
  for task, tweak, label in CASES:
    p = subprocess.run(
      [sys.executable, "-c", CHILD, task, tweak],
      env={**os.environ, "MUJOCO_GL": "egl"},
      capture_output=True,
      text=True,
      timeout=1200,
    )
    good = "RENDER_OK" in p.stdout
    print(f"{'OK  ' if good else 'FAIL'}  {label}", flush=True)
    if not good:
      for ln in (p.stdout + p.stderr).splitlines():
        if "Catastrophic" in ln or "NVRTC" in ln or "(error)" in ln:
          print("        ", ln.strip()[:140])


if __name__ == "__main__":
  main()
