"""Are the capsule and njmax workarounds still needed with PCH off?

Both were added while the NVRTC failure was misread as "the generated kernel is
too big". The real cause was precompiled headers (see mjlab/__init__.py), so
each needs re-testing before its comment can claim it is load-bearing.

  uv run python scripts/wide_diag_workarounds.py
"""

from __future__ import annotations

import os
import subprocess
import sys

CHILD = """
import sys, mujoco, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

case = sys.argv[2]
cfg = load_env_cfg(sys.argv[1], play=True)
cfg.scene.num_envs = 4
cfg.terminations = {}

if case in ("capsule", "both"):
  # Put the D435i's collider back to the capsule Menagerie fits to its casing,
  # which reintroduces CAPSULE-BOX as a third primitive collision pair type.
  robot = cfg.scene.entities["robot"]
  inner = robot.spec_fn
  def spec_fn():
    spec = inner()
    for g in spec.geoms:
      if g.name == "" and g.type == mujoco.mjtGeom.mjGEOM_BOX and g.meshname:
        pass
    for b in spec.bodies:
      if b.name.startswith("d435i"):
        for g in b.geoms:
          if g.contype or g.conaffinity:
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    return spec
  robot.spec_fn = spec_fn

if case in ("njmax", "both"):
  cfg.sim.njmax = 1100
  cfg.sim.nconmax = 220

env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
env.step(torch.zeros(4, env.action_manager.total_action_dim, device="cuda:0"))
env.close()
print("OK_BUILD")
"""

TASK = "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success"

for case, label in (
  ("none", "as shipped (capsule->box, njmax 600)"),
  ("capsule", "D435i capsule restored"),
  ("njmax", "njmax 1100 / nconmax 220"),
  ("both", "capsule restored AND njmax 1100"),
):
  p = subprocess.run(
    [sys.executable, "-c", CHILD, TASK, case],
    env={**os.environ, "MUJOCO_GL": "egl"},
    capture_output=True,
    text=True,
    timeout=2400,
  )
  good = "OK_BUILD" in p.stdout
  print(f"{'OK  ' if good else 'FAIL'}  {label}", flush=True)
  if not good:
    for ln in (p.stdout + p.stderr).splitlines():
      if "Catastrophic" in ln or "NVRTC" in ln or "Error" in ln or "error:" in ln:
        print("        ", ln.strip()[:130])
