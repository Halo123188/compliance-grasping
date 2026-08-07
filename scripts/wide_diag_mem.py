"""Is the render-kernel NVRTC failure about MEMORY rather than kernel size?

Every geom-group subset of the wide scene fails to compile
`_render_megakernel`, including a MESH-only one whose kernel must be smaller
than the old scene's PLANE+BOX+MESH kernel -- which compiles. That rules out
"the generated code is too big" and points at NVRTC's own allocation failing,
i.e. "unable to obtain mapped memory" meaning exactly what it says.

The wide scene's meshes are far heavier than the old one's: the camera mount
alone is 48.7k faces, and the arm, the claw and the D435i model all ship visual
meshes that get a BVH built per mesh. This reports what the process is actually
holding when the compile is attempted.

  uv run python scripts/wide_diag_mem.py
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys

CHILD = """
import os, resource, sys, mujoco, torch

def rss_gb():
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = sys.argv[1]
print(f"  rss after imports        {rss_gb():7.2f} GB", flush=True)
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 4
cfg.terminations = {}
print(f"  rss after cfg            {rss_gb():7.2f} GB", flush=True)
try:
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
except Exception as e:
  print(f"  rss at FAILURE           {rss_gb():7.2f} GB", flush=True)
  print(f"  {type(e).__name__}: {str(e)[:90]}", flush=True)
  sys.exit(1)
print(f"  rss after env build      {rss_gb():7.2f} GB", flush=True)
m = env.sim.mj_model
print(f"  nmesh {m.nmesh}  total mesh faces {int(m.mesh_facenum.sum())}"
      f"  verts {int(m.mesh_vertnum.sum())}", flush=True)
print(f"  ntex {m.ntex}  nmat {m.nmat}  nlight {m.nlight}  ncam {m.ncam}", flush=True)
try:
  env.step(torch.zeros(4, env.action_manager.total_action_dim, device="cuda:0"))
except Exception as e:
  print(f"  rss at STEP FAILURE      {rss_gb():7.2f} GB", flush=True)
  print(f"  {type(e).__name__}: {str(e)[:90]}", flush=True)
  sys.exit(1)
print(f"  rss after first step     {rss_gb():7.2f} GB", flush=True)
print("RENDER_OK", flush=True)
"""


def main() -> None:
  print("host limits")
  for name, key in (("address space", "RLIMIT_AS"), ("locked", "RLIMIT_MEMLOCK")):
    soft, hard = resource.getrlimit(getattr(resource, key))
    fmt = lambda v: "unlimited" if v == resource.RLIM_INFINITY else f"{v / 1e9:.1f} GB"  # noqa: E731
    print(f"  {name:14s} soft {fmt(soft)}  hard {fmt(hard)}")
  for path in (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
  ):
    if os.path.exists(path):
      print(f"  {path}: {open(path).read().strip()}")
  with open("/proc/meminfo") as f:
    for line in f:
      if line.startswith(("MemTotal", "MemAvailable")):
        print("  " + line.strip())

  for task in (
    "Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth",
    "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success",
  ):
    print(f"\n=== {task}")
    subprocess.run(
      [sys.executable, "-c", CHILD, task],
      env={**os.environ, "MUJOCO_GL": "egl"},
      timeout=1800,
    )


if __name__ == "__main__":
  main()
