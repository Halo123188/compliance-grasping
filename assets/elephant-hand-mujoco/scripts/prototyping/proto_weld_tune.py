"""Tune weld stiffness for sub-0.1mm anchor alignment under movement.

Reuses build() / sweep_input() from proto_weld_stabilize. Sweeps solref time-constant,
solimp, impratio, kp, and per-target settle steps to drive the WELD ANCHOR ALIGNMENT
error (world distance between the two coupler bodies' shared anchor) as low as possible
while staying finite across the input-joint range.

Run: /home/alok/.conda/envs/g313/bin/python scripts/proto_weld_tune.py
"""

from __future__ import annotations

import mujoco
import numpy as np
import proto_weld_stabilize as P  # same dir


def eval_cfg(cfg, settle=120, label=""):
  m = P.build(cfg)
  rows = []
  worst = 0.0
  finite_all = True
  for f in P.FINGERS:
    worlds, residuals, inputs, finite = P.sweep_input(m, f, gravity=True, settle=settle)
    if not worlds or not finite:
      finite_all = False
      rows.append((f["name"], None, None, finite))
      continue
    align = np.array(worlds) * 1000
    rows.append((f["name"], align.max(), align.mean(), finite))
    worst = max(worst, align.max())
  print(f"[{label}] worst-align={worst:.4f}mm finite={finite_all}", flush=True)
  for n, mx, mn, _fin in rows:
    if mx is None:
      print(f"    {n}: DIVERGED", flush=True)
    else:
      print(f"    {n}: max={mx:.4f} mean={mn:.4f} mm", flush=True)
  return worst, finite_all


def main():
  base = dict(
    mass_floor=0.0,
    inertia_floor=5e-5,
    armature=0.01,
    damping=0.3,
    solref=[0.01, 1.0],
    solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
    impratio=10.0,
    timestep=0.001,
    iterations=150,
    ls_iterations=50,
    solver=mujoco.mjtSolver.mjSOL_NEWTON,
    kp=5.0,
    kv=0.5,
    torquescale=1.0,
  )
  print("=== solref time-constant sweep (stiffer weld) ===", flush=True)
  for tc in [0.02, 0.01, 0.005, 0.003, 0.002]:
    eval_cfg(dict(base, solref=[tc, 1.0]), label=f"solref=[{tc},1]")

  print("\n=== impratio sweep (solref=0.005) ===", flush=True)
  for ir in [1.0, 5.0, 20.0, 50.0]:
    eval_cfg(dict(base, solref=[0.005, 1.0], impratio=ir), label=f"impratio={ir}")

  print("\n=== solimp dmax sweep (solref=0.005, impratio=20) ===", flush=True)
  for dmax in [0.999, 0.9999, 0.99999]:
    si = [0.99, dmax, 0.0005, 0.5, 2.0]
    eval_cfg(
      dict(base, solref=[0.005, 1.0], impratio=20.0, solimp=si), label=f"dmax={dmax}"
    )

  print("\n=== timestep sweep (best stiff config) ===", flush=True)
  bestish = dict(
    base, solref=[0.005, 1.0], impratio=20.0, solimp=[0.99, 0.9999, 0.0005, 0.5, 2.0]
  )
  for dt in [0.002, 0.001, 0.0005]:
    eval_cfg(dict(bestish, timestep=dt), label=f"dt={dt}", settle=int(120 * 0.001 / dt))

  print("\n=== longer settle (converged alignment), best config ===", flush=True)
  eval_cfg(bestish, settle=400, label="settle=400")

  # The minimal-inertia-inflation question: 5e-5 floor inflates ALL 15 bodies 3-700x.
  # Prefer ARMATURE (reflected inertia at joints) which is far less invasive. Find the
  # lightest body-inertia floor that still holds sub-0.2mm alignment under movement.
  print(
    "\n=== minimal regularization: inertia_floor sweep w/ armature=0.01 (solref=0.005) ===",
    flush=True,
  )
  stiff = dict(
    base,
    solref=[0.005, 1.0],
    solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
    impratio=1.0,
    armature=0.01,
    damping=0.3,
  )
  for inf in [0.0, 1e-6, 5e-6, 1e-5, 5e-5]:
    eval_cfg(dict(stiff, inertia_floor=inf), label=f"inertia_floor={inf:.0e}")

  print(
    "\n=== armature sweep with inertia_floor=0 (armature-only regularization) ===",
    flush=True,
  )
  for arm in [0.005, 0.01, 0.02, 0.05]:
    eval_cfg(
      dict(stiff, inertia_floor=0.0, armature=arm), label=f"armature={arm} (inf=0)"
    )


if __name__ == "__main__":
  main()
