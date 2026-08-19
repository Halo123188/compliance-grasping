"""Confirm clean 1-DOF articulation + robustness of the chosen weld config (hand-alone).

For the FINAL config (armature-only regularization, stiff weld) check:
  (a) sweeping the INPUT loop joint moves the fingertip monotonically (clean 1-DOF, no flop):
      net tip travel ~= path length.
  (b) sweeping the BASE swing joint also stays stable & weld-aligned (base is independent).
  (c) the 3 passive loop joints actually move (loop is coupled, not frozen).

Run: /home/alok/.conda/envs/g313/bin/python scripts/proto_weld_articulate.py
"""

from __future__ import annotations

import mujoco
import numpy as np
import proto_weld_stabilize as P

FINAL = dict(
  mass_floor=0.0,
  inertia_floor=0.0,
  armature=0.01,
  damping=0.3,
  solref=[0.002, 1.0],
  solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
  impratio=1.0,
  timestep=0.001,
  iterations=150,
  ls_iterations=50,
  solver=mujoco.mjtSolver.mjSOL_NEWTON,
  kp=5.0,
  kv=0.5,
  torquescale=1.0,
)


def tip_track(m, f, drive_jn, lo, hi, n=25, settle=120):
  """Drive joint `drive_jn` across [lo,hi] rel to assembly; record tip path + passive motion."""
  d = mujoco.MjData(m)
  q0 = P.q_assembly(m)
  d.qpos[:] = q0
  mujoco.mj_forward(m, d)
  for ff in P.FINGERS:
    for jn in (ff["base"], ff["input"]):
      ai = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jn}".replace(" ", "_")
      )
      d.ctrl[ai] = d.qpos[P.jadr(m, jn)]
  for _ in range(100):
    mujoco.mj_step(m, d)
  ai = mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{drive_jn}".replace(" ", "_")
  )
  tip = P.bid(m, f["tip"])
  b1, b2 = P.bid(m, f["b1"]), P.bid(m, f["b2"])
  pos_rel, _ = P.relpose(f["b1"], f["b2"])
  j0 = d.qpos[P.jadr(m, drive_jn)]
  sweep = np.linspace(j0 + lo, j0 + hi, n)
  tips, aligns, passive_ranges = (
    [],
    [],
    {j: [] for j in f["loop"] if j not in (f["base"], f["input"])},
  )
  finite = True
  for tgt in sweep:
    d.ctrl[ai] = tgt
    for _ in range(settle):
      mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
      finite = False
      break
    tips.append(d.xpos[tip].copy())
    R1 = d.xmat[b1].reshape(3, 3)
    aligns.append(np.linalg.norm((d.xpos[b1] + R1 @ pos_rel) - d.xpos[b2]) * 1000)
    for j in passive_ranges:
      passive_ranges[j].append(d.qpos[P.jadr(m, j)])
  tips = np.array(tips)
  if len(tips) >= 2:
    net = np.linalg.norm(tips[-1] - tips[0]) * 1000
    path = np.sum(np.linalg.norm(np.diff(tips, axis=0), axis=1)) * 1000
  else:
    net = path = 0.0
  pr = {j: (max(v) - min(v)) for j, v in passive_ranges.items()}
  return net, path, max(aligns) if aligns else float("inf"), pr, finite


def main():
  m = P.build(FINAL)
  print(
    f"FINAL config: armature={FINAL['armature']} damping={FINAL['damping']} "
    f"inertia_floor={FINAL['inertia_floor']} mass_floor={FINAL['mass_floor']} "
    f"solref={FINAL['solref']} Newton/{FINAL['iterations']} dt={FINAL['timestep']}",
    flush=True,
  )
  print(f"nq={m.nq} nu={m.nu} neq={m.neq}\n", flush=True)

  print("=== drive INPUT loop joint (should be clean 1-DOF: net~=path) ===", flush=True)
  for f in P.FINGERS:
    net, path, al, pr, fin = tip_track(m, f, f["input"], -0.6, 1.2)
    ratio = net / path if path else 0
    pr_str = " ".join(f"{j.split()[-1]}={np.rad2deg(v):.1f}deg" for j, v in pr.items())
    print(
      f"  {f['name']}: net={net:.1f} path={path:.1f} mm (net/path={ratio:.3f}) "
      f"align_max={al:.4f}mm finite={fin}  passive_range[{pr_str}]",
      flush=True,
    )

  print(
    "\n=== drive BASE swing joint (independent DOF, weld must stay aligned) ===",
    flush=True,
  )
  for f in P.FINGERS:
    net, path, al, pr, fin = tip_track(m, f, f["base"], -0.5, 0.5)
    print(
      f"  {f['name']}: base-swing tip net={net:.1f} mm align_max={al:.4f}mm finite={fin}",
      flush=True,
    )


if __name__ == "__main__":
  main()
