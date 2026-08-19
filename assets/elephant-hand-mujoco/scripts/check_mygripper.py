"""Load check: Rizon 4S + mygripper_H100_R ("elephant") hand via MjSpec.attach().

Mirrors the combination-script pattern from assets/robots/allegro_arm_cfg.py:
load the arm + the hand URDF as MjSpec, add a hand_mount site at link7, attach
the hand, compile, step. Pure-mujoco (no mjlab/viser) so it gives fast feedback
on mesh/name/compile issues before we wire up the interactive viewer.

Run:  python scripts/check_mygripper.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MENAGERIE = Path(os.environ.get("MJ_MENAGERIE", str(Path.home() / "mujoco_menagerie")))
ARM_XML = MENAGERIE / "flexiv_rizon4s" / "flexiv_rizon4s.xml"
HAND_URDF = ROOT / "assets/robots/mygripper_H100_R/robot.urdf"


def _names(model, objtype, n) -> list[str]:
  return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


def main() -> None:
  print(f"ARM_XML  : {ARM_XML}  exists={ARM_XML.exists()}")
  print(f"HAND_URDF: {HAND_URDF}  exists={HAND_URDF.exists()}")

  # 1) Hand alone — isolates mesh/name/compile problems.
  print("\n=== hand alone ===")
  hand = mujoco.MjSpec.from_file(str(HAND_URDF))
  hm = hand.compile()
  print(
    f"nq={hm.nq} nv={hm.nv} nu={hm.nu} nbody={hm.nbody} njnt={hm.njnt} nmesh={hm.nmesh}"
  )
  print("bodies:", _names(hm, mujoco.mjtObj.mjOBJ_BODY, hm.nbody))
  print("joints:", _names(hm, mujoco.mjtObj.mjOBJ_JOINT, hm.njnt))

  # 2) Combined arm + hand via attach (the real integration).
  print("\n=== arm + hand (attach to link7) ===")
  arm = mujoco.MjSpec.from_file(str(ARM_XML))
  for b in arm.bodies:
    if b.name == "link7":
      s = b.add_site()
      s.name = "hand_mount"
      s.pos = [0.0, 0.0, 0.095]
      s.quat = [1.0, 0.0, 0.0, 0.0]
      break
  hand2 = mujoco.MjSpec.from_file(str(HAND_URDF))
  arm.attach(hand2, suffix="_h", site="hand_mount")
  for k in list(arm.keys):
    arm.delete(k)
  m = arm.compile()
  d = mujoco.MjData(m)
  print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} njnt={m.njnt}")
  attached = [
    n for n in _names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody) if "base_step" in (n or "")
  ]
  print("attached hand root body name(s):", attached)
  print("actuators:", _names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu))

  for _ in range(50):
    mujoco.mj_step(m, d)
  print("after 50 steps: qpos finite =", bool(np.all(np.isfinite(d.qpos))))
  print("OK")


if __name__ == "__main__":
  main()
