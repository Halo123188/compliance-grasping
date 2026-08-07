"""Do the finger joints actually stop where the mechanism does?

  uv run python scripts/diag_hand_limits.py

``constants.py`` claims the closing stop is physical: "finger self-collision is
the physical backstop... The XML joint limits (+-0.42 proximal, +-0.55 distal)
are a generous safety cap". A later paragraph in the SAME comment says
self-collision was then turned off to stop a mujoco-warp EPA overflow, leaving
"their joint limits in empty air" as the only stop.

This checks both halves of that claim against the compiled model, and sweeps the
proximal joints closed to find where the two fingers actually interpenetrate --
i.e. the angle the limits ought to encode.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  get_two_finger_hand_robot_cfg,
)
from mjlab.entity import Entity

FINGERS = ("left_1", "left_2", "right_1", "right_2")
TRAINED = {"left_1": -0.745, "left_2": 1.054, "right_1": 0.080, "right_2": -0.471}
HOVER = {"left_1": 0.70, "left_2": 0.0, "right_1": -0.70, "right_2": 0.0}

# Build through Entity, not spec_fn() directly: CollisionCfg (contype/conaffinity,
# condim, friction) is applied as a spec edit at Entity init, and it is exactly
# those fields this script is checking.
model = Entity(get_two_finger_hand_robot_cfg()).spec.compile()
data = mujoco.MjData(model)

print("=== joint ranges (compiled) ===")
for n in FINGERS:
  j = model.joint(n)
  print(f"  {n:9s} range = [{j.range[0]:+.3f}, {j.range[1]:+.3f}]  limited={j.limited}")

print("\n=== finger collider contype/conaffinity ===")
gids = []
for i in range(model.ngeom):
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
  if name.endswith("_col") and ("left" in name or "right" in name):
    gids.append((name, i))
    print(
      f"  {name:14s} contype={model.geom_contype[i]} "
      f"conaffinity={model.geom_conaffinity[i]}"
    )
lg = [i for n, i in gids if n.startswith("left")]
rg = [i for n, i in gids if n.startswith("right")]
ct, ca = model.geom_contype, model.geom_conaffinity
pair_ok = any((ct[a] & ca[b]) or (ct[b] & ca[a]) for a in lg for b in rg)
print(f"  -> left-vs-right pairs can generate contact: {pair_ok}")


def set_pose(q: dict) -> None:
  data.qpos[:] = 0.0
  for n, v in q.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  mujoco.mj_forward(model, data)


gname = {i: n for n, i in gids}


def closest_pair() -> tuple[float, str]:
  """Smallest signed distance between any left and any right finger collider.

  Returned with the pair that achieved it: the proximal knuckles are only 56 mm
  apart at the base and swing toward each other, so the binding contact is often
  NOT the fingertips, and a bare minimum would hide that.
  """
  best, who = np.inf, ""
  for a in lg:
    for b in rg:
      d = mujoco.mj_geomDistance(model, data, a, b, 1.0, np.zeros(6))
      if d < best:
        best, who = d, f"{gname[a]}/{gname[b]}"
  return best, who


sl, sr = model.site("left_pad").id, model.site("right_pad").id


def jaw_gap() -> float:
  """Distance between the two pad sites -- the number the calibration table used.

  The pads are the working surfaces; geom-to-geom distance is a different
  quantity and the two diverge badly once the fingers curl.
  """
  return float(np.linalg.norm(data.site_xpos[sl] - data.site_xpos[sr]))


print("\n=== closing sweep: proximal joints mirrored, distals at 0 ===")
print("  jaw gap = pad site separation; min gap = nearest left/right collider")
crossed = None
for a in np.arange(0.90, -1.61, -0.05):
  set_pose({"left_1": a, "left_2": 0.0, "right_1": -a, "right_2": 0.0})
  g, who = closest_pair()
  if g <= 0.0 and crossed is None:
    crossed = a
  if abs(round(a, 2) * 100) % 20 < 1e-6 or (crossed is not None and a == crossed):
    print(
      f"  left_1={a:+.2f} right_1={-a:+.2f}  jaw gap ={jaw_gap() * 1000:7.1f} mm"
      f"   min gap ={g * 1000:+7.2f} mm  ({who})"
    )
print(f"  -> colliders first touch at left_1 = {crossed:+.2f}")

print("\n=== distal sweep: proximals held at the grip pose (+0.10/-0.10) ===")
print("  which sign curls the fingertip INWARD? pad z is in the base_link frame")
for b in np.arange(1.2, -1.21, -0.2):
  set_pose({"left_1": 0.10, "left_2": b, "right_1": -0.10, "right_2": -b})
  g, who = closest_pair()
  pz = float((data.site_xpos[sl][2] + data.site_xpos[sr][2]) / 2)
  print(
    f"  left_2={b:+.2f} right_2={-b:+.2f}  jaw gap ={jaw_gap() * 1000:7.1f} mm"
    f"   pad z ={pz * 1000:+7.1f} mm   min gap ={g * 1000:+7.2f} mm  ({who})"
  )

print("\n=== the pose the trained policy actually settles in ===")
for label, q in (("trained", TRAINED), ("hover ", HOVER)):
  set_pose(q)
  g, who = closest_pair()
  print(
    f"  {label}: jaw gap ={jaw_gap() * 1000:7.1f} mm   "
    f"min gap ={g * 1000:+7.2f} mm  ({who})"
  )

soft = 0.9  # soft_joint_pos_limit_factor in ARTICULATION
print("\n=== does joint_pos_limits penalise the trained pose? ===")
for n, v in TRAINED.items():
  lo, hi = model.joint(n).range
  print(
    f"  {n:9s} q={v:+.3f}  soft limit = [{lo * soft:+.3f}, {hi * soft:+.3f}]  "
    f"violating={'YES' if not (lo * soft <= v <= hi * soft) else 'no'}"
  )
