"""Where does the cube actually touch the finger, versus where the pad site is?

  uv run python scripts/diag_pad_contact.py

``diag_pad_site.py`` shows the site sits on the flat inner face the XML says it
does. That is not the same question as where a 50 mm cube ends up bearing: the
finger keeps going for another 24 mm past that face down to a tapered tip, so
the cube can perfectly well load a part of the finger the site is nowhere near.

The cube is anchored to ``grasp_site``, NOT to the pad sites. Placing it at the
pad midpoint makes the test circular -- it would put the cube wherever the sites
already are and then "discover" that the sites are right. ``grasp_site`` is the
point the ARM drives to the cube centre, so it is the one reference in the hand
that does not depend on the answer.

The distal curl is swept, because the pad tilts as the finger curls and the
contact slides along it. A single curl would fit the site to one pose.
"""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.two_finger_hand.constants import (
  _GRASP_SITE_POS,
  FINGER_DAMPING,
  FINGER_EFFORT_LIMIT,
  FINGER_STIFFNESS,
  HAND_XML,
)

CUBE = 0.050
HOVER = {"left_1": 0.70, "left_2": 0.0, "right_1": -0.70, "right_2": 0.0}
CURLS = (-0.20, -0.30, -0.40, -0.50, -0.60)

spec = mujoco.MjSpec.from_file(str(HAND_XML))

b = spec.worldbody.add_body(name="cube", pos=list(_GRASP_SITE_POS))
cube_geom = b.add_geom(
  type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE / 2] * 3, rgba=[0.9, 0.9, 0.9, 1]
)
cube_geom.name = "cube_geom"

# constants.py excludes these pairs when it builds the real robot; without them
# the palm blocks the proximal links and the jaw cannot close at all.
for a, b_ in (
  ("base_link", "left_1"),
  ("base_link", "right_1"),
  ("left_1", "left_2"),
  ("right_1", "right_2"),
):
  ex = spec.add_exclude()
  ex.bodyname1, ex.bodyname2 = a, b_

model = spec.compile()
# The hand XML's own actuators are kp=1 placeholders that mjlab deletes and
# replaces at load (constants.py). Left as-is the cube simply shoves the fingers
# back open and the test measures nothing, so apply the gains training uses.
for n in HOVER:
  a = model.actuator(n)
  model.actuator_gainprm[a.id, 0] = FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 1] = -FINGER_STIFFNESS
  model.actuator_biasprm[a.id, 2] = -FINGER_DAMPING
  model.actuator_forcerange[a.id] = (-FINGER_EFFORT_LIMIT, FINGER_EFFORT_LIMIT)

cube_gid = model.geom("cube_geom").id
data = mujoco.MjData(model)

print(f"cube 50 mm centred on grasp_site {np.array(_GRASP_SITE_POS) * 1000} mm")
print("closing from hover onto it, proximal driven to +-0.10\n")
print(f"{'curl':>6} {'side':>6} {'n':>3}  contact x/y/z (mm, link frame)")

rows = []
for curl in CURLS:
  grip = {"left_1": 0.10, "left_2": curl, "right_1": -0.10, "right_2": -curl}
  mujoco.mj_resetData(model, data)
  # Start OPEN and close onto the cube. Starting at the grip pose puts the
  # fingers inside the cube and the first step blows them apart.
  for n, v in HOVER.items():
    data.qpos[model.joint(n).qposadr[0]] = v
  for n, v in grip.items():
    data.ctrl[model.actuator(n).id] = v
  for _ in range(4000):
    mujoco.mj_step(model, data)

  for side in ("left", "right"):
    bid = model.body(f"{side}_2").id
    pts = []
    for i in range(data.ncon):
      c = data.contact[i]
      g = {c.geom1, c.geom2}
      if cube_gid not in g:
        continue
      if model.geom_bodyid[(g - {cube_gid}).pop()] != bid:
        continue
      # World -> link frame, so the contact is comparable with the site's pos.
      pts.append(data.xmat[bid].reshape(3, 3).T @ (c.pos - data.xpos[bid]))
    if not pts:
      print(f"{curl:+6.2f} {side:>6} {0:3d}  (no contact)")
      continue
    p = np.array(pts).mean(0) * 1000
    rows.append(p)
    print(
      f"{curl:+6.2f} {side:>6} {len(pts):3d}  {p[0]:+7.2f} {p[1]:+7.2f} {p[2]:+7.2f}"
    )

if rows:
  r = np.abs(np.array(rows))  # fold left/right together; they mirror in x
  print(
    f"\nmean over all curls:  x=+-{r[:, 0].mean():.2f}  y={np.array(rows)[:, 1].mean():+.2f}"
    f"  z={np.array(rows)[:, 2].mean():+.2f} mm"
  )
  print(
    f"spread (max-min):     x={np.ptp(r[:, 0]):.2f}   y={np.ptp(np.array(rows)[:, 1]):.2f}"
    f"   z={np.ptp(np.array(rows)[:, 2]):.2f} mm"
  )
  for side in ("left", "right"):
    s = model.site(f"{side}_pad").pos * 1000
    print(f"current {side}_pad site: ({s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f}) mm")
