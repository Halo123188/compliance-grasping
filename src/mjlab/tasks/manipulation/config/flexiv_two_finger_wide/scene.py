"""The measured bench: table, foam work surface, wall, and the 50 mm cube.

Ported verbatim from cubegrasp-env
(``src/cubegrasp_tf/tasks/cube_grasp/config/twofinger/env_cfgs.py``), which is
the upstream of record for the geometry. Only the imports changed. Re-port
rather than re-derive if that file moves -- every number here was measured on
the real bench on 2026-08-04 and the comments say what against.

The one structural difference from the old ``flexiv_two_finger`` scene: the
table top is no longer the surface the cube rests on. The bench carries 50 mm of
foam, cut around the arm's base plate and the camera-mount foot, so

    MOUNT_SURFACE_Z  0.350   the arm's mounting height (table top)
    WORK_SURFACE_Z   0.400   foam top -- what the cube rests on
    RESTING_Z        0.425   cube centre at rest

Anything that used to say ``TABLE_H`` has to pick one of those three
deliberately. Conflating them puts the cube 50 mm inside the foam.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (
  BASE_PLATE_HALF,
  MOUNT_SURFACE_Z,
  OBJECT_BIT,
  camera_mount_xy_aabb,
)

# --- geometry ---------------------------------------------------------------
CUBE_HALF = 0.025  # 50 mm cube
# White cube against the uniformly blue claw (_CLAW_BLUE in arm_cfg). Not pure
# white: 1.0 clips against the specular highlight, so a face turned to the light
# saturates and the cube's edges stop being findable exactly when it is nearest
# the camera. 0.93 keeps headroom for the highlight.
CUBE_RGBA = (0.93, 0.93, 0.93, 1.0)
# The arm's base plate is bolted to this same top surface, so the table height is
# owned by the robot cfg -- one source of truth for both.
TABLE_TOP_Z = MOUNT_SURFACE_Z

# Manipulation centre, pinned by the D435's 200 mm minimum range rather than by
# reach: aim_camera puts the nearest cube corner at 205.3 mm and the worst frame
# coordinate over the whole jitter box at 0.646 of half-frame, so 0.00% of the
# 22869-pose envelope is blind or out of frame. The optical axis would centre the
# cube way in at 0.3777 and going there costs that 5 mm of range margin.
#
# The three arm poses in arm_cfg.py do NOT follow this automatically -- they are
# solved through the physics. Re-run the pose solver whenever it changes.
TABLE_X = 0.46

# --- the real bench (MEASURED 2026-08-04) ------------------------------------
# Kept in inches with the conversion visible, because that is how they were taken
# and a silently-converted number is unauditable.
_IN = 0.0254
TABLE_LONG = 63.00 * _IN  # 1.6002 m, along X -- the axis the arm reaches
TABLE_SHORT = 31.75 * _IN  # 0.8065 m, along Y -- across the bench
PLATE_TO_EDGE = 5.70 * _IN  # 0.1448 m, base plate near edge to the table edge

# Table edge to the wall face. WAS 16.30 in (0.4140 m); the bench has since been
# moved and the table now stands FLUSH against the wall, so the default is 0.
# Left as a knob purely so a checkpoint trained on the old bench can be evaluated
# where it was trained (CG_WALL_GAP=0.414). The gap is not a thing to tune.
CG_WALL_GAP = float(os.environ.get("CG_WALL_GAP") or "0.0")
WALL_TO_TABLE = CG_WALL_GAP

# The LONG side runs along X, the direction the arm reaches. The arm is centred
# across the bench at y = 0 with its base plate set back PLATE_TO_EDGE from the
# near edge.
_TABLE_X_SPAN = (
  -(BASE_PLATE_HALF[0] + PLATE_TO_EDGE),
  -(BASE_PLATE_HALF[0] + PLATE_TO_EDGE) + TABLE_LONG,
)
_TABLE_Y_SPAN = (-TABLE_SHORT / 2.0, TABLE_SHORT / 2.0)
_TABLE_CENTER_X = sum(_TABLE_X_SPAN) / 2.0  # 0.325
_TABLE_HALF = (
  (_TABLE_X_SPAN[1] - _TABLE_X_SPAN[0]) / 2.0,  # 0.575
  0.40,
  TABLE_TOP_Z / 2.0,
)

# Cube spawn jitter, CLAMPED so the cube can never spawn overhanging the bench.
# Derived rather than hardcoded so it follows the table and TABLE_X instead of
# silently going stale the next time either moves.
_SPAWN_EDGE_MARGIN = 0.005  # 5 mm of table under the far face
SPAWN_X_JITTER = min(0.08, _TABLE_X_SPAN[1] - CUBE_HALF - _SPAWN_EDGE_MARGIN - TABLE_X)
SPAWN_Y_JITTER = min(0.10, _TABLE_Y_SPAN[1] - CUBE_HALF - _SPAWN_EDGE_MARGIN)

# --- the wall ----------------------------------------------------------------
# ON by default since the bench was moved flush against it. It changes the depth
# background and so invalidates comparison against any checkpoint trained without
# it -- that is the cost, and it is worth paying, because the wall face now sits
# AT the table edge instead of 414 mm behind it, which makes it the dominant
# thing behind the cube in every frame. A default that omits it is not "the
# neutral configuration", it is a scene that does not exist.
CG_WALL = (os.environ.get("CG_WALL") or "1") not in ("0", "")
CG_WALL_SIDE = os.environ.get("CG_WALL_SIDE") or "far"  # far/near (x) | left/right (y)
_WALL_THICK = 0.05
_WALL_HEIGHT = 2.5  # floor to well above frame
_WALL_RGBA = (0.72, 0.70, 0.66, 1.0)  # matte off-white
_WALL_EDGES = {"far": (0, +1), "near": (0, -1), "left": (1, +1), "right": (1, -1)}
if CG_WALL_SIDE not in _WALL_EDGES:
  raise ValueError(
    f"CG_WALL_SIDE={CG_WALL_SIDE!r}; expected one of {sorted(_WALL_EDGES)}"
  )
_WALL_AXIS, _WALL_SIGN = _WALL_EDGES[CG_WALL_SIDE]
_span = _TABLE_X_SPAN if _WALL_AXIS == 0 else _TABLE_Y_SPAN
WALL_FACE = _span[1] + WALL_TO_TABLE if _WALL_SIGN > 0 else _span[0] - WALL_TO_TABLE
_WALL_CENTER = WALL_FACE + _WALL_SIGN * _WALL_THICK / 2.0
_WALL_RUN = (_TABLE_Y_SPAN[1] if _WALL_AXIS == 0 else _TABLE_X_SPAN[1]) + 0.6

# --- foam layer --------------------------------------------------------------
# 50 mm of foam over the whole bench EXCEPT a cutout around the arm's hardware
# (base plate AND camera-mount foot), so the arm stays bolted at MOUNT_SURFACE_Z
# while everything it manipulates sits 50 mm higher. Modelled as rigid boxes: a
# real deformable layer needs flex/soft contacts mujoco-warp does not do at this
# throughput, and what matters for the policy is that the work surface MOVED, not
# that it yields.
CG_FOAM_H = float(os.environ.get("CG_FOAM_H") or "0.05")

# COMPLIANT foam. The layer is 30 kg/m^3 open-cell foam and was nonetheless given
# near-rigid contact parameters, which was harmless while nothing touched it. The
# WIDE claw changed that: its gripping face sits 23-33 mm above the fingertip and
# the cube centre is only 25 mm above the work surface, so gripping the cube ON
# ITS CENTRE LINE needs the tips a few mm below the surface. Against a rigid box
# that is simply blocked; against real foam it is what actually happens.
#
# MEASURED (scripted expert, wide claw, 150 mm lift):
#     rigid foam   grip lands 9.3 mm above the cube centre, holds
#     soft  foam   grip lands 2.9 mm above the cube centre, holds
# Default ON here, unlike upstream: upstream defaults it off to preserve the
# contact model old narrow-hand checkpoints trained under, and this package has
# no such checkpoints to preserve. The wide claw is the only hand it builds.
CG_FOAM_SOFT = (os.environ.get("CG_FOAM_SOFT") or "1") not in ("0", "")
_FOAM_SOLREF_SOFT = np.array([0.05, 1.0])  # slow, soft response
_FOAM_SOLIMP_SOFT = np.array([0.0, 0.35, 0.02, 0.5, 2.0])  # low impedance -> compresses
_FOAM_RGBA = (0.29, 0.31, 0.34, 1.0)  # dark neutral, against the white cube
_FOAM_DENSITY = 30.0  # kg/m^3; static, so purely cosmetic
_FOAM_CLEARANCE = 0.003  # knife-cut slop around the hardware

# Near-rigid contact for the work surfaces. Default solimp lets the solver sink
# the cube cm-deep into the table under arm-scale forces.
HARD_SOLREF = np.array([0.01, 1.0])
HARD_SOLIMP = np.array([0.98, 0.999, 0.001, 0.5, 2.0])


def _foam_cutout() -> tuple[float, float, float, float]:
  """(x0, x1, y0, y1) of the hole the foam is cut with, in world xy.

  The plate footprint alone is NOT enough. The camera mount is bolted to the
  plate but its foot overhangs the plate edge by 7 mm on the +x side and sits
  inside the foam's z-slab, so cutting only to the plate buried part of the
  mount in foam -- silently, since both are welded to the world and MuJoCo
  filters the pair.
  """
  p = BASE_PLATE_HALF[0]
  mx0, mx1, my0, my1 = camera_mount_xy_aabb(TABLE_TOP_Z, TABLE_TOP_Z + CG_FOAM_H)
  c = _FOAM_CLEARANCE
  return (min(-p, mx0 - c), max(p, mx1 + c), min(-p, my0 - c), max(p, my1 + c))


# The surface the CUBE rests on: table top plus any foam. Every task height
# derives from this rather than from TABLE_TOP_Z -- TABLE_TOP_Z is now only the
# arm's mounting height, and conflating the two puts the cube inside the foam.
WORK_SURFACE_Z = TABLE_TOP_Z + CG_FOAM_H
RESTING_Z = WORK_SURFACE_Z + CUBE_HALF  # cube centre at rest


def get_cube_spec() -> mujoco.MjSpec:
  """Rigid free-jointed 50 mm cube, graspable by the two-bit finger scheme."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  geom = body.add_geom(
    name="cube",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(CUBE_HALF,) * 3,
    mass=0.06,
    rgba=CUBE_RGBA,
  )
  # condim=4: torsional friction so the cube can't spin out of the pinch.
  geom.condim = 4
  geom.friction = np.array([1.0, 0.05, 0.001])
  geom.priority = 1  # cube friction/sol params win its pairs
  geom.contype = OBJECT_BIT  # 4: fingers (ca=6) see it
  geom.conaffinity = 3  # 1|2: collides with table/arm + fingers
  geom.solref = HARD_SOLREF
  geom.solimp = HARD_SOLIMP
  return spec


def get_table_spec() -> mujoco.MjSpec:
  """Static table + foam layer + wall, as ONE body named ``table``.

  contype 5 / conaffinity 3 so the FINGERS also collide with it (they can't
  sweep through the top) alongside arm + cube. The slab reaches back under the
  arm's base plate; table-vs-plate and table-vs-arm-base need no explicit
  exclude, since both are welded to the world and MuJoCo filters those pairs.

  ONE BODY, several geoms -- upstream gives the table, each foam strip and the
  wall a body of its own, and that shape does not survive contact with mjlab's
  ContactSensor. A sensor's SECONDARY match must resolve to a single name, and
  under the default ``secondary_policy="first"`` a pattern that matches six
  bodies silently keeps ONE of them. That is how run 55060 ended up with a
  fingertip-vs-surface penalty wired to the table slab while the fingertips were
  pressing into foam: the sensor was correct, complete and reporting nothing.
  Everything here is static and welded to the world, so merging them costs no
  dynamics -- MuJoCo already filtered every pair among them.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.pos = np.array([_TABLE_CENTER_X, 0.0, _TABLE_HALF[2]])
  bx, by, bz = body.pos
  geom = body.add_geom(
    name="table",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=_TABLE_HALF,
    mass=50.0,
    rgba=(0.45, 0.32, 0.22, 1.0),
  )
  geom.contype = 5
  geom.conaffinity = 3
  geom.solref = HARD_SOLREF
  geom.solimp = HARD_SOLIMP

  # Foam, as four strips framing the cutout around the arm. A box cannot have a
  # hole, so the layer is tiled: back / front spanning full y, then the two side
  # pieces filling the remaining y on either side of the cutout. Inner faces sit
  # exactly on the cutout boundary -- coincident, but foam, table, plate and arm
  # base are all welded to the world, so MuJoCo filters every one of those pairs
  # and no contact is generated. Leaving a gap instead would open a slot the cube
  # could drop into.
  if CG_FOAM_H > 0.0:
    cx0, cx1, cy0, cy1 = _foam_cutout()
    (x0, x1), (y0, y1) = _TABLE_X_SPAN, _TABLE_Y_SPAN
    strips = (
      ("back", (x0, cx0), (y0, y1)),
      ("front", (cx1, x1), (y0, y1)),
      ("right", (cx0, cx1), (y0, cy0)),
      ("left", (cx0, cx1), (cy1, y1)),
    )
    for name, (a0, a1), (b0, b1) in strips:
      half = ((a1 - a0) / 2.0, (b1 - b0) / 2.0, CG_FOAM_H / 2.0)
      fg = body.add_geom(
        name=f"foam_{name}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=(
          (a0 + a1) / 2.0 - bx,
          (b0 + b1) / 2.0 - by,
          TABLE_TOP_Z + CG_FOAM_H / 2.0 - bz,
        ),
        mass=_FOAM_DENSITY * 8.0 * half[0] * half[1] * half[2],
        rgba=_FOAM_RGBA,
      )
      fg.contype = 5
      fg.conaffinity = 3
      if CG_FOAM_SOFT:
        fg.solref = _FOAM_SOLREF_SOFT
        fg.solimp = _FOAM_SOLIMP_SOFT
      else:
        fg.solref = HARD_SOLREF
        fg.solimp = HARD_SOLIMP

  if CG_WALL:
    # A slab, not a plane: the student SEES this, so it needs a real face at a
    # real distance. Static and welded to the world, so it never enters the
    # dynamics -- it exists to be seen, and to stop an arm commanded past the
    # bench from sailing on through empty space.
    pos = [0.0, 0.0, _WALL_HEIGHT / 2.0]
    pos[_WALL_AXIS] = _WALL_CENTER
    half = [_WALL_RUN, _WALL_RUN, _WALL_HEIGHT / 2.0]
    half[_WALL_AXIS] = _WALL_THICK / 2.0
    wg = body.add_geom(
      name="wall",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=tuple(half),
      pos=(pos[0] - bx, pos[1] - by, pos[2] - bz),
      mass=500.0,
      rgba=_WALL_RGBA,
    )
    wg.contype = 5
    wg.conaffinity = 3
  return spec
