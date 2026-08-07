"""Build the two-finger hand's MjSpec from the CAD URDF, not from transcription.

The narrow hand in ``twofinger_arm_cfg`` carries its geometry as hand-copied
constants and its masses as *estimates* -- documented as the least-trusted
numbers in the project. The wide claw shipped with mass properties computed from
the geometry, so there is no reason to retype any of it: this module reads
``hands/<CG_HAND>/hand.urdf`` and emits the same shape of spec
that ``_hand_spec`` builds by hand.

Reading the CAD directly is the point, not a convenience. The stated use is a
live sim2real comparison with the real arm alongside, fitting parameters against
it -- which only means anything if "the sim" and "the CAD" are the same object.
Re-syncing after a hardware revision is then a file copy plus a re-run, and a
divergence is a diff rather than something you discover from a residual.

WHAT IS TAKEN FROM THE URDF
    link mass, centre of mass, and the FULL inertia tensor (the CAD tensors have
    real off-diagonal terms; MuJoCo takes them via ``fullinertia``)
    joint origins, axes, limits, effort and velocity limits
    visual mesh references

WHAT IS NOT
    Collision geometry. The URDF ships a COACD convex decomposition -- 21 to 29
    hulls per link, 151 in total. That is the right representation for a
    single-arm contact study and the wrong one for 1024 parallel envs, where the
    existing hand runs on five boxes. Boxes are FITTED to the wide meshes here
    (``fit_collision_boxes``) rather than inherited, and the fit is checked
    against the mesh bounds it came from.

    Actuator gains. Those are the sim's own tuning (``_FINGER_KP``/``KV``), not a
    CAD property, and the URDF's effort limit is carried over as the force range
    only.
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Assets are PACKAGE DATA, resolved through importlib.resources rather than by
# walking up from __file__. Walking up works from a source checkout and breaks
# the moment the package is installed into a venv, which is exactly what `uv
# sync` does -- so it would have failed only for whoever installed it properly.
#
# One directory per gripper under assets/hands/, selected by CG_HAND.
# See assets/hands/README.md, next to this file.
ASSETS_DIR = _ASSETS = Path(__file__).resolve().parent / "assets"
HANDS_DIR = _ASSETS / "hands"
HAND_NAME = (os.environ.get("CG_HAND") or "wide").lower()
HAND_CAD_DIR = HANDS_DIR / HAND_NAME
MESH_DIR = HAND_CAD_DIR / "meshes"
_URDF = HAND_CAD_DIR / "hand.urdf"


@dataclass(frozen=True)
class Link:
  name: str
  mass: float
  com: tuple[float, float, float]
  # (ixx, iyy, izz, ixy, ixz, iyz) -- MuJoCo's fullinertia order.
  fullinertia: tuple[float, float, float, float, float, float]
  mesh: str | None


@dataclass(frozen=True)
class Joint:
  name: str
  parent: str
  child: str
  origin: tuple[float, float, float]
  rpy: tuple[float, float, float]
  axis: tuple[float, float, float]
  lower: float
  upper: float
  effort: float
  velocity: float


@dataclass(frozen=True)
class HandCad:
  links: dict[str, Link]
  joints: dict[str, Joint]
  source: str

  @property
  def knuckle_x(self) -> float:
    """Half the knuckle spacing, +x side. 0.028 narrow -> 0.035 wide."""
    return abs(self.joints["right_1"].origin[0])

  @property
  def knuckle_yz(self) -> tuple[float, float]:
    j = self.joints["right_1"].origin
    return (j[1], j[2])

  @property
  def prox_len(self) -> float:
    """Knuckle-to-distal-joint distance along the finger (positive)."""
    return abs(self.joints["right_2"].origin[2])


def _f3(node, attr: str, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
  if node is None or node.get(attr) is None:
    return default
  return tuple(float(v) for v in node.get(attr).split())  # type: ignore[return-value]


def load_hand_cad(urdf: Path | None = None) -> HandCad:
  """Parse the vendored wide-claw URDF into plain data."""
  path = Path(urdf) if urdf is not None else _URDF
  if not path.exists():
    raise FileNotFoundError(
      f"{path} is missing. It is vendored from robot-actuator-model; see "
      f"{HAND_CAD_DIR / 'PROVENANCE'} for the commit it came from."
    )
  root = ET.parse(path).getroot()

  links: dict[str, Link] = {}
  for link_el in root.findall("link"):
    inertial = link_el.find("inertial")
    if inertial is None:
      continue
    a = inertial.find("inertia").attrib  # type: ignore[union-attr]
    vis = link_el.find("visual/geometry/mesh")
    links[link_el.get("name")] = Link(  # type: ignore[index]
      name=link_el.get("name"),  # type: ignore[arg-type]
      mass=float(inertial.find("mass").get("value")),  # type: ignore[union-attr,arg-type]
      com=_f3(inertial.find("origin"), "xyz"),
      fullinertia=(
        float(a["ixx"]),
        float(a["iyy"]),
        float(a["izz"]),
        float(a["ixy"]),
        float(a["ixz"]),
        float(a["iyz"]),
      ),
      mesh=Path(vis.get("filename")).stem if vis is not None else None,  # type: ignore[arg-type]
    )

  joints: dict[str, Joint] = {}
  for j in root.findall("joint"):
    lim = j.find("limit")
    joints[j.get("name")] = Joint(  # type: ignore[index]
      name=j.get("name"),  # type: ignore[arg-type]
      parent=j.find("parent").get("link"),  # type: ignore[union-attr,arg-type]
      child=j.find("child").get("link"),  # type: ignore[union-attr,arg-type]
      origin=_f3(j.find("origin"), "xyz"),
      rpy=_f3(j.find("origin"), "rpy"),
      axis=_f3(j.find("axis"), "xyz", (0.0, 0.0, 1.0)),
      lower=float(lim.get("lower")) if lim is not None else 0.0,  # type: ignore[arg-type]
      upper=float(lim.get("upper")) if lim is not None else 0.0,  # type: ignore[arg-type]
      effort=float(lim.get("effort")) if lim is not None else 0.0,  # type: ignore[arg-type]
      velocity=float(lim.get("velocity")) if lim is not None else 0.0,  # type: ignore[arg-type]
    )

  return HandCad(links=links, joints=joints, source=str(path))


def stl_vertices(path: Path) -> np.ndarray:
  """(N, 3) vertices of a BINARY stl, without pulling in trimesh.

  These meshes are 20-40k triangles each and are read at env-build time, so
  this stays a numpy view rather than a full mesh load.
  """
  raw = Path(path).read_bytes()
  n = struct.unpack("<I", raw[80:84])[0]
  recs = np.frombuffer(raw[84 : 84 + n * 50], dtype=np.uint8).reshape(n, 50)
  # Each 50-byte record: 12B normal, 3x12B vertices, 2B attribute.
  return np.concatenate(
    [
      np.frombuffer(recs[:, 12 + k * 12 : 24 + k * 12].tobytes(), dtype="<f4").reshape(
        n, 3
      )
      for k in range(3)
    ]
  ).astype(float)


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
  """(min, max) vertex bounds of a binary stl."""
  v = stl_vertices(path)
  return v.min(axis=0), v.max(axis=0)


def fit_collision_boxes(
  cad: HandCad, horn_z: float = 0.0, bands: int = 4
) -> dict[str, list[tuple]]:
  """Axis-aligned box per link, fitted to its visual mesh bounds.

  Returns ``{link: [(halfsize, pos), ...]}`` in the link's own frame, each
  entry in the form ``_add_collision_box`` takes.

  THE HORN. For the finger links the box's x extent is taken only from
  vertices at ``z <= horn_z``, while y and z still span the whole mesh. Every
  finger link carries a linkage horn above the joint plane that reaches 6.3 mm
  FURTHER INWARD than the gripping face does. A box fitted to the raw bounds
  inherits that 6.3 mm along the entire length of the finger, so the two
  fingers' boxes foul each other 12.6 mm before the pads meet -- the claw
  simply stops, 26 mm short of a 50 mm cube, with no contact against the cube
  at all. The result looks exactly like a policy that has not learned to grip.

  Restricting x to the working region gives x = +-0.0100 on both finger links,
  which is the same half-width the narrow hand's hand-fitted boxes used. The
  horn is then not represented in collision, which is correct for this task:
  it faces the OTHER finger's horn above the joint plane and never the object.

  THE TAPER. One box per finger is not enough for the distal link. Below the
  pad the link narrows into a tip that curves away from the object, and a
  single box squares that taper back out to full width: the box's lowest
  corner ends up 12-16 mm below the real mesh (44.5 mm vs 32.7 mm below the
  pad). That phantom material is what stops the claw -- the fingers foul the
  foam and stall 26 mm short of the cube while the mesh itself would have
  cleared. ``bands`` slices the link in z and fits one box per slice, so the
  taper is approximated by a staircase instead of erased.
  """
  finger_links = {"left_1", "left_2", "right_1", "right_2"}
  tapered = {"left_2", "right_2"}
  out: dict[str, list[tuple]] = {}
  for name, link in cad.links.items():
    if link.mesh is None:
      continue
    stl = MESH_DIR / f"{link.mesh}.stl"
    if not stl.exists():
      continue
    v = stl_vertices(stl)
    if name in finger_links:
      # x from the working region only, so the horn cannot inflate it.
      working = v[v[:, 2] <= horn_z]
      if len(working) < 100:
        raise ValueError(f"horn_z={horn_z} leaves {len(working)} vertices on {name!r}")
    else:
      working = v

    n_bands = bands if name in tapered else 1
    edges = np.linspace(v[:, 2].min(), v[:, 2].max(), n_bands + 1)
    boxes: list[tuple] = []
    for i in range(n_bands):
      z0, z1 = edges[i], edges[i + 1]
      sel = v[(v[:, 2] >= z0) & (v[:, 2] <= z1)]
      if len(sel) < 20:
        continue
      lo, hi = sel.min(axis=0), sel.max(axis=0)
      if name in finger_links:
        w = working[(working[:, 2] >= z0) & (working[:, 2] <= z1)]
        if len(w) >= 20:
          lo[0], hi[0] = w[:, 0].min(), w[:, 0].max()
      # Span the band exactly in z so the staircase has no gaps.
      lo[2], hi[2] = z0, z1
      boxes.append((tuple((hi - lo) / 2.0), tuple((hi + lo) / 2.0)))
    out[name] = boxes
  return out


def pad_marker_pos(
  side: str = "right", face_tol: float = 0.0015, z_max: float = 0.0
) -> tuple[float, float, float]:
  """Centroid of the distal link's INNER gripping face, in that link's frame.

  The pad marker is the point the reward, the straddle gate and the contact
  distance all measure against, so it has to sit on the surface that actually
  touches the cube -- not the link's centre, and not a number carried over
  from a different finger.

  It is MEASURED, not assumed: take the vertices within ``face_tol`` of the
  innermost x, and return their centroid. The wide distal link is asymmetric
  in x (it protrudes inward toward the object) and deeper in y than the narrow
  one, so both of those offsets would be wrong if transcribed.

  Two things here are easy to get wrong, and both were, before this was
  measured properly rather than reasoned about:

  SIGN. The knuckles sit at +-x and the fingers grip INWARD toward the centre
  line, so the right finger (at +x) touches the cube with its x-MINIMUM face
  and the left is mirrored. Inverted, the markers land on the OUTSIDE of the
  claw, where the reported aperture is the claw's outer width and does not
  respond to the proximal joint at all.

  THE HORN. The distal link's extreme inner x is NOT the pad -- it is the
  linkage horn, a tab that protrudes 6 mm further inward at the top of the
  link (z +0.008..+0.024 on the wide claw), above the joint plane and facing
  the other finger's horn, never the object. Taking the global extremum puts
  the marker on that tab, 40 mm from the real contact patch. ``z_max`` cuts
  the search to the finger's working length below the joint plane, where the
  broad flat pad face is.
  """
  v = stl_vertices(MESH_DIR / f"{side}_2.stl")
  working = v[v[:, 2] <= z_max]
  if len(working) < 100:
    raise ValueError(
      f"z_max={z_max} leaves only {len(working)} vertices on "
      f"the {side} distal link; check the mesh orientation"
    )
  if side == "right":
    inner_x = float(working[:, 0].min())
    face = working[working[:, 0] <= inner_x + face_tol]
  else:
    inner_x = float(working[:, 0].max())
    face = working[working[:, 0] >= inner_x - face_tol]
  if len(face) < 3:
    raise ValueError(
      f"face_tol={face_tol} caught only {len(face)} vertices "
      f"on the {side} inner face; the mesh may be rotated"
    )
  return (inner_x, float(face[:, 1].mean()), float(face[:, 2].mean()))


# --- fingertip reference, carried from the 56 mm claw and CHECKED ------------ #
# The 56 mm claw's left_pad sits at (1.33, 18.85, -44.00) mm in the distal
# link's frame. Every mjlab manipulation constant -- PAD_SEP_AT_GRASP above all
# -- is calibrated against that exact point, so this claw uses it rather than
# re-deriving one, and the transfer is VERIFIED rather than assumed.
#
# It is legitimate because the two claws' distal links are the same part.
# Measured on both meshes: identical bounds (x -10..+10, y -3.3..29.7,
# z -55..+8), the same flat pad at inner x = 10.00 over z -20..-32, and the same
# taper below it (inner x at z -36 / -40 / -44: 6.91 / 4.62 / 1.33 on the 56 mm
# claw, 6.66 / 4.52 / 0.04 on this one). The 70 mm knuckle spacing lives in
# base_link, not in the finger.
#
# The check that settles it is point-to-SURFACE distance, not point-to-vertex:
# the mesh is sparse at the tip (29 verts in a 4 mm band), so a nearest-vertex
# figure says nothing -- a point can sit dead on a large triangle and still be
# 7 mm from its corners.
#
#     the 56 mm site against its OWN mesh      0.387 mm off the surface
#     the same point against THIS mesh         0.404 mm off the surface
#
# 0.017 mm apart. The point occupies the same position relative to the gripping
# surface on both claws, so the constant transfers exactly.
#
# RE-DERIVING IT INDEPENDENTLY IS WORSE, which is why this is not measured from
# scratch. Taking the inner-surface x from a band around z = -44 lands at
# x = 0.037, which is 0.686 mm off the surface -- further out than either claw's
# own convention -- and 1.3 mm from the tuned point, which would move
# PAD_SEP_AT_GRASP by 2.6 mm. The band picks up vertices from deeper z where the
# taper has already moved inboard.
TIP_SITE_LINK_POS = (0.00133, 0.01885, -0.04400)
# How far the reference is allowed to sit off this claw's surface before the
# carry-over stops being justified. 0.404 mm measured; 1.5 mm leaves room for a
# mesh re-export without hiding a finger that has actually changed shape.
TIP_SITE_MAX_STANDOFF = 0.0015


def stl_triangles(path: Path) -> np.ndarray:
  """(N, 3, 3) triangles of a BINARY stl."""
  raw = Path(path).read_bytes()
  n = struct.unpack("<I", raw[80:84])[0]
  recs = np.frombuffer(raw[84 : 84 + n * 50], dtype=np.uint8).reshape(n, 50)
  return np.stack(
    [
      np.frombuffer(recs[:, 12 + k * 12 : 24 + k * 12].tobytes(), dtype="<f4").reshape(
        n, 3
      )
      for k in range(3)
    ],
    axis=1,
  ).astype(float)


def distance_to_surface(pt: np.ndarray, tri: np.ndarray) -> float:
  """Min distance from ``pt`` to the triangle soup ``tri`` (N, 3, 3).

  Point-to-TRIANGLE with a barycentric clamp, not point-to-vertex. On a mesh
  with large triangles the two differ by millimetres, and the tip of this
  finger is exactly that mesh.
  """
  a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
  ab, ac, ap = b - a, c - a, pt - a
  d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
  bp = pt - b
  d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
  cp = pt - c
  d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
  va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
  den = np.where((va + vb + vc) == 0.0, 1e-12, va + vb + vc)
  v, w = np.clip(vb / den, 0.0, 1.0), np.clip(vc / den, 0.0, 1.0)
  s = v + w
  # Degenerate triangles give s == 0; divide by 1 there rather than warn. They
  # cannot be the nearest face anyway -- a zero-area triangle's projection is
  # its own vertex, which some real triangle also carries.
  safe = np.where(s > 0.0, s, 1.0)
  v, w = np.where(s > 1, v / safe, v), np.where(s > 1, w / safe, w)
  proj = a + v[:, None] * ab + w[:, None] * ac
  return float(np.linalg.norm(proj - pt, axis=1).min())


def tip_marker_pos(side: str = "right") -> tuple[float, float, float]:
  """FINGERTIP reference point on the distal link, in that link's frame.

  NOT the gripping face -- ``pad_marker_pos`` is that, and the two are 22 mm
  apart along the finger with the tip point sitting BEHIND the gripping face.
  Both exist because two reward sets were calibrated against different
  references:

    * ``pad_marker_pos`` is where a 50 mm cube actually BEARS. This repo's own
      rewards use it, and the aperture law ``sep(q) = 50.1 + 134.0 q`` was
      fitted to marker bodies placed there, so moving it invalidates that fit
      and CONTACT_DIST with it.
    * ``tip_marker_pos`` is the point the mjlab manipulation rewards were
      tuned against on the 56 mm claw -- see TIP_SITE_LINK_POS for why the
      constant transfers and what was measured to establish it.

  THE TIP POINT IS NOT A CONTACT POINT and nothing may treat it as one. The
  flat pad ends at z = -31.5; below that the finger tapers hard, so this point
  sits several mm behind the gripping face and the pair straddles the cube
  with clearance on both sides instead of touching it. Anything comparing this
  separation against an object width must use a measured value, never
  2 * cube_half.

  RAISES if the point has drifted off this claw's surface -- a hardware
  revision that reshapes the finger should fail here rather than silently
  place the reward's reference in mid-air.
  """
  x, y, z = TIP_SITE_LINK_POS
  pos = (x if side == "left" else -x, y, z)
  off = distance_to_surface(np.array(pos), stl_triangles(MESH_DIR / f"{side}_2.stl"))
  if off > TIP_SITE_MAX_STANDOFF:
    raise ValueError(
      f"tip reference sits {off * 1000:.3f} mm off the {side} distal "
      f"link's surface (limit {TIP_SITE_MAX_STANDOFF * 1000:.1f} mm). The "
      f"finger has changed shape; re-derive TIP_SITE_LINK_POS against the "
      f"new mesh instead of carrying the 56 mm claw's value."
    )
  return pos
