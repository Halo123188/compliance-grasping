"""Derive physically-plausible per-joint ROM limits for the mygripper_H100_R
("elephant" hand) MECHANICALLY in the WELDED sim, anchored to the only 2 real specs
(aperture 0-130 mm, velocity 60 deg/s). Headless only (no viser / no port bind).

Outputs:
  1. CURL feasibility: sweep each finger's input joint both directions from assembly;
     stop where weld alignment spikes (>1 mm) or qpos goes non-finite = singularity.
  2. APERTURE calibration: for each input angle in the feasible band, compute aperture
     (circumscribed-circle diameter through the 3 fingertip geom CENTERS, the pad
     surface). Find the curl angle where aperture ~= 130 mm (open) and ~= 0 mm (closed).
  3. BASE swing: sweep each base joint; detect self-collision via a SELECTIVE geometric
     mesh-proximity check (fingertip/finger geoms vs palm/base/other-finger geoms;
     internal linkage overlaps ignored).
  4. PASSIVE ROM: record min/max each passive joint reaches over the feasible curl range.

Run: PYTHONPATH=src /home/alok/.conda/envs/g313/bin/python scripts/derive_joint_limits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from my_mjlab_project import mygripper_arm_viewer as V  # noqa: E402

R2D = 180.0 / np.pi
WELD_THRESH_MM = 1.0  # four-bar feasibility: weld alignment spike above this = locked


def _jadr(mjm, jn):
  ji = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
  return int(mjm.jnt_qposadr[ji])


def _aid(mjm, jn):
  return V._attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)


def _tip_geoms(mjm):
  out = {}
  for f in V.WELD_FINGERS:
    bi = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["tip"])
    gi = [g for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == bi][0]
    out[f["name"]] = gi
  return out


def _tip_pts(mjm, mjd, tg):
  return {n: mjd.geom_xpos[gi].copy() for n, gi in tg.items()}


def _circumdiam_mm(pts):
  """Diameter of the circumscribed circle through the 3 fingertip contact points =
  the largest cylinder/sphere the 3 pads can cage = the gripping aperture proxy."""
  P = np.array([pts["F1"], pts["F2"], pts["F3"]])
  a = np.linalg.norm(P[1] - P[2])
  b = np.linalg.norm(P[0] - P[2])
  c = np.linalg.norm(P[0] - P[1])
  s = (a + b + c) / 2
  area = max(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0)), 1e-12)
  return 2.0 * (a * b * c / (4 * area)) * 1000.0


def _reset_assembly(mjm, mjd):
  mujoco.mj_resetData(mjm, mjd)
  mjd.qpos[:7] = [0, 0, 0, 1.57, 0, 0, 0]
  mjd.ctrl[:7] = [0, 0, 0, 1.57, 0, 0, 0]
  for jn, val in V.WELD_ASSEMBLY.items():
    mjd.qpos[_jadr(mjm, jn)] = val
  mujoco.mj_forward(mjm, mjd)
  for f in V.WELD_FINGERS:
    for jn in (f["base"], f["input"]):
      mjd.ctrl[_aid(mjm, jn)] = mjd.qpos[_jadr(mjm, jn)]


def _settle(mjm, mjd, n=120):
  for _ in range(n):
    mujoco.mj_step(mjm, mjd)


# ---------------------------------------------------------------------------
# 1+2. CURL feasibility + aperture mapping (sweep one finger's input at a time,
# others held at assembly, so the per-finger aperture contribution is isolated).
# ---------------------------------------------------------------------------
def sweep_curl(mjm, mjd, tg, lo=-1.6, hi=1.6, n=41, settle=150):
  """For each finger, sweep input from assembly+lo to assembly+hi (others held at
  assembly). Record (input_abs, aperture_mm, weld_max_mm, finite). Returns dict."""
  print("\n===== 1+2. CURL feasibility + aperture (per-finger sweep) =====", flush=True)
  res = {}
  for f in V.WELD_FINGERS:
    _reset_assembly(mjm, mjd)
    _settle(mjm, mjd, 150)
    inp0 = mjd.qpos[_jadr(mjm, f["input"])]
    ai = _aid(mjm, f["input"])
    rows = []
    # sweep symmetric outward from assembly so we catch the lock in each direction
    targets = np.linspace(inp0 + lo, inp0 + hi, n)
    # go from assembly up, then reset and go down, to avoid hysteresis past a lock
    for direction in ("up", "down"):
      _reset_assembly(mjm, mjd)
      _settle(mjm, mjd, 150)
      seq = (
        [t for t in targets if t >= inp0]
        if direction == "up"
        else [t for t in targets if t < inp0][::-1]
      )
      for tgt in seq:
        mjd.ctrl[ai] = tgt
        for _ in range(settle):
          mujoco.mj_step(mjm, mjd)
        fin = bool(np.all(np.isfinite(mjd.qpos)))
        al = V.weld_alignment_mm(mjm, mjd)
        wmax = max(al.values()) if al else 0.0
        if not fin:
          rows.append((float(tgt), float("nan"), float("inf"), False))
          break
        ap = _circumdiam_mm(_tip_pts(mjm, mjd, tg))
        rows.append((float(mjd.qpos[_jadr(mjm, f["input"])]), ap, wmax, True))
        if wmax > WELD_THRESH_MM:
          break
    rows.sort(key=lambda r: r[0])
    res[f["name"]] = (inp0, rows)
    print(
      f"\n  -- {f['name']} input (assembly={inp0:+.3f} rad = {inp0 * R2D:+.1f} deg) --",
      flush=True,
    )
    print(
      f"     {'input rad':>10s} {'input deg':>10s} {'aperture mm':>12s} {'weld mm':>9s} {'feasible':>9s}",
      flush=True,
    )
    for u, ap, w, fin in rows:
      flag = "ok" if (fin and w <= WELD_THRESH_MM) else "LOCK"
      aps = f"{ap:12.1f}" if np.isfinite(ap) else f"{'nan':>12s}"
      print(f"     {u:10.4f} {u * R2D:10.1f} {aps} {w:9.3f} {flag:>9s}", flush=True)
  return res


def report_aperture_calibration(curl_res):
  """From the per-finger sweep, report the aperture(curl) mapping and the open(130mm)
  / closed(0mm) calibration anchors, AND a 3-finger SYNCHRONOUS aperture map."""
  print("\n===== APERTURE CALIBRATION (per-finger feasible band) =====", flush=True)
  feas = {}
  for name, (inp0, rows) in curl_res.items():
    good = [(u, ap, w) for (u, ap, w, fin) in rows if fin and w <= WELD_THRESH_MM]
    us = np.array([g[0] for g in good])
    aps = np.array([g[1] for g in good])
    feas[name] = (us, aps)
    amin, amax = aps.min(), aps.max()
    umin_feas, umax_feas = us.min(), us.max()
    print(
      f"  {name}: feasible curl [{umin_feas:+.3f},{umax_feas:+.3f}] rad "
      f"([{umin_feas * R2D:+.1f},{umax_feas * R2D:+.1f}] deg); "
      f"aperture spans {amin:.1f}..{amax:.1f} mm",
      flush=True,
    )
  return feas


# ---------------------------------------------------------------------------
# 1b. SYNCHRONOUS aperture: drive all 3 inputs together (the real grasp motion) to
# map a single "curl command" -> aperture, and find the 130mm (open) + min (closed).
# ---------------------------------------------------------------------------
def sweep_curl_sync(mjm, mjd, tg, lo=-0.9, hi=1.6, n=51, settle=180):
  print(
    "\n===== 1b. SYNCHRONOUS curl -> aperture (all 3 inputs together) =====", flush=True
  )
  _reset_assembly(mjm, mjd)
  _settle(mjm, mjd, 200)
  inp0 = {f["name"]: mjd.qpos[_jadr(mjm, f["input"])] for f in V.WELD_FINGERS}
  deltas = np.linspace(lo, hi, n)
  rows = []
  print(
    f"     {'delta rad':>10s} {'aperture mm':>12s} {'weld mm':>9s} {'feasible':>9s}",
    flush=True,
  )
  for d in deltas:
    _reset_assembly(mjm, mjd)
    _settle(mjm, mjd, 120)
    for f in V.WELD_FINGERS:
      mjd.ctrl[_aid(mjm, f["input"])] = inp0[f["name"]] + d
    for _ in range(settle):
      mujoco.mj_step(mjm, mjd)
    fin = bool(np.all(np.isfinite(mjd.qpos)))
    al = V.weld_alignment_mm(mjm, mjd)
    wmax = max(al.values()) if al else 0.0
    ap = _circumdiam_mm(_tip_pts(mjm, mjd, tg)) if fin else float("nan")
    flag = "ok" if (fin and wmax <= WELD_THRESH_MM) else "LOCK"
    rows.append((float(d), ap, wmax, fin and wmax <= WELD_THRESH_MM))
    aps = f"{ap:12.1f}" if np.isfinite(ap) else f"{'nan':>12s}"
    print(f"     {d:10.4f} {aps} {wmax:9.3f} {flag:>9s}", flush=True)
  good = [(d, ap) for (d, ap, w, ok) in rows if ok]
  ds = np.array([g[0] for g in good])
  aps = np.array([g[1] for g in good])
  # find delta where aperture crosses 130mm (open) and the minimum (closed)
  print(
    f"\n  synchronous aperture range: {aps.min():.1f}..{aps.max():.1f} mm over "
    f"delta [{ds.min():+.3f},{ds.max():+.3f}] rad",
    flush=True,
  )
  # interpolate delta @ 130 mm (aperture decreases as delta increases = curl closes)
  order = np.argsort(aps)
  if aps.min() <= 130 <= aps.max():
    d130 = float(np.interp(130.0, aps[order], ds[order]))
    print(f"  aperture=130 mm (OPEN) at curl delta = {d130:+.3f} rad", flush=True)
  else:
    print(
      f"  130 mm not bracketed; closest aperture {aps.min():.1f}..{aps.max():.1f}",
      flush=True,
    )
  dmin_ap = ds[np.argmin(aps)]
  print(
    f"  min aperture {aps.min():.1f} mm (most CLOSED) at curl delta = {dmin_ap:+.3f} rad",
    flush=True,
  )
  return rows, inp0


# ---------------------------------------------------------------------------
# 3. BASE swing range with SELECTIVE self-collision (geometric mesh-proximity).
# We can't use mujoco contacts (hand geoms have contype/conaffinity=0 and overlap
# internally). Instead: for each base angle, recompute kinematics and check the
# min distance between the swept finger's tip/finger geoms and the palm/base/OTHER
# fingers via mujoco's geom-pair distance (mj_geomDistance), ignoring same-finger
# internal pairs. Collision when min distance < margin.
# ---------------------------------------------------------------------------
def _finger_geom_sets(mjm):
  """Map each finger -> its set of geom ids, plus palm/base geoms."""
  finger_bodies = {
    "F1": [
      "m_link_step",
      "s_link2_step",
      "s_link1_step",
      "finger1_step",
      "s_link2_step_2",
    ],
    "F2": [
      "dae_mhl_step",
      "finger3_step",
      "finger2_step",
      "finger1_step_2",
      "finger3_step_2",
    ],
    "F3": [
      "dae_mhr_step",
      "finger4_step",
      "finger2_step_2",
      "finger1_step_3",
      "finger4_step_2",
    ],
  }
  palm_bodies = ["base_step"]
  fg = {}
  for n, bods in finger_bodies.items():
    gs = []
    for bn in bods:
      bi = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, bn)
      if bi < 0:
        continue
      gs += [g for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == bi]
    fg[n] = gs
  pg = []
  for bn in palm_bodies:
    bi = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, bn)
    if bi >= 0:
      pg += [g for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == bi]
  # tip geoms (the most distal, most likely to collide)
  tipg = {
    f["name"]: [V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["tip"])]
    for f in V.WELD_FINGERS
  }
  return fg, pg


def _min_dist(mjm, mjd, ga, gb):
  """Min distance (m) over geom-pair (a in gset_a, b in gset_b) via mj_geomDistance."""
  fromto = np.zeros(6)
  dmin = np.inf
  for ga_i in ga:
    for gb_i in gb:
      d = mujoco.mj_geomDistance(mjm, mjd, ga_i, gb_i, 10.0, fromto)
      dmin = min(dmin, d)
  return dmin


def sweep_base(mjm, mjd, lo=-1.2, hi=1.2, n=49, settle=80, margin=0.002):
  """Sweep each base joint; report min self-clearance (swept finger vs palm + the two
  other fingers, excluding same-finger internal pairs). Collision = clearance < margin."""
  print(
    "\n===== 3. BASE swing range (selective self-collision, mesh-proximity) =====",
    flush=True,
  )
  fg, pg = _finger_geom_sets(mjm)
  res = {}
  for f in V.WELD_FINGERS:
    name = f["name"]
    others = [g for on in fg if on != name for g in fg[on]] + pg
    swept = fg[name]
    _reset_assembly(mjm, mjd)
    _settle(mjm, mjd, 120)
    base0 = mjd.qpos[_jadr(mjm, f["base"])]
    ai = _aid(mjm, f["base"])
    rows = []
    for direction in ("up", "down"):
      _reset_assembly(mjm, mjd)
      _settle(mjm, mjd, 100)
      targets = np.linspace(
        base0, base0 + (hi if direction == "up" else lo), n // 2 + 1
      )
      for tgt in targets:
        mjd.ctrl[ai] = tgt
        for _ in range(settle):
          mujoco.mj_step(mjm, mjd)
        if not np.all(np.isfinite(mjd.qpos)):
          rows.append((float(tgt), float("nan"), False, False))
          break
        clr = _min_dist(mjm, mjd, swept, others)
        al = V.weld_alignment_mm(mjm, mjd)
        wmax = max(al.values()) if al else 0.0
        collide = clr < margin
        rows.append(
          (
            float(mjd.qpos[_jadr(mjm, f["base"])]),
            float(clr),
            bool(collide),
            wmax <= WELD_THRESH_MM,
          )
        )
        if collide:
          break
    rows.sort(key=lambda r: r[0])
    res[name] = (base0, rows)
    print(
      f"\n  -- {name} base (assembly={base0:+.3f} rad = {base0 * R2D:+.1f} deg), margin={margin * 1000:.0f} mm --",
      flush=True,
    )
    print(
      f"     {'base rad':>10s} {'base deg':>10s} {'clearance mm':>13s} {'collide':>8s} {'weldOK':>7s}",
      flush=True,
    )
    for b, clr, col, wok in rows:
      clrs = f"{clr * 1000:13.1f}" if np.isfinite(clr) else f"{'nan':>13s}"
      print(
        f"     {b:10.4f} {b * R2D:10.1f} {clrs} {str(col):>8s} {str(wok):>7s}",
        flush=True,
      )
  return res


# ---------------------------------------------------------------------------
# 4. PASSIVE ROM: over the feasible synchronous curl band, record min/max each passive
# joint reaches (their physics-determined ROM). Also include the per-finger curl band.
# ---------------------------------------------------------------------------
def sweep_passive_rom(mjm, mjd, curl_feas_band):
  """curl_feas_band: dict finger -> (lo_delta, hi_delta) feasible curl deltas. Drive
  each finger's input across its feasible band, record passive joint min/max (rad)."""
  print("\n===== 4. PASSIVE joint ROM over feasible curl band =====", flush=True)
  rom = {}
  # also track driven joint extremes
  for f in V.WELD_FINGERS:
    name = f["name"]
    lo, hi = curl_feas_band[name]
    passive = f["passive"]
    mins = {p: np.inf for p in passive}
    maxs = {p: -np.inf for p in passive}
    _reset_assembly(mjm, mjd)
    _settle(mjm, mjd, 150)
    inp0 = mjd.qpos[_jadr(mjm, f["input"])]
    ai = _aid(mjm, f["input"])
    for tgt in np.linspace(inp0 + lo, inp0 + hi, 31):
      mjd.ctrl[ai] = tgt
      for _ in range(150):
        mujoco.mj_step(mjm, mjd)
      if not np.all(np.isfinite(mjd.qpos)):
        break
      for p in passive:
        v = mjd.qpos[_jadr(mjm, p)]
        mins[p] = min(mins[p], v)
        maxs[p] = max(maxs[p], v)
    rom[name] = (mins, maxs)
    print(
      f"\n  -- {name} passive ROM (curl delta [{lo:+.2f},{hi:+.2f}] rad) --", flush=True
    )
    for p in passive:
      print(
        f"     {p:16s}: [{mins[p]:+.4f}, {maxs[p]:+.4f}] rad  "
        f"([{mins[p] * R2D:+.1f}, {maxs[p] * R2D:+.1f}] deg)",
        flush=True,
      )
  return rom


def main():
  mjm, mjd = V.build_model_welded()
  tg = _tip_geoms(mjm)
  print(f"WELD model: nq={mjm.nq} nu={mjm.nu} neq={mjm.neq}", flush=True)

  curl_res = sweep_curl(mjm, mjd, tg, lo=-1.4, hi=1.4, n=29, settle=150)
  feas = report_aperture_calibration(curl_res)
  sync_rows, inp0 = sweep_curl_sync(mjm, mjd, tg, lo=-0.9, hi=1.6, n=26, settle=180)

  # feasible curl band per finger (from per-finger sweep, relative to assembly)
  band = {}
  for name, (inp0f, rows) in curl_res.items():
    good = [u for (u, ap, w, fin) in rows if fin and w <= WELD_THRESH_MM]
    band[name] = (min(good) - inp0f, max(good) - inp0f)
  sweep_base(mjm, mjd, lo=-1.2, hi=1.2, n=33, settle=80, margin=0.002)
  sweep_passive_rom(mjm, mjd, band)

  print("\n===== DERIVATION COMPLETE =====", flush=True)


if __name__ == "__main__":
  main()
