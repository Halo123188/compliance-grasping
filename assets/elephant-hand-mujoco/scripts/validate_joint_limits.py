"""VALIDATION for the DERIVED per-joint ROM limits (mygripper_H100_R "elephant").

Asserts, headless (no viser / no port bind):
 (1) limits are actually applied: jnt_range/jnt_limited set for all 15 listed joints;
     actuator_ctrlrange set for the 6 driven joints; matches JOINT_LIMITS.
 (2) drive all 6 driven joints to BOTH their new limits, mj_step settle ->
       (a) qpos finite, (b) weld alignment stays < 0.5 mm settled.
 (3) APERTURE spans ~0..130 mm at the base-swing extremes (the calibration anchors).
 (4) the assembly qpos0 starts inside all limits (no clamp snap at reset).
 (5) coupled model still compiles with the limits (additive, non-breaking).

Run: PYTHONPATH=src /home/alok/.conda/envs/g313/bin/python scripts/validate_joint_limits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from my_mjlab_project import mygripper_arm_viewer as V  # noqa: E402

R2D = 180.0 / np.pi


def _jadr(mjm, jn):
    ji = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
    return int(mjm.jnt_qposadr[ji])


def _aid(mjm, jn):
    return V._attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)


def _tip_geoms(mjm):
    out = {}
    for f in V.WELD_FINGERS:
        bi = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["tip"])
        out[f["name"]] = [g for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == bi][0]
    return out


CLOSED_BASELINE_MM = 87.3  # circumdiam through tip centers at the closed pose = 0 object


def _aperture_mm(mjm, mjd, tg):
    P = np.array([mjd.geom_xpos[tg[n]] for n in ("F1", "F2", "F3")])
    a = np.linalg.norm(P[1] - P[2]); b = np.linalg.norm(P[0] - P[2]); c = np.linalg.norm(P[0] - P[1])
    s = (a + b + c) / 2
    area = max(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0)), 1e-12)
    return 2 * (a * b * c / (4 * area)) * 1000.0 - CLOSED_BASELINE_MM


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


def _settle(mjm, mjd, n=200):
    for _ in range(n):
        mujoco.mj_step(mjm, mjd)


def main():
    ok = True
    mjm, mjd = V.build_model_welded()
    tg = _tip_geoms(mjm)

    # (1) limits applied
    print("=== (1) limits applied (jnt_range / jnt_limited / ctrlrange) ===", flush=True)
    driven = set(V.WELD_DRIVEN)
    for jn, (lo, hi) in V.JOINT_LIMITS.items():
        ji = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        r = mjm.jnt_range[ji]; lim = mjm.jnt_limited[ji]
        match = bool(lim) and abs(r[0] - lo) < 1e-6 and abs(r[1] - hi) < 1e-6
        ok &= match
        extra = ""
        if jn in driven:
            ai = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            cr = mjm.actuator_ctrlrange[ai]
            cmatch = abs(cr[0] - lo) < 1e-6 and abs(cr[1] - hi) < 1e-6 and bool(mjm.actuator_ctrllimited[ai])
            ok &= cmatch
            extra = f" ctrlrange=[{cr[0]:+.3f},{cr[1]:+.3f}] {'OK' if cmatch else 'MISMATCH'}"
        print(f"  {jn:16s} range=[{r[0]:+.3f},{r[1]:+.3f}] lim={lim} {'OK' if match else 'MISMATCH'}{extra}", flush=True)

    # (4) assembly qpos0 inside limits
    print("\n=== (4) assembly qpos0 inside all limits (no reset snap) ===", flush=True)
    _reset_assembly(mjm, mjd)
    inside = True
    for jn, (lo, hi) in V.JOINT_LIMITS.items():
        q = mjd.qpos[_jadr(mjm, jn)]
        within = lo - 1e-6 <= q <= hi + 1e-6
        inside &= within
        if not within:
            print(f"  OUT: {jn} q={q:+.4f} not in [{lo:+.3f},{hi:+.3f}]", flush=True)
    print(f"  all assembly qpos0 inside limits: {inside}", flush=True)
    ok &= inside

    # (2) drive 6 driven joints to BOTH limits; weld stays small, finite
    print("\n=== (2) drive 6 driven joints to BOTH limits; finite + weld small ===", flush=True)
    for which, idx in (("MIN", 0), ("MAX", 1)):
        _reset_assembly(mjm, mjd)
        _settle(mjm, mjd, 120)
        for f in V.WELD_FINGERS:
            for jn in (f["base"], f["input"]):
                mjd.ctrl[_aid(mjm, jn)] = V.JOINT_LIMITS[jn][idx]
        _settle(mjm, mjd, 600)
        fin = bool(np.all(np.isfinite(mjd.qpos)))
        al = V.weld_alignment_mm(mjm, mjd)
        wmax = max(al.values())
        passed = fin and wmax < 0.5
        ok &= passed
        print(f"  all driven -> {which} limit: finite={fin} weld_max={wmax:.3f} mm "
              f"({'PASS' if passed else 'FAIL'})", flush=True)

    # (3) aperture spans ~0..130 mm at base-swing extremes
    print("\n=== (3) aperture at base-swing limits (calibration anchors) ===", flush=True)
    _reset_assembly(mjm, mjd); _settle(mjm, mjd, 200)
    base0 = {f["name"]: mjd.qpos[_jadr(mjm, f["base"])] for f in V.WELD_FINGERS}
    aps = {}
    for which, target in (("OPEN(min,-0.70)", "lo"), ("CLOSED(max,+0.90)", "hi")):
        _reset_assembly(mjm, mjd); _settle(mjm, mjd, 80)
        for f in V.WELD_FINGERS:
            r = V.JOINT_LIMITS[f["base"]]
            mjd.ctrl[_aid(mjm, f["base"])] = r[0] if target == "lo" else r[1]
        _settle(mjm, mjd, 400)
        ap = _aperture_mm(mjm, mjd, tg)
        aps[which] = ap
        print(f"  base -> {which}: aperture = {ap:6.1f} mm", flush=True)
    open_ap = aps["OPEN(min,-0.70)"]; closed_ap = aps["CLOSED(max,+0.90)"]
    ap_ok = abs(open_ap - 130) < 12 and abs(closed_ap - 0) < 6
    ok &= ap_ok
    print(f"  aperture spans ~0..130 mm: open={open_ap:.1f} (~130), closed={closed_ap:.1f} (~0) "
          f"-> {'PASS' if ap_ok else 'FAIL'}", flush=True)

    # (5) coupled model compiles with limits
    print("\n=== (5) coupled model compiles with limits (non-breaking) ===", flush=True)
    try:
        mjm_c, mjd_c = V.build_model_coupled()
        n_lim = sum(int(mjm_c.jnt_limited[V._attached_id(mjm_c, mujoco.mjtObj.mjOBJ_JOINT, jn)])
                    for jn in V.JOINT_LIMITS)
        for _ in range(100):
            mujoco.mj_step(mjm_c, mjd_c)
        cfin = bool(np.all(np.isfinite(mjd_c.qpos)))
        print(f"  coupled: {n_lim}/15 joints limited, 100 steps finite={cfin}", flush=True)
        ok &= (n_lim == 15 and cfin)
    except Exception as e:
        print(f"  coupled FAILED: {e}", flush=True)
        ok = False

    # base build untouched: hand joints keep the URDF placeholder +/-pi ranges, NOT our
    # derived limits (build_model is unchanged; limits only go on welded/coupled).
    print("\n=== base build_model() unchanged (keeps URDF +/-pi placeholders) ===", flush=True)
    mjm_b, _ = V.build_model()
    still_placeholder = True
    for jn in V.JOINT_LIMITS:
        ji = V._attached_id(mjm_b, mujoco.mjtObj.mjOBJ_JOINT, jn)
        r = mjm_b.jnt_range[ji]
        if not (abs(abs(r[0]) - np.pi) < 1e-3 and abs(abs(r[1]) - np.pi) < 1e-3):
            still_placeholder = False
            print(f"  CHANGED: {jn} range=[{r[0]:+.3f},{r[1]:+.3f}] (expected +/-pi)", flush=True)
    print(f"  base build hand joints still at URDF +/-pi (derived limits NOT applied): "
          f"{still_placeholder}", flush=True)
    ok &= still_placeholder

    print(f"\n=== OVERALL: {'PASS' if ok else 'FAIL'} ===", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
