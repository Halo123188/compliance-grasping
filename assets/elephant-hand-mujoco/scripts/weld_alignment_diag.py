"""Weld alignment-under-movement diagnostic (the user's explicit ask).

For the COMBINED arm+hand model, for each finger, sweep the actuated INPUT loop joint
across its feasible range, settle the sim, and measure the WELD ALIGNMENT ERROR =
world distance between the two duplicate coupler bodies' shared weld anchor point.
Produces a table: input angle -> alignment error (mm) -> stable?, and the max/mean
across the swept range per finger.

Then runs the SAME measurement on the MIMIC (build_model_coupled) model, where the
loop is closed only in joint space and the two coupler bodies are NOT physically
joined -- so they drift. The contrast is the whole point: a correct weld holds
alignment ~0 (sub-0.2 mm) where the mimic drifts ~1.5-2 mm.

Headless only (no viser, no port bind).

Run: /home/alok/.conda/envs/g313/bin/python scripts/weld_alignment_diag.py
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from my_mjlab_project import mygripper_arm_viewer as V  # noqa: E402

# Feasible input range relative to the assembly value (the loop closes over roughly
# this band; sweeping wider drives the four-bar toward a singular toggle pose). The
# compiled joints are unlimited (jnt_limited=False) so we clamp to the loop's range.
INPUT_LO, INPUT_HI = -0.6, 1.2
N_STEPS = 21
SETTLE = 150  # mj_step iterations per swept target (let alignment converge)


def _aid(mjm, jn):
    return V._attached_id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)


def _jadr(mjm, jn):
    ji = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
    return int(mjm.jnt_qposadr[ji])


def _coupler_separation_mm(mjm, mjd, f):
    """World distance between the two coupler bodies' shared weld anchor point,
    computed from body1's pose vs body2's pose. Works for weld OR mimic model
    (same bodies exist in both)."""
    b1 = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["b1"])
    b2 = V._attached_id(mjm, mujoco.mjtObj.mjOBJ_BODY, f["b2"])
    R1 = mjd.xmat[b1].reshape(3, 3)
    p_from_b1 = mjd.xpos[b1] + R1 @ np.array(f["relpos"])
    p_from_b2 = mjd.xpos[b2]
    return float(np.linalg.norm(p_from_b1 - p_from_b2) * 1000.0)


def _hold_driven(mjm, mjd):
    for f in V.WELD_FINGERS:
        for jn in (f["base"], f["input"]):
            ai = _aid(mjm, jn)
            if ai >= 0:
                mjd.ctrl[ai] = mjd.qpos[_jadr(mjm, jn)]


def sweep_model(mjm, mjd, label, per_target_rows=False):
    """Sweep each finger's input across [assembly+LO, assembly+HI], settle, record
    coupler separation. Returns {finger: (inputs, seps, finite)}."""
    print(f"\n===== {label} =====", flush=True)
    results = {}
    for f in V.WELD_FINGERS:
        # fresh reset
        mujoco.mj_resetData(mjm, mjd)
        mjd.qpos[:7] = [0, 0, 0, 1.57, 0, 0, 0]
        for jn, val in V.WELD_ASSEMBLY.items():
            mjd.qpos[_jadr(mjm, jn)] = val
        mjd.ctrl[:7] = [0, 0, 0, 1.57, 0, 0, 0]
        mujoco.mj_forward(mjm, mjd)
        _hold_driven(mjm, mjd)
        for _ in range(150):
            mujoco.mj_step(mjm, mjd)

        ai = _aid(mjm, f["input"])
        inp0 = mjd.qpos[_jadr(mjm, f["input"])]
        targets = np.linspace(inp0 + INPUT_LO, inp0 + INPUT_HI, N_STEPS)
        inputs, seps = [], []
        finite = True
        for tgt in targets:
            mjd.ctrl[ai] = tgt
            for _ in range(SETTLE):
                mujoco.mj_step(mjm, mjd)
            if not np.all(np.isfinite(mjd.qpos)):
                finite = False
                break
            inputs.append(float(mjd.qpos[_jadr(mjm, f["input"])]))
            seps.append(_coupler_separation_mm(mjm, mjd, f))
        results[f["name"]] = (np.array(inputs), np.array(seps), finite)

        if per_target_rows and seps:
            print(f"  -- {f['name']} (input {inputs[0]:+.3f}..{inputs[-1]:+.3f} rad) --", flush=True)
            for u, s in zip(inputs, seps):
                stable = "ok" if s < 0.5 else ("hi" if s < 2 else "DRIFT")
                print(f"      input={u:+.4f} rad  align={s:7.4f} mm  [{stable}]", flush=True)
    return results


def main():
    print("Loop-closure alignment-under-movement diagnostic (combined arm+hand).", flush=True)
    print(f"sweep: input in assembly+[{INPUT_LO},{INPUT_HI}] rad, {N_STEPS} pts, "
          f"{SETTLE} settle steps/pt, gravity ON, arm at pre-grasp.", flush=True)

    # ---- WELD model ----
    mjm_w, mjd_w = V.build_model_welded()
    print(f"\nWELD model: nq={mjm_w.nq} nu={mjm_w.nu} neq={mjm_w.neq} "
          f"(armature={V.WELD_ARMATURE} damping={V.WELD_DAMPING} solref={V.WELD_SOLREF} "
          f"Newton/{V.WELD_ITERATIONS} dt={V.WELD_TIMESTEP})", flush=True)
    weld_res = sweep_model(mjm_w, mjd_w, "WELD — alignment per swept input", per_target_rows=True)

    # ---- MIMIC model (build_model_coupled) ----
    mjm_m, mjd_m = V.build_model_coupled()
    print(f"\nMIMIC model: nq={mjm_m.nq} nu={mjm_m.nu} neq={mjm_m.neq} (mjEQ_JOINT cubic)", flush=True)
    mimic_res = sweep_model(mjm_m, mjd_m, "MIMIC — coupler separation per swept input")

    # ---- Summary table ----
    print("\n\n================ SUMMARY: coupler alignment under movement ================", flush=True)
    print(f"{'Finger':6s} | {'WELD max':>9s} {'WELD mean':>10s} {'stable':>7s} | "
          f"{'MIMIC max':>10s} {'MIMIC mean':>11s} {'stable':>7s} | {'improvement':>12s}", flush=True)
    print("-" * 86, flush=True)
    for f in V.WELD_FINGERS:
        n = f["name"]
        wi, ws, wfin = weld_res[n]
        mi, ms, mfin = mimic_res[n]
        wmax, wmean = (ws.max(), ws.mean()) if len(ws) else (float("nan"),) * 2
        mmax, mmean = (ms.max(), ms.mean()) if len(ms) else (float("nan"),) * 2
        impr = mmax / wmax if wmax > 0 else float("inf")
        print(f"{n:6s} | {wmax:8.4f}m {wmean:9.4f}m {str(wfin):>7s} | "
              f"{mmax:9.4f}m {mmean:10.4f}m {str(mfin):>7s} | {impr:10.1f}x tighter", flush=True)
    print("-" * 86, flush=True)
    print("(units mm; 'align' = world distance between the two duplicate coupler bodies'", flush=True)
    print(" shared weld anchor. WELD physically fuses them -> ~0; MIMIC only joint-couples", flush=True)
    print(" -> the two bodies drift apart = the 'two mounts appear' symptom.)", flush=True)


if __name__ == "__main__":
    main()
