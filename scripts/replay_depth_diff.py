"""Replay a recorded run's depth frames against the sim's render of the same pose.

`deploy/run.py --record` stores the depth frame the policy actually saw at every
step. This puts the recorded JOINT ANGLES back into the sim, renders scene_cam
at each one, and diffs the two images. Where they disagree is where the policy
was fed something its training never contained.

  uv run python scripts/replay_depth_diff.py run3.npz --cube-xy 0.5207 0.1016

The comparison is only meaningful because the arm is in the frame: at the home
pose the two agree to a couple of millimetres (deploy/live_view.py measured
real - sim p50 +2 mm, |d| p90 12 mm), so anything that appears once the arm
starts moving is about the arm, not the bench.

Recorded depth is float16 of the normalised observation (~0.06 mm at bench
range). Recordings made before 2026-08-07 store uint8 instead, quantised to
11.8 mm -- enough to see holes and gross errors, not enough to see the 1-3 mm
bias this policy is sensitive to. Both are read. 0 means no return, in the
recording and -- after the cutoff is applied -- in the render.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mjlab.asset_zoo.robots.twofinger_wide.arm_cfg import (  # noqa: E402
  ARM_BASE_Z,
  twofinger_arm_spec,
)
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (  # noqa: E402
  RESTING_Z,
  get_cube_spec,
  get_table_spec,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import calib  # noqa: E402
from deploy.check_camera import depth_to_grey, encode_png  # noqa: E402
from deploy.live_view import diff_rgb  # noqa: E402
from sim_depth import RENDER_WH, render_observation  # noqa: E402


def build(cube_xy):
  arm = twofinger_arm_spec()
  for b in arm.worldbody.bodies:
    if b.name == "base":
      b.pos = [0.0, 0.0, ARM_BASE_Z]
  arm.attach(get_table_spec(), prefix="", frame=arm.worldbody.add_frame())
  arm.attach(get_cube_spec(), prefix="", frame=arm.worldbody.add_frame())
  m = arm.compile()
  d = mujoco.MjData(m)
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
  adr = m.jnt_qposadr[m.body_jntadr[bid]]
  d.qpos[adr : adr + 7] = [*cube_xy, RESTING_Z, 1.0, 0.0, 0.0, 0.0]
  jadr = [
    m.jnt_qposadr[
      mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, n if n.startswith("joint") else f"{n}_tf"
      )
    ]
    for n in calib.JOINT_NAMES
  ]
  return m, d, jadr


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("recording", help="npz from deploy/run.py --record")
  ap.add_argument("--cube-xy", type=float, nargs=2, default=(0.5207, 0.1016))
  ap.add_argument("--out", default=None, help="directory for per-step PNGs")
  ap.add_argument("--every", type=int, default=1, help="stride over steps")
  ap.add_argument(
    "--png-steps",
    type=int,
    nargs="*",
    default=None,
    help="steps to write images for (default: the last 6 examined)",
  )
  args = ap.parse_args()

  z = np.load(args.recording)
  q = z["joint_pos"]
  # float16 normalised observation since 2026-08-07; uint8 before that.
  if "depth" in z:
    depth_obs = z["depth"].astype(np.float64)
  else:
    depth_obs = z["depth_u8"].astype(np.float64) / 255.0
  m, d, jadr = build((float(args.cube_xy[0]), float(args.cube_xy[1])))
  w, h = RENDER_WH
  renderer = mujoco.Renderer(m, height=h, width=w)
  renderer.enable_depth_rendering()

  steps = list(range(0, len(q), args.every))
  rows = []
  frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
  for i in steps:
    d.qpos[jadr] = q[i]
    mujoco.mj_forward(m, d)
    sim = render_observation(renderer, d)
    real = depth_obs[i] * calib.DEPTH_CUTOFF_M
    real[depth_obs[i] <= 0] = 0.0

    rv, sv = real > 0, sim > 0
    both = rv & sv
    # The asymmetry is the point: a pixel the renderer resolves and the sensor
    # does not is a hole the policy never met in training. The reverse (real
    # sees something the model has no geometry for) matters too but is rarer.
    hole = sv & ~rv
    extra = rv & ~sv
    diff = np.abs(real[both] - sim[both]) if both.any() else np.array([0.0])
    rows.append(
      (
        i,
        100 * rv.mean(),
        100 * sv.mean(),
        100 * hole.mean(),
        100 * extra.mean(),
        1000 * float(np.median(diff)),
        1000 * float(np.percentile(diff, 90)),
      )
    )
    frames[i] = (real, sim)
  renderer.close()

  print(f"{args.recording}: {len(q)} steps, examining {len(steps)}")
  print(
    f"{'step':>5}{'real ok':>9}{'sim ok':>8}{'HOLE':>7}{'extra':>7}"
    f"{'|d| p50':>9}{'p90':>7}   (% of frame, mm)"
  )
  for r in rows:
    print(
      f"{r[0]:5d}{r[1]:8.1f}%{r[2]:7.1f}%{r[3]:6.1f}%{r[4]:6.1f}%{r[5]:9.1f}{r[6]:7.1f}"
    )
  a = np.array([r[3] for r in rows])
  print(f"\nholes (sim sees, real does not): mean {a.mean():.1f}%  max {a.max():.1f}%")
  print(
    f"  first step over 10%: {steps[int(np.argmax(a > 10))] if (a > 10).any() else 'never'}"
  )

  if args.out:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = args.png_steps if args.png_steps else steps[-6:]
    for i in want:
      if i not in frames:
        continue
      real, sim = frames[i]
      scale = (0.15, 1.35)
      encoded = {
        "real": encode_png(depth_to_grey(real, *scale)),
        "sim": encode_png(depth_to_grey(sim, *scale)),
        "diff": encode_png(diff_rgb(real, sim)),
      }
      for kind, blob in encoded.items():
        (out / f"step{i:04d}_{kind}.png").write_bytes(blob)
    print(f"\nwrote PNGs for steps {list(want)} to {out}/")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
