"""Live D435 next to the sim's render of the same scene, on localhost.

The policy sees one depth frame and nothing else, 120x160. Everything about
whether the real bench looks like the trained bench -- background distance,
where the claw and the mast sit in frame, how much of the foam returns nothing
-- is in that image, and no summary statistic substitutes for putting the two
side by side.

  # once, on the workstation (needs mjlab + a GPU):
  uv run python scripts/render_sim_depth.py sim_ref.npz --cube-xy 0.5207 0.1016
  # then, in the deploy venv:
  .venv-deploy/bin/python -m deploy.live_view --sim sim_ref.npz
  # -> http://localhost:8000

The sim half is a still, because the scene does not move; the real half is
live. Put the arm at the policy's home pose and the cube where you rendered it,
or the comparison is of two different scenes and every difference is yours.

BINDS TO LOCALHOST ONLY. It serves the robot's camera feed with no
authentication of any kind; --host is there for an ssh tunnel endpoint, not for
putting this on a network.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from . import calib
from .check_camera import depth_to_grey, encode_png

# Pinned so the two panels share a scale. The bench runs 0.17 m (the mast, right
# under the lens) to about 1.3 m (the far wall), and letting each frame pick its
# own range makes a difference in the FAR background rescale the CUBE.
SCALE_M = (0.15, 1.35)


PAGE = """<!doctype html>
<meta charset="utf-8"><title>D435 vs sim</title>
<style>
 body{background:#111;color:#ddd;font:13px/1.5 ui-monospace,monospace;margin:0;
      padding:16px}
 h1{font-size:14px;font-weight:600;margin:0 0 12px}
 .row{display:flex;gap:14px;flex-wrap:wrap}
 figure{margin:0}
 figcaption{margin:6px 0 0;color:#9aa}
 img{image-rendering:pixelated;width:480px;height:360px;background:#000;
     border:1px solid #333;display:block}
 #stats{margin-top:14px;white-space:pre;color:#9aa}
 b{color:#fff}
</style>
<h1>live D435 vs sim render &mdash; near is bright, black is no return</h1>
<div class="row">
  <figure><img id="real"><figcaption>REAL (live)</figcaption></figure>
  <figure><img id="sim"><figcaption>SIM (reference &mdash; see the server log
    for the bench it assumes)</figcaption></figure>
  <figure><img id="diff"><figcaption>DIFF &mdash; blue: real NEARER (or sees
    what sim does not) &middot; red: real FURTHER (or blank where sim is
    solid)</figcaption></figure>
  <figure><img id="over"><figcaption>OVERLAY &mdash; sim in RED over real in
    CYAN. Aligned reads grey; a shift shows as a red ghost beside a cyan
    one</figcaption></figure>
</div>
<div id="stats">connecting...</div>
<script>
const el = id => document.getElementById(id);
let n = 0;
async function tick(){
  const t = Date.now();
  el('real').src = '/real.png?t=' + t;
  el('diff').src = '/diff.png?t=' + t;
  el('over').src = '/overlay.png?t=' + t;
  // The reference is a still, but it changes when the server is restarted
  // against a different bench -- and a page left open would otherwise show the
  // old one for ever while real and diff kept updating. Re-fetch it every ~4 s;
  // it is 1.5 kB.
  if (n % 20 === 0) el('sim').src = '/sim.png?t=' + t;
  n++;
  try {
    const s = await (await fetch('/stats.json?t=' + t)).json();
    el('stats').textContent = s.text;
  } catch (e) { el('stats').textContent = 'lost the server: ' + e; }
  setTimeout(tick, 200);
}
tick();
</script>
"""


def diff_rgb(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
  """Signed depth difference as RGB. Two colours only: blue nearer, red further.

  Where both images have a reading the colour is the signed difference,
  saturating at 50 mm. Where only ONE has a reading there is no difference to
  take, so it folds into the same two colours by its sign in the limit: a pixel
  the sensor blanks and the renderer fills reads as "real is further than
  anything sim has" -> saturated RED, and the reverse -> saturated BLUE.

  Validity disagreement used to get its own (green) channel, which is more
  honest about the KIND of disagreement but reads as a bright outline hugging
  the gripper silhouette -- about 6% of the frame, right where the red/blue
  alignment signal matters most. Two colours make the left/right offset between
  the two grippers legible at a glance; `deploy.live_view`'s HOLES stat is what
  quantifies the blanking, and it does that better than a colour ever did.
  """
  both = (real > 0) & (sim > 0)
  out = np.zeros((*real.shape, 3), np.uint8)
  d = np.zeros(real.shape)
  d[both] = real[both] - sim[both]
  mag = np.clip(np.abs(d) / 0.05, 0, 1)  # saturate at 50 mm
  scaled = (mag * 255).astype(np.uint8)
  out[..., 2][both & (d < 0)] = scaled[both & (d < 0)]
  out[..., 0][both & (d > 0)] = scaled[both & (d > 0)]
  out[..., 0][(sim > 0) & (real <= 0)] = 255  # sensor blank, renderer solid
  out[..., 2][(real > 0) & (sim <= 0)] = 255  # renderer blank, sensor solid
  return out


def overlay_rgb(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
  """The two frames superimposed: sim in RED, real in CYAN.

  A signed difference image answers "how far apart is this pixel" but not "is
  the same OBJECT in the same PLACE", because a shifted claw produces a red
  band and a blue band with nothing tying them together. Putting sim on the red
  channel and real on green+blue ties them together the way an anaglyph does:
  where the two agree the channels sum to neutral grey, and where the claw has
  moved you see a red claw next to a cyan one and can read the offset straight
  off the image.

  Greyscale, not silhouettes, on purpose. Nothing here segments the gripper --
  a depth threshold cannot, since the near edge of the table is as close to the
  lens as the claw is -- so the honest thing is to show both images whole.
  """
  r = depth_to_grey(sim, *SCALE_M)
  c = depth_to_grey(real, *SCALE_M)
  out = np.zeros((*real.shape, 3), np.uint8)
  out[..., 0] = r
  out[..., 1] = c
  out[..., 2] = c
  return out


# How far the shift search looks, in pixels each way. At the bench the frame is
# about 2.7 mm/px, so +-10 px is +-27 mm -- past that a "best" match is more
# likely to be a different part of the scene lining up than the same one.
SHIFT_RANGE = 10

# One pixel of the 160x120 frame, in millimetres, at the range the gripper sits
# at. The 848x480 profile's focal length is 428.3 px; the 640x480 centre crop
# downsampled by 4 makes that 107.1 px, so a pixel subtends z/107.1 and at the
# claw's ~0.29 m that is 2.7 mm. Only valid near that range -- the table at
# 1.2 m is 11 mm/px -- which is why the fit is restricted to the gripper band.
MM_PER_PX = 2.7


def best_shift(
  real: np.ndarray, sim: np.ndarray, rows: slice
) -> tuple[int, int, float]:
  """-> (du, dv, mean |d| in metres) of the whole-image shift that best fits.

  du > 0 means the REAL image has to move LEFT to land on the sim, i.e. the
  real scene sits to the RIGHT of where the sim puts it. `rows` restricts the
  fit to a band -- the gripper occupies the top of the frame and the table
  below it is nearly shift-invariant, so fitting the whole frame dilutes
  exactly the signal being measured.
  """
  best = (0, 0, float("inf"))
  for dv in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
    for du in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
      shifted = np.roll(np.roll(real, -dv, axis=0), -du, axis=1)[rows]
      s = sim[rows]
      both = (shifted > 0) & (s > 0)
      if both.sum() < 200:
        continue
      err = float(np.abs(shifted[both] - s[both]).mean())
      if err < best[2]:
        best = (du, dv, err)
  return best


class _Handler(BaseHTTPRequestHandler):
  # Class attributes rather than constructor arguments because
  # ThreadingHTTPServer instantiates the handler per request. `Any` on the
  # camera keeps `deploy.perception` out of this module's import path, which
  # matters: importing it pulls in the pyrealsense2 probe.
  camera: Any = None
  sim: np.ndarray = np.zeros(calib.DEPTH_HW)
  # The frame both halves are read at, taken from the reference render so the
  # two panels are never two different pictures. 120x160 for every checkpoint
  # to date; `render_sim_depth.py --depth-hw` is what would change it.
  depth_hw: tuple[int, int] = calib.DEPTH_HW
  lock = threading.Lock()

  def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
    pass  # one line per image at 5 Hz would bury anything worth reading

  def _send(self, body: bytes, ctype: str) -> None:
    self.send_response(200)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def _real(self) -> np.ndarray:
    with self.lock:
      return self.camera.read().reshape(self.depth_hw) * calib.DEPTH_CUTOFF_M

  def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's name
    path = self.path.split("?")[0]
    try:
      if path == "/":
        self._send(PAGE.encode(), "text/html; charset=utf-8")
      elif path == "/sim.png":
        self._send(encode_png(depth_to_grey(self.sim, *SCALE_M)), "image/png")
      elif path == "/real.png":
        self._send(encode_png(depth_to_grey(self._real(), *SCALE_M)), "image/png")
      elif path == "/diff.png":
        self._send(encode_png(diff_rgb(self._real(), self.sim)), "image/png")
      elif path == "/overlay.png":
        self._send(encode_png(overlay_rgb(self._real(), self.sim)), "image/png")
      elif path == "/stats.json":
        self._send(json.dumps({"text": self._stats()}).encode(), "application/json")
      else:
        self.send_error(404)
    except Exception as exc:  # noqa: BLE001 -- a browser must not kill the server
      self.send_error(500, str(exc))

  def _stats(self) -> str:
    real, sim = self._real(), self.sim
    both = (real > 0) & (sim > 0)
    d = real[both] - sim[both]
    lines = [
      f"{'':10}{'valid':>8}{'p05':>9}{'p50':>9}{'p95':>9}   metres",
      _row("real", real),
      _row("sim", sim),
    ]
    if d.size:
      lines.append(
        f"\noverlap {100 * both.mean():.1f}% of pixels    real - sim: "
        f"p50 {np.median(d) * 1000:+.0f} mm   "
        f"|d| p90 {np.percentile(np.abs(d), 90) * 1000:.0f} mm   "
        f"mean {d.mean() * 1000:+.0f} mm"
      )
    # HOLES is the number to watch, and it is why this page exists in its
    # current form. Measured on run5: pixels where the sensor returns nothing
    # and the renderer returns solid geometry are 2.4% of the frame, and
    # patching just those pixels with the sim's values took the policy's
    # command from 0.205 rad away from its sim-frame behaviour to 0.085.
    # Fixing the geometry everywhere ELSE in the same band did nothing (0.202).
    # So a change to the bench or the sensor preset is an improvement if and
    # only if this number goes down.
    holes = (sim > 0) & (real <= 0)
    top = holes[: holes.shape[0] // 3]
    lines.append(
      f"\nHOLES (sim solid, sensor blank) {100 * holes.mean():5.2f}% of frame"
      f"   {100 * top.mean():5.2f}% in the top third   <-- MINIMISE THIS"
    )
    # The overlay panel shows the offset; this is the same thing as a number,
    # fitted over the top half where the gripper is. mm/px is the depth frame's
    # angular scale at bench range, so it converts a pixel shift into how far
    # the claw actually is from where the sim puts it.
    band = slice(0, real.shape[0] // 2)
    du, dv, err = best_shift(real, sim, band)
    lines.append(
      f"BEST SHIFT of real onto sim (top half)  du {du:+d} px  dv {dv:+d} px"
      f"   -> {abs(du) * MM_PER_PX:.0f} mm sideways"
      f"   residual {err * 1000:.1f} mm"
    )
    return "\n".join(lines)


def _row(label: str, depth: np.ndarray) -> str:
  v = depth[depth > 0]
  if not v.size:
    return f"{label:10}{'0%':>8}"
  return (
    f"{label:10}{100 * (depth > 0).mean():7.1f}%"
    f"{np.percentile(v, 5):9.3f}{np.percentile(v, 50):9.3f}"
    f"{np.percentile(v, 95):9.3f}"
  )


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--sim", required=True, help="npz from scripts/render_sim_depth.py")
  ap.add_argument("--port", type=int, default=8000)
  ap.add_argument(
    "--host",
    default="127.0.0.1",
    help=(
      "localhost by default and it should stay there: this serves the robot's "
      "camera with no authentication. Tunnel with ssh -L rather than binding "
      "to 0.0.0.0."
    ),
  )
  ap.add_argument("--camera-fps", type=int, default=90)
  ap.add_argument(
    "--camera-preset",
    default=None,
    help=(
      "D400 visual preset: high_density blanks the fewest pixels, "
      "high_accuracy the most. Watch the HOLES line while you change it."
    ),
  )
  args = ap.parse_args()

  ref = np.load(args.sim)
  sim = ref["depth_m"]
  # The REFERENCE decides the resolution and the camera is opened to match, so
  # that a reference rendered at some other size cannot end up beside a 120x160
  # live frame. Any frame the 640x480 crop divides exactly is one this can
  # serve.
  cw, ch = calib.D435_CROP_WH
  if sim.ndim != 2 or ch % sim.shape[0] or cw % sim.shape[1]:
    print(
      f"!! {args.sim} is {sim.shape}; expected a 2-D frame the {(cw, ch)} crop "
      "divides exactly, e.g. (120, 160)."
    )
    return 1
  depth_hw = (int(sim.shape[0]), int(sim.shape[1]))
  print(f"[ref] frame {depth_hw[0]}x{depth_hw[1]}")
  # What the reference assumes about the bench. Printed rather than checked,
  # because nothing here can see the bench -- but an operator reading "foam
  # 0.000 m" next to a bench that still has foam on it will catch in a second
  # what would otherwise show up as a 50 mm sim-to-real gap.
  print(f"[ref] {args.sim}")
  if "foam_h" in ref:
    print(f"[ref]   foam height {float(ref['foam_h']):.3f} m")
  if "cube_xy" in ref:
    c = ref["cube_xy"]
    print(
      f"[ref]   cube {'at ' + str(np.round(c, 4).tolist()) if c.size else 'NOT in the scene'}"
    )
  if "arm_joints" in ref and ref["arm_joints"].size:
    print(f"[ref]   arm joints {np.round(ref['arm_joints'][:7], 4).tolist()}")
  else:
    print("[ref]   arm at the policy home pose")

  from .perception import RealSenseDepth

  # BIND FIRST, then open the camera. The other order looks harmless and is not:
  # the bind is the step that refuses (a stale viewer still holding the port),
  # and if it raises after the D435 is streaming, nothing calls `close()` --
  # the SDK's destructor runs during interpreter teardown with a live pipeline
  # and aborts the process. Observed exactly that on "Address already in use".
  try:
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
  except OSError as exc:
    print(f"!! cannot serve on {args.host}:{args.port}: {exc}")
    print("   Something already holds it. Pass --port, or stop the other one.")
    return 1

  camera = None
  try:
    camera = RealSenseDepth(
      fps=args.camera_fps, preset=args.camera_preset, out_hw=depth_hw
    )
    _Handler.camera = camera
    _Handler.sim = sim
    _Handler.depth_hw = depth_hw
    print(f"http://{args.host}:{args.port}   (ctrl-c to stop)")
    print(
      "put the arm at the policy home pose and the cube where you rendered it, "
      "or every difference you see is your own."
    )
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nstopping")
  finally:
    server.server_close()
    if camera is not None:
      camera.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
