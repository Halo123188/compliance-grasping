"""Render one clip per outcome CATEGORY, scored from the render pass itself.

  uv run python scripts/render_cases.py OUTDIR NAME=TASK:CKPT:CASE [...]

CASE is one of:
  below80      peak never exceeded 80 mm  -- the cube was essentially never lifted
  lift_nohold  peak > 80 mm but never held 1 s above the 100 mm bar
  lift_hold    peak > 80 mm and succeeded

Why categories instead of "success/failure": the peak-height CDF (job 47186)
showed 98.4% / 96.9% of envs clear 80 mm, so "failure" almost never means "failed
to lift". AlignCurr's dominant failure is peak 102.1 mm held for 7 of 50 steps --
it clears the bar and drops. Splitting on 80 mm separates the two.

Selection runs one pass, then each clip is RE-SCORED from its own render pass and
rejected if the category no longer holds: GPU physics is not bit-reproducible and
two rollouts from identical cube poses diverge by up to 25.1 mm of peak height
(scripts/_chk_rollout.py). Up to MAX_TRIES candidates are tried per category.

ONE CAMERA ENV PER PROCESS. mujoco-warp's CUDA graph capture dies on the THIRD
camera env built in a process ("operation not permitted when stream is
capturing"), and the obvious structure costs one build for the selection pass
plus one per render attempt. So the top-level invocation is a pure DRIVER that
never builds anything; it shells out to itself once for selection and once per
render attempt. State-only tasks are unaffected, but they go through the same
path so there is only one code path to be right.

  --select OUTDIR SPEC   child: score N envs, print candidate indices as JSON
  --clip IDX OUTDIR SPEC child: render env IDX, exit 3 if its category flipped
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import TaskGeometry, geometry_for  # noqa: E402
from tools.video_out import video_path  # noqa: E402

ARGV = sys.argv[1:]
MODE = "drive"
CLIP_IDX = -1
if ARGV and ARGV[0] == "--select":
  MODE, ARGV = "select", ARGV[1:]
elif ARGV and ARGV[0] == "--clip":
  MODE, CLIP_IDX, ARGV = "clip", int(ARGV[1]), ARGV[2:]

OUTDIR = video_path(ARGV[0], is_dir=True)
SPECS = ARGV[1:]
N, STEPS, DEV = 128, 300, "cuda:0"
HOLD_STEPS = 50
LIFT80 = 0.080
# Set per spec from the TASK id. The wide-claw bench measures cube height from
# the top of 50 mm of foam, not from the table -- hardcoding the old constant
# shifts every peak by 50 mm in the direction that makes a working policy look
# like it never lifted. See scripts/tools/task_geometry.py.
GEO: TaskGeometry
W, H_PX = 960, 720
MAX_TRIES = 8  # each try is its own process now, so retries are cheap and safe


def build(task: str, render: bool, env_idx: int = 0):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  if render:
    cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      env_idx=env_idx,
      distance=1.05,
      elevation=-10.0,
      azimuth=125.0,
      max_extra_envs=0,
      height=H_PX,
      width=W,
    )
  return ManagerBasedRlEnv(
    cfg=cfg, device=DEV, render_mode="rgb_array" if render else None
  )


def make_policy(task: str, ckpt: str, env):
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  # A distilled checkpoint stores the camera student under a different key; ask
  # for "actor" and it loads nothing and renders an untrained policy.
  is_distilled = "student_state_dict" in torch.load(
    ckpt, map_location="cpu", weights_only=False
  )
  load_cfg = {"student": True} if is_distilled else {"actor": True}
  runner.load(ckpt, load_cfg=load_cfg, strict=True, map_location=DEV)
  return wrapped, runner.get_inference_policy(device=DEV)


def rollout(env, wrapped, policy, render: bool):
  torch.manual_seed(0)
  cube = env.scene["cube"]
  obs = wrapped.reset()[0]
  hs, frames = [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    hs.append(
      (
        cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - GEO.surface_z
      ).clone()
    )
    if render:
      frames.append(env.render())
  return torch.stack(hs), frames


def longest_run(above_1d) -> int:
  r = c = 0
  for v in above_1d.tolist():
    c = c + 1 if v else 0
    r = max(r, c)
  return r


def classify(h_1d):
  """-> (case, peak, held) for a single env's height trace."""
  pk = float(h_1d.max())
  held = longest_run((h_1d > GEO.lift_height).cpu())
  if pk <= LIFT80:
    return "below80", pk, held
  return ("lift_hold" if held >= HOLD_STEPS else "lift_nohold"), pk, held


CAPTION = {
  "below80": "NEVER LIFTED (peak under 80 mm)",
  "lift_nohold": "LIFTED BUT DROPPED (over 80 mm, hold failed)",
  "lift_hold": "SUCCESS (held 1 s over 100 mm)",
}

try:
  font = ImageFont.truetype("DejaVuSans.ttf", 26)
  small = ImageFont.truetype("DejaVuSans.ttf", 20)
except OSError:
  font = small = ImageFont.load_default()


def do_select(task: str, ckpt: str, case: str) -> None:
  """Child: score N envs, print the candidate indices for `case` as JSON."""
  e = build(task, render=False)
  w, p = make_policy(task, ckpt, e)
  hh, _ = rollout(e, w, p, render=False)
  e.close()
  info = [classify(hh[:, i]) + (i,) for i in range(N)]
  cats = [t[0] for t in info]
  print(
    f"[{task.split('-')[-1]}] "
    + ", ".join(f"{c} {cats.count(c)}" for c in CAPTION)
    + f" (of {N})"
  )
  cands = sorted((t for t in info if t[0] == case), key=lambda t: t[1])
  # Most typical first: sort by peak, take from the middle outwards.
  order = sorted(range(len(cands)), key=lambda k: abs(k - len(cands) // 2))
  print("CANDIDATES " + json.dumps([cands[k][3] for k in order]))


def drive() -> None:
  """Parent: never builds an env, so the CUDA-graph limit cannot be hit."""
  sel_cache: dict[tuple[str, str, str], list[int]] = {}
  for spec in SPECS:
    name, rest = spec.split("=", 1)
    task, ckpt, case = rest.rsplit(":", 2)
    key = (task, ckpt, case)
    if key not in sel_cache:
      r = subprocess.run(
        [sys.executable, __file__, "--select", str(OUTDIR), spec],
        env=os.environ,
        capture_output=True,
        text=True,
      )
      line = next(
        (ln for ln in r.stdout.splitlines() if ln.startswith("CANDIDATES ")), ""
      )
      if not line:
        print(f"!! {name}: selection failed\n{(r.stdout + r.stderr)[-1200:]}")
        continue
      print(next(ln for ln in r.stdout.splitlines() if ln.startswith("[")))
      sel_cache[key] = json.loads(line.removeprefix("CANDIDATES "))
    cands = sel_cache[key]
    if not cands:
      print(f"!! {name}: no env in category {case}; skipping")
      continue
    print(f"{name}: {len(cands)} candidates for {case}")
    for idx in cands[:MAX_TRIES]:
      r = subprocess.run(
        [sys.executable, __file__, "--clip", str(idx), str(OUTDIR), spec],
        env=os.environ,
        capture_output=True,
        text=True,
      )
      print("\n".join(ln for ln in r.stdout.splitlines() if ln.startswith("  ")))
      if r.returncode == 0:
        break
      if r.returncode != 3:
        print(f"!! {name}: clip child failed\n{(r.stdout + r.stderr)[-1200:]}")
        break
    else:
      print(f"!! {name}: no candidate reproduced {case} in {MAX_TRIES} tries")


def do_clip(name: str, task: str, ckpt: str, case: str, idx: int) -> int:
  """Child: render ONE env. Returns 3 if its category flipped on this pass."""
  e = build(task, render=True, env_idx=idx)
  w, p = make_policy(task, ckpt, e)
  hh, frames = rollout(e, w, p, render=True)
  e.close()
  frames = [fr for fr in frames if fr is not None]

  got, pk, held = classify(hh[:, idx])
  print(f"  env {idx}: render pass peak {pk * 1000:.1f} mm, held {held} -> {got}")
  if got != case:
    print("    category flipped on the render pass; trying another env")
    return 3

  out = []
  for t, fr in enumerate(frames):
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im)
    h_mm = float(hh[t, idx]) * 1000
    ok = h_mm > GEO.lift_height * 1000
    d.text((14, 12), f"{name}  env {idx}", fill=(255, 255, 255), font=font)
    d.text((14, 44), CAPTION[case], fill=(255, 235, 160), font=small)
    d.text(
      (14, 70),
      f"cube {h_mm:6.1f} mm   bar {GEO.lift_height * 1000:.0f} mm   80 mm line",
      fill=(130, 255, 150) if ok else (255, 180, 120),
      font=small,
    )
    d.text(
      (14, 96),
      f"peak {pk * 1000:.1f} mm   held {held} of {HOLD_STEPS} steps",
      fill=(190, 190, 190),
      font=small,
    )
    x0, y1 = 40, H_PX - 40
    scale = 1.6  # px per mm
    d.rectangle(
      [x0, y1 - GEO.lift_height * 1000 * scale, x0 + 8, y1], outline=(120, 120, 120)
    )
    d.line(
      [x0 - 6, y1 - LIFT80 * 1000 * scale, x0 + 14, y1 - LIFT80 * 1000 * scale],
      fill=(255, 120, 120),
      width=2,
    )
    d.rectangle(
      [x0, y1 - max(h_mm, 0) * scale, x0 + 8, y1],
      fill=(130, 255, 150) if ok else (255, 180, 120),
    )
    out.append(np.asarray(im))

  path = OUTDIR / f"{name}.mp4"
  imageio.mimwrite(path, out, fps=30, quality=8)
  print(f"  wrote {path}  peak {pk * 1000:.1f} mm  held {held}")
  return 0


if MODE == "drive":
  drive()
else:
  _name, _rest = SPECS[0].split("=", 1)
  _task, _ckpt, _case = _rest.rsplit(":", 2)
  GEO = geometry_for(_task)
  if MODE == "select":
    do_select(_task, _ckpt, _case)
  else:
    raise SystemExit(do_clip(_name, _task, _ckpt, _case, CLIP_IDX))
