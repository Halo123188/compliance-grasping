"""Render a distilled student next to the camera image it is actually driving on.

  uv run python scripts/render_student.py OUTDIR CKPT [TASK] [case=X] [env=N]

Produces a success clip and a failure clip, each a side-by-side of the
third-person scene and the D435 stream the policy is driving on.

The D435 is asked for colour as well as depth, but the OBSERVATION GROUP is left
alone, so the policy's input stays 1-channel and the depth checkpoint loads. The
RGB panel is for the viewer only and is labelled as such -- building the env from
the RGB-D task instead would infer a 4-channel first conv and fail to load.

``env=N`` renders one environment directly and names the clip after what that
render pass actually produced. Prefer it: GPU physics is not bit-reproducible
here, two rollouts from identical cube poses diverge by up to 25 mm of peak
height (scripts/_chk_rollout.py), so an episode picked as a success in a
selection pass can render as a failure. Without ``env=N`` the script runs a
selection pass and re-scores each clip, retrying up to MAX_TRIES times.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import geometry_for  # noqa: E402

OUTDIR = Path(sys.argv[1])
CKPT = sys.argv[2]
TASK = (
  sys.argv[3] if len(sys.argv) > 3 else ("Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth")
)
# Read off the TASK. The wide-claw bench measures cube rise from the top of 50 mm
# of foam; using the old table top would report every peak 50 mm too high.
_GEO = geometry_for(TASK)
LIFT_HEIGHT, TABLE_H = _GEO.lift_height, _GEO.surface_z
N, STEPS, DEV = 64, 300, "cuda:0"
HOLD_STEPS = 50
SCENE_W, SCENE_H = 800, 600
# The camera panel follows the render aspect rather than assuming 16:9: the wide
# task renders 160x120 (4:3), the old one 128x72 (16:9). A fixed panel would
# stretch one of them, and the panel is the only view of what the policy sees.
PANEL_W = 356
PANEL_H = 24 + (100 if "TwoFingerWide" not in TASK else int(round(356 * 120 / 160)))
MAX_TRIES = 3


def build(render: bool, env_idx: int = 0):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}  # continuous rollout, no reset jumps mid-video

  # Ask the D435 for colour as well as depth WITHOUT touching the observation
  # group, which still contains only the depth term. The sensor gains an RGB
  # buffer for the video panel; the policy's input stays 1-channel, so the
  # checkpoint loads. Building the model against an RGB-D env instead would
  # infer a 4-channel first conv and fail to load.
  for sensor in cfg.scene.sensors or ():
    if getattr(sensor, "name", None) == "d435":
      sensor.data_types = ("rgb", "depth")
  if render:
    cfg.viewer = ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      env_idx=env_idx,
      distance=1.05,
      elevation=-10.0,
      azimuth=125.0,
      max_extra_envs=0,
      height=SCENE_H,
      width=SCENE_W,
    )
  return ManagerBasedRlEnv(
    cfg=cfg, device=DEV, render_mode="rgb_array" if render else None
  )


def make_policy(env):
  a = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
  runner.load(CKPT, load_cfg={"student": True}, strict=True, map_location=DEV)
  return wrapped, runner.get_inference_policy(device=DEV)


def camera_frame(env, env_idx: int):
  """(rgb HxWx3 float, depth HxW float) straight off the sensor."""
  sensor = env.scene["d435"]
  rgb = sensor.data.rgb[env_idx].float().cpu().numpy() / 255.0
  depth = sensor.data.depth[env_idx, ..., 0].float().cpu().numpy()
  return rgb, depth


def rollout(env, wrapped, policy, render: bool, env_idx: int = 0):
  torch.manual_seed(0)
  cube = env.scene["cube"]
  obs = wrapped.reset()[0]
  hs, scene, cams = [], [], []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    if render:
      cams.append(camera_frame(env, env_idx))
    obs, _, _, _ = wrapped.step(act)
    hs.append(
      (cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - TABLE_H).clone()
    )
    if render:
      scene.append(env.render())
  return torch.stack(hs), scene, cams


def longest_run(above_1d) -> int:
  r = c = 0
  for v in above_1d.tolist():
    c = c + 1 if v else 0
    r = max(r, c)
  return r


def classify(h_1d):
  pk = float(h_1d.max())
  held = longest_run((h_1d > LIFT_HEIGHT).cpu())
  return ("success" if held >= HOLD_STEPS else "failure"), pk, held


try:
  font = ImageFont.truetype("DejaVuSans.ttf", 22)
  small = ImageFont.truetype("DejaVuSans.ttf", 16)
except OSError:
  font = small = ImageFont.load_default()


_TURBO = colormaps["turbo"]


def colorize_depth(depth_m: np.ndarray, span: float = 0.30) -> np.ndarray:
  """Depth -> RGB coloured by height above the fitted table plane. HxWx3 in [0,1].

  No linear map of ABSOLUTE depth can show this scene. Two failed attempts,
  both measured on a rendered frame:

    min/max stretch          median panel pixel 0, mean 6.6/255 -- near-black.
                             The fingers sit centimetres from the lens and the
                             background is clamped at the 3 m cutoff, so those
                             two own the entire range.
    2-98 percentile stretch  mean 59/255 but the table saturates to one flat
                             red. The table plane recedes across ~110 mm of
                             depth while the cube stands only 50 mm proud of
                             it, so any window wide enough to hold the table is
                             six times too wide to resolve the cube.

  The table is a plane, so fit it and colour the height above it. That is the
  same data the network receives, re-referenced for display -- not extra
  information -- and it puts the cube and fingers on a scale where they show up.

  Fit in INVERSE depth, not depth. Under a pinhole camera a planar surface is
  linear in (u, v) only in 1/z; fitting z directly leaves a curved residual that
  saturates half the panel. ``sensor.data.depth`` is raw metres -- the /3.0
  normalisation belongs to the observation term, not here.

  The fit is least squares run three times with 2-sigma rejection, so the cube
  and the gripper (exactly what we want to see) cannot drag the plane toward
  themselves.
  """
  h, w = depth_m.shape
  valid = (depth_m > 0.05) & (depth_m < 2.0) & np.isfinite(depth_m)  # drop background
  disp = 1.0 / np.clip(depth_m, 0.05, None)

  vv, uu = np.mgrid[0:h, 0:w]
  A = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], axis=1)
  dd = disp.ravel()
  keep = valid.ravel().copy()
  coef = np.zeros(3)
  for _ in range(3):
    if keep.sum() < 16:
      break
    coef, *_ = np.linalg.lstsq(A[keep], dd[keep], rcond=None)
    resid = dd - A @ coef
    keep = valid.ravel() & (np.abs(resid) < 2.0 * resid[keep].std())

  z_plane = 1.0 / np.clip((A @ coef).reshape(h, w), 1e-3, None)
  height = z_plane - depth_m  # metres along the ray; positive = above the table

  rgb = _TURBO(np.clip(height / span, 0.0, 1.0))[..., :3]
  rgb[height < 0.004] = 0.10  # the table itself reads as flat dark
  rgb[~valid] = 0.03  # background
  return rgb


def panel(rgb_hwc: np.ndarray, label: str) -> Image.Image:
  """One camera panel, upscaled with nearest-neighbour so pixels stay honest."""
  img = Image.fromarray((np.clip(rgb_hwc, 0, 1) * 255).astype(np.uint8))
  img = img.convert("RGB").resize((PANEL_W, PANEL_H - 24), Image.NEAREST)
  out = Image.new("RGB", (PANEL_W, PANEL_H), (18, 18, 18))
  out.paste(img, (0, 24))
  ImageDraw.Draw(out).text((6, 4), label, font=small, fill=(230, 230, 230))
  return out


def compose(scene_frame, cam, caption: str, peak: float, held: int):
  rgb, depth = cam
  canvas = Image.new("RGB", (SCENE_W + PANEL_W, SCENE_H), (18, 18, 18))
  canvas.paste(Image.fromarray(scene_frame).resize((SCENE_W, SCENE_H)), (0, 0))

  rgb_img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
  rgb_img = rgb_img.resize((PANEL_W, PANEL_H - 24), Image.NEAREST)
  rgb_panel = Image.new("RGB", (PANEL_W, PANEL_H), (18, 18, 18))
  rgb_panel.paste(rgb_img, (0, 24))
  ImageDraw.Draw(rgb_panel).text(
    (6, 4), "D435 RGB  (NOT seen by policy)", font=small, fill=(150, 150, 150)
  )
  canvas.paste(rgb_panel, (SCENE_W, 0))
  canvas.paste(
    panel(colorize_depth(depth), "D435 DEPTH  (the policy's input)"),
    (SCENE_W, PANEL_H),
  )

  d = ImageDraw.Draw(canvas)
  d.text((14, 12), caption, font=font, fill=(255, 255, 255))
  d.text(
    (14, 40),
    f"peak {peak * 1000:.1f} mm   held {held / 50:.2f} s   bar 100 mm / 1.00 s",
    font=small,
    fill=(200, 200, 200),
  )
  d.text(
    (SCENE_W + 6, 2 * PANEL_H + 8),
    "depth panel: height above the fitted\ntable plane, 0-300 mm. Same data the\nnet sees, re-referenced to be legible.",
    font=small,
    fill=(140, 140, 140),
  )
  return np.asarray(canvas)


OUTDIR.mkdir(parents=True, exist_ok=True)

# Pass 1: find candidates.
CAPTION = {
  "success": "SUCCESS - held 1 s above 100 mm",
  "failure": "FAILURE - never held 1 s above 100 mm",
}

# `env=<idx>` renders that environment directly and names the clip after what
# the render pass ACTUALLY produced. This skips the selection env entirely --
# building a third env in one process reliably trips "CUDA graph capture
# failed" in warp -- and it cannot mislabel a clip, because the outcome is read
# off the same rollout that was drawn.
DIRECT = next((int(a[4:]) for a in sys.argv if a.startswith("env=")), None)
if DIRECT is not None:
  env = build(render=True, env_idx=DIRECT)
  wrapped, policy = make_policy(env)
  h, scene, cams = rollout(env, wrapped, policy, render=True, env_idx=DIRECT)
  env.close()
  got, pk, held = classify(h[:, DIRECT])
  OUTDIR.mkdir(parents=True, exist_ok=True)
  out = OUTDIR / f"student_{got}_env{DIRECT}.mp4"
  imageio.mimsave(
    str(out),
    [compose(scene[t], cams[t], CAPTION[got], pk, held) for t in range(len(scene))],
    fps=50,
    quality=8,
  )
  print(f"wrote {out}  (env {DIRECT}, {got}, peak {pk * 1000:.1f} mm, held {held})")
  raise SystemExit(0)

env = build(render=False)
wrapped, policy = make_policy(env)
H, _, _ = rollout(env, wrapped, policy, render=False)
env.close()

cats = [classify(H[:, i]) for i in range(N)]
n_succ = sum(c == "success" for c, _, _ in cats)
print(f"selection pass: {n_succ}/{N} success ({n_succ / N * 100:.1f}%)")

# Best success = longest hold; representative failure = the median peak among
# failures, so the clip is typical rather than the worst case.
succ = sorted(
  [i for i in range(N) if cats[i][0] == "success"], key=lambda i: -cats[i][2]
)
fails = sorted(
  [i for i in range(N) if cats[i][0] == "failure"], key=lambda i: cats[i][1]
)
fail_order = (
  [fails[len(fails) // 2]] + fails[len(fails) // 2 + 1 :] + fails[: len(fails) // 2]
)
print(
  "  success envs (by hold, longest first): "
  + ", ".join(f"{i}(pk {cats[i][1] * 1000:.0f}mm,hold {cats[i][2]})" for i in succ[:12])
)

# Each rendered clip needs its own env, and building a third one in a single
# process reliably trips "CUDA graph capture failed" in warp. Pass
# `case=success` / `case=failure` to render one per invocation.
WANT = next((a[5:] for a in sys.argv if a.startswith("case=")), None)

for want, order in (("success", succ), ("failure", fail_order)):
  if WANT is not None and want != WANT:
    continue
  if not order:
    print(f"no {want} episode in the selection pass; skipping")
    continue
  for env_idx in order[:MAX_TRIES]:
    env = build(render=True, env_idx=env_idx)
    wrapped, policy = make_policy(env)
    h, scene, cams = rollout(env, wrapped, policy, render=True, env_idx=env_idx)
    env.close()
    got, pk, held = classify(h[:, env_idx])
    if got != want:
      print(f"  env {env_idx}: re-render came out {got}, retrying")
      continue
    out = OUTDIR / f"student_{want}.mp4"
    frames = [
      compose(scene[t], cams[t], CAPTION[want], pk, held) for t in range(len(scene))
    ]
    imageio.mimsave(str(out), frames, fps=50, quality=8)
    print(f"  wrote {out}  (env {env_idx}, peak {pk * 1000:.1f} mm, held {held} steps)")
    break
  else:
    print(f"could not get a stable {want} clip in {MAX_TRIES} tries")
