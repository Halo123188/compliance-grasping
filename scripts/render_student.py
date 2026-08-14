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

READ THAT 25 mm AS A FLOOR, not a bound. It is the spread among episodes that
grasp either way. An episode near the grasp/no-grasp boundary does not drift --
it FLIPS, and the peak moves by the whole lift. Measured on s_fl_rl3 (99.2% on
`eval_deploy`), every failure the selection pass found re-rendered as a success:

  env  35   selection  42 mm  ->  re-render  157.7 mm, held 213
  env 103   selection  48 mm  ->  re-render  156.0 mm, held 214
  env   9   selection  68 mm  ->  re-render  success (`case=failure` pass)

So a FAILURE CLIP IS NOT OBTAINABLE for a policy this good, at any pool size --
n=512 surfaced six failures and the retry loop still could not hold one down.
Widening the pool buys candidates, not reproducibility. Characterize the tail
from `eval_deploy` (near-miss share, `peak|fail`) and the pose audit instead.
"""

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
from tools.depth_view import colorize_depth  # noqa: E402
from tools.task_geometry import geometry_for  # noqa: E402
from tools.video_out import video_path  # noqa: E402

OUTDIR = video_path(sys.argv[1], is_dir=True)
CKPT = sys.argv[2]
TASK = (
  sys.argv[3] if len(sys.argv) > 3 else ("Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth")
)
# Read off the TASK. The wide-claw bench measures cube rise from the top of 50 mm
# of foam; using the old table top would report every peak 50 mm too high.
_GEO = geometry_for(TASK)
LIFT_HEIGHT, TABLE_H = _GEO.lift_height, _GEO.surface_z
# `n=` widens the SELECTION pool. A policy that fails 2% of the time yields one
# candidate in 64 envs, and if that one re-renders as a success there is nothing
# left to retry -- MAX_TRIES cannot help, because it indexes into a list of one.
N = next((int(a[2:]) for a in sys.argv if a.startswith("n=")), 64)
STEPS, DEV = 300, "cuda:0"
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
  #
  # ADD rgb, do not REPLACE the tuple. The wide task also asks for
  # 'segmentation', which is not an observation but feeds `camera_depth`'s
  # surface-blinding DR; overwriting data_types with ("rgb", "depth") drops it
  # and the obs term asserts at env build time.
  for sensor in cfg.scene.sensors or ():
    if getattr(sensor, "name", None) == "d435":
      existing = getattr(sensor, "data_types", ())
      sensor.data_types = tuple(dict.fromkeys((*existing, "rgb")))
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


def _load_cfg_for(ckpt: str) -> dict:
  """Which weights to pull, read off the CHECKPOINT rather than assumed.

  A distillation checkpoint keys its policy under `student_state_dict`; a PPO
  one -- which is what the FINE-TUNE of a student produces -- keys it under
  `actor_state_dict`. Asking for the wrong one is a SILENT no-op: the runner's
  load just skips a key that was not requested, so the network stays at its
  random initialization and renders as a policy that never lifts anything. That
  is what job 71978 produced for s_fl_rl3 -- peak 31.2 mm and no success clip,
  from a checkpoint that evaluates at 99.2%.
  """
  keys = torch.load(ckpt, map_location="cpu", weights_only=False).keys()
  if "student_state_dict" in keys:
    return {"student": True}
  if "actor_state_dict" in keys:
    return {"actor": True, "critic": False, "optimizer": False, "iteration": False}
  raise KeyError(f"{ckpt} has neither student_state_dict nor actor_state_dict")


def make_policy(env):
  a = load_rl_cfg(TASK)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
  runner.load(CKPT, load_cfg=_load_cfg_for(CKPT), strict=True, map_location=DEV)
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
# Rotated so the MEDIAN failure is tried first -- a typical clip, not the worst
# one. Guarded because a policy can have no failures at all: the loop below
# already skips an empty order, but building this list indexed into it first and
# took the whole render down with an IndexError. That is not a hypothetical --
# s_facelevel succeeds in every env of the selection pass.
fail_order = (
  [fails[len(fails) // 2]] + fails[len(fails) // 2 + 1 :] + fails[: len(fails) // 2]
  if fails
  else []
)
print(
  "  success envs (by hold, longest first): "
  + ", ".join(f"{i}(pk {cats[i][1] * 1000:.0f}mm,hold {cats[i][2]})" for i in succ[:12])
)
# The FAILURE envs too, in the order the retry loop would try them. Without this
# there is no way to feed `env=N` a failing episode, which is the only route to
# a failure clip once the build below starts tripping warp (see next comment).
print(
  "  failure envs (median-first, the retry order): "
  + (", ".join(f"{i}(pk {cats[i][1] * 1000:.0f}mm)" for i in fail_order[:12]) or "none")
)

# Each rendered clip needs its own env, and a REBUILD in a single process trips
# "CUDA graph capture failed" in warp -- error 901, preceded by a cascade of
# code 906 "would make the legacy stream depend on a capturing blocking stream".
# The threshold is not fixed: job 71981 survived two builds and died on the
# third, job 71991 died on the second. So `case=` splits the work across jobs
# but does NOT make one job safe -- the selection pass above is already build 1.
# The only single-build route is `env=N`, which returns before this loop.
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
