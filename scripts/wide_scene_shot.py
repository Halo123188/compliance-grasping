"""Still of the training scene at reset: three views plus what the policy sees.

  uv run python scripts/wide_scene_shot.py OUTDIR [TASK]

Writes two PNGs:

  scene_initial.png    env 0 at reset -- third-person, front and top-down views
                       of the bench, next to the D435 RGB and the depth frame
                       that is the policy's actual input.
  scene_initial_grid.png   the same reset across 8 environments, which is the
                       readable form of the cube-pose randomisation the policy
                       is trained against.

Nothing is stepped: every frame is the state the episode STARTS from, so the
arm is at its hover pose and the cube is wherever the reset event dropped it.

One env is built, and the camera is moved between renders rather than rebuilt.
That is not just cheaper -- mujoco-warp's CUDA graph capture dies on the third
camera env built in a process (see render_cases.py), so the views have to share
one build. ``OffscreenRenderer.update()`` re-reads ``cfg.env_idx`` on every call
and ``_cam`` is a plain MjvCamera, so both are safe to mutate in place.
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.env_cfgs import PALM_BODY
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

sys.path.insert(0, str(Path(__file__).parent))
from tools.depth_view import colorize_depth  # noqa: E402
from tools.task_geometry import geometry_for  # noqa: E402

OUTDIR = Path(sys.argv[1])
TASK = (
  sys.argv[2]
  if len(sys.argv) > 2
  else "Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-Success-Dr-Fine"
)
GEO = geometry_for(TASK)
N, DEV = 8, "cuda:0"
VIEW_W, VIEW_H = 640, 480

# (label, azimuth, elevation, distance, tracked body). The rollout view is
# reproduced exactly, which means tracking the robot ROOT -- and that is why the
# others do not: the tracked point is then the arm's BASE, so shortening the
# distance zooms into the base and pushes the gripper out of frame. The rest
# track the palm.
#
# A free camera aimed at the cube was tried first and is the wrong tool: the
# renderer copies one env's qpos into a single-world MjData, so a lookat has to
# be expressed in THAT model's frame, and getting it wrong points the camera at
# empty floor with nothing to say it went wrong. A tracked body id needs no frame
# conversion -- it is resolved against the same model the renderer draws.
VIEWS = (
  ("third-person (the rollout video's camera)", 125.0, -10.0, 1.05, None),
  ("front, along the bench", 175.0, -20.0, 0.75, PALM_BODY),
  ("top-down", 90.0, -75.0, 1.05, PALM_BODY),
  ("gripper close-up, at the hover pose", 125.0, -12.0, 0.30, PALM_BODY),
)
# The per-env tiles track the ROOT, which is the one body that does not move
# between them: track the palm or the cube instead and every tile re-centres on
# its own subject, hiding the pose spread the grid exists to show. Tracking the
# root also means the tracked point is the arm BASE, well above the cube, so the
# tilt has to make up the difference or the cube lands on the bottom edge.
GRID_VIEW = (125.0, -24.0, 1.10)

try:
  font = ImageFont.truetype("DejaVuSans.ttf", 20)
  small = ImageFont.truetype("DejaVuSans.ttf", 16)
except OSError:
  font = small = ImageFont.load_default()


def build():
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  # Colour as well as depth, without touching the observation group -- the same
  # trick render_student.py uses. No policy is loaded here, but keeping the
  # observation 1-channel keeps this still honest about what the net receives.
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "d435":
      sensor.data_types = ("rgb", "depth")
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    env_idx=0,
    distance=VIEWS[0][3],
    elevation=VIEWS[0][2],
    azimuth=VIEWS[0][1],
    max_extra_envs=0,
    height=VIEW_H,
    width=VIEW_W,
  )
  return ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array")


def body_id(env, name: str | None) -> int:
  """Render-model body id, resolved the way OffscreenRenderer resolves its own."""
  robot = env.scene["robot"]
  if name is None:
    return robot.indexing.root_body_id
  idx, _ = robot.find_bodies(name)
  return robot.indexing.bodies[idx[0]].id


def shoot(
  env,
  env_idx: int,
  azimuth: float,
  elevation: float,
  distance: float,
  track: str | None = None,
):
  """One frame of env ``env_idx``, from a camera moved in place."""
  renderer = env._offline_renderer  # noqa: SLF001  -- no public setter for this
  assert renderer is not None
  renderer._cfg.env_idx = env_idx  # noqa: SLF001  -- re-read by update()
  cam = renderer._cam  # noqa: SLF001
  cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
  cam.fixedcamid = -1
  cam.trackbodyid = body_id(env, track)
  frame = env.render()
  # `render()` is typed optional because it returns None under render_mode=None;
  # `build()` asks for "rgb_array", so a None here is a real failure, not a case
  # to carry through to every caller.
  assert frame is not None
  return frame


def labelled(frame: np.ndarray, text: str, sub: str = "") -> Image.Image:
  img = Image.fromarray(frame).convert("RGB")
  d = ImageDraw.Draw(img)
  d.rectangle([0, 0, img.width, 30 + (22 if sub else 0)], fill=(18, 18, 18))
  d.text((10, 5), text, font=font, fill=(240, 240, 240))
  if sub:
    d.text((10, 30), sub, font=small, fill=(160, 160, 160))
  return img


def camera_panels(env, env_idx: int, width: int, height: int):
  sensor = env.scene["d435"]
  rgb = sensor.data.rgb[env_idx].float().cpu().numpy() / 255.0
  depth = sensor.data.depth[env_idx, ..., 0].float().cpu().numpy()
  out = []
  for arr, title, sub in (
    (rgb, "D435 RGB", "not seen by the policy"),
    (
      colorize_depth(depth),
      "D435 DEPTH -- the policy's whole view of the world",
      f"{depth.shape[1]}x{depth.shape[0]}, coloured by height over the fitted"
      " table plane",
    ),
  ):
    # Nearest-neighbour: the panel is the only view of the policy's input, and
    # smoothing a 160x120 frame up to 640 px invents detail the net never had.
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img = img.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
    out.append(labelled(np.asarray(img), title, sub))
  return out


def grid(images, cols: int, pad: int = 6) -> Image.Image:
  w, h = images[0].size
  rows = (len(images) + cols - 1) // cols
  canvas = Image.new(
    "RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), (18, 18, 18)
  )
  for i, im in enumerate(images):
    r, c = divmod(i, cols)
    canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
  return canvas


OUTDIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
env = build()
env.reset()

cube = env.scene["cube"]
z = cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - GEO.surface_z
print(f"cube height above the work surface at reset: {z.mul(1000).tolist()} mm")

views = [
  labelled(shoot(env, 0, az, el, d, track), name) for name, az, el, d, track in VIEWS
]
views += camera_panels(env, 0, VIEW_W, VIEW_H)
out = OUTDIR / "scene_initial.png"
imageio.imwrite(out, np.asarray(grid(views, cols=3)))
print(f"wrote {out}")

az, el, dist = GRID_VIEW
tiles = [labelled(shoot(env, i, az, el, dist), f"env {i}") for i in range(N)]
out = OUTDIR / "scene_initial_grid.png"
imageio.imwrite(out, np.asarray(grid(tiles, cols=4)))
print(f"wrote {out}")

env.close()
