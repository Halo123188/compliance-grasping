import os
import sys

# Default to EGL for GPU-accelerated offscreen rendering on Linux. Must be set
# before any mujoco import: mujoco's gl_context module captures MUJOCO_GL once
# at load time. Override with e.g. MUJOCO_GL=osmesa on clusters without EGL.
# Linux-only because mujoco's gl_context rejects "egl" on macOS/Windows and
# raises at import. On those platforms we leave MUJOCO_GL alone so mujoco
# defaults to GLFW.
if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")

import traceback
from importlib.metadata import entry_points
from pathlib import Path

import tyro
import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent

TYRO_FLAGS = (
  # Don't let users switch between types in unions. This produces a simpler CLI
  # with flatter helptext, at the cost of some flexibility. Type changes can
  # just be done in code.
  tyro.conf.AvoidSubcommands,
  # Disable automatic flag conversion (e.g., use `--flag False` instead of
  # `--no-flag` for booleans).
  tyro.conf.FlagConversionOff,
  # Use Python syntax for collections: --tuple (1,2,3) instead of --tuple 1 2 3.
  # Helps with wandb sweep compatibility: https://brentyi.github.io/tyro/wandb_sweeps/
  tyro.conf.UsePythonSyntaxForLiteralCollections,
)


def _configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  # NVRTC precompiled headers OFF. They are a large blob NVRTC maps into its own
  # arena, and on a big enough scene that mapping FAILS -- the compile then dies
  # with
  #
  #     NVRTC compilation error 6: NVRTC_ERROR_COMPILATION
  #     Catastrophic error: unable to obtain mapped memory
  #
  # which names neither the scene nor the header cache, and is not fixed by
  # giving the job more memory (measured: it fails at 2.1 GB RSS with 96 GB
  # requested). It presents as "this environment cannot be built at all" on the
  # first kernel that happens to be large.
  #
  # Bisected on 2026-08-06 while bringing up the wide-claw scene, which hit it on
  # three separate kernels (`primitive_narrowphase`,
  # `update_gradient_JTDAJ_dense_tiled`, `render.__locals__._render_megakernel`).
  # Every scene-side workaround failed: dropping geom groups until only meshes
  # were renderable, shrinking the camera to 80x60, disabling textures, cutting
  # njmax. Turning PCH off fixes all three, and it is the only thing that does --
  # note that `max_unroll = 1` does NOT help and re-breaks it when combined.
  #
  # The cost is compile time only, and only on a cache miss. Set
  # MJLAB_WARP_PCH=1 to restore the default if a future warp fixes the arena.
  wp.config.use_precompiled_headers = os.environ.get("MJLAB_WARP_PCH", "0").lower() in (
    "1",
    "true",
    "yes",
  )

  # Keep warp verbose by default to show kernel compilation progress.
  # Override with MJLAB_WARP_QUIET=1 environment variable if needed.
  quiet = os.environ.get("MJLAB_WARP_QUIET", "0").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


def _import_registered_packages() -> None:
  """Auto-discover and import packages registered via entry points.

  Looks for packages registered under the 'mjlab.tasks' entry point group.
  Each discovered package is imported, which allows it to register custom
  environments with gymnasium.
  """
  mjlab_tasks = entry_points().select(group="mjlab.tasks")
  for entry_point in mjlab_tasks:
    try:
      entry_point.load()
    except Exception:
      print(
        f"[WARN] Failed to load task package '{entry_point.name}' ({entry_point.value}):",
        file=sys.stderr,
      )
      traceback.print_exc(file=sys.stderr)


def _configure_mediapy() -> None:
  """Point mediapy at the bundled imageio-ffmpeg binary."""
  import imageio_ffmpeg
  import mediapy

  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


_configure_warp()
try:
  _configure_mediapy()
except Exception:
  pass
_import_registered_packages()
