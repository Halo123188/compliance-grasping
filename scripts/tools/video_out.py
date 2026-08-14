"""Resolve a render output path, and keep renders inside the repo's `videos/`.

Every clip and still this project produces belongs under
`compliance-grasping/videos/`. The reason is not tidiness: renders were being
written to `/work/yiboc`, `/work/yiboc/videos`, `/work/yiboc/renders` and
`~/cg-logs` at the same time, and by the time anyone went looking for "the video
of the best policy" there was no single place that held it -- 130 MB of clips
sat outside the repo, on a filesystem that is not backed up with it.

`video_path` takes whatever the caller was given on the command line:

  * a RELATIVE path resolves against `videos/`, so `render_student.py R2-Small`
    and `render_student.py videos/R2-Small` mean the same thing;
  * an ABSOLUTE path outside `videos/` is passed through with a warning rather
    than rejected. A one-off render to `/tmp` is legitimate; scattering the
    archive again by accident is what this is here to catch.

Parent directories are created, so callers do not each repeat `mkdir(parents=)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/tools/video_out.py -> the repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = REPO_ROOT / "videos"


def video_path(arg: str | Path, *, is_dir: bool = False) -> Path:
  """Resolve `arg` to a render output path under `videos/` where it can be.

  Args:
    arg: the path the caller was given. Relative paths resolve against
      `videos/`; absolute paths are honoured as given.
    is_dir: create `arg` itself as a directory rather than its parent. Set it
      when the script writes several files into one output directory.

  Returns:
    The resolved path, with the directory it needs already created.
  """
  path = Path(arg)
  if not path.is_absolute():
    # `videos/foo` and `foo` both mean `videos/foo`, so passing the path the
    # cleanup convention asks for is never punished by nesting it twice.
    try:
      path = VIDEOS_DIR / path.relative_to("videos")
    except ValueError:
      path = VIDEOS_DIR / path
  path = path.resolve()

  if VIDEOS_DIR not in path.parents and path != VIDEOS_DIR:
    print(
      f"[video_out] WARNING: writing outside {VIDEOS_DIR} -> {path}\n"
      "[video_out] renders belong under videos/; see CLAUDE.md.",
      file=sys.stderr,
    )

  (path if is_dir else path.parent).mkdir(parents=True, exist_ok=True)
  return path
