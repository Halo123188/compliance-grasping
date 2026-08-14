"""Success against object SIZE, and against WHERE on the bench the object is.

  uv run python scripts/wide_eval_sweep.py OUTDIR NAME=TASK:CKPT [NAME=TASK:CKPT ...]
  uv run python scripts/wide_eval_sweep.py OUTDIR NAME=TASK:CKPT envs=256 sweep=size
  uv run python scripts/wide_eval_sweep.py OUTDIR NAME=TASK:CKPT sweep=grid \
      box=0.32,0.72,-0.36,0.36 grid=8x9

`eval_deploy.py` scores a policy on ONE distribution -- its own training
distribution, averaged over every size and every spawn at once. Three of the
four round-1/round-2 students sit at 96-100% there, which says nothing about
WHERE the remaining failures are. This splits that single number two ways:

  size   the cube is pinned to one edge length, swept 15-60 mm, with the spawn
         left at the task's own box. The sweep deliberately runs OUTSIDE every
         policy's training range at both ends (R1 trained on 42.5-57.5 mm,
         R2-Small on 15-57.5) -- the point is where each one falls off.
  grid   the cube is pinned to 50 mm and the spawn box is cut into cells, one
         rollout per cell. The trained box is +-100 mm in x and +-120 mm in y
         about the manipulation centre, and it is NOT symmetric in what it asks
         of the arm: the far corner is a longer reach and a shallower camera
         view than the near one. `box=` sweeps a WIDER area than the policy was
         trained on -- out to the bench's own edges -- where the limit stops
         being the policy: the D435's cone and 200 mm minimum range cut off the
         near corners and the far sides, and the arm passes 1.5 sigma of action
         beyond ~720 mm. The plot draws both boundaries.

Both are driven by mutating the LIVE cfg between rollouts rather than by
rebuilding the env, which is what makes 35 rollouts affordable -- the scene
compile and the checkpoint load happen once per policy.

TWO THINGS THAT WOULD SILENTLY CORRUPT THIS, both already burnt elsewhere in
this directory:

  * `LiftingCommand` RESPAWNS the cube every resample, and the play cfg resamples
    every 5 s against a 6 s rollout. Left alone, every env teleports its cube at
    step 250 -- out of the hand, if the policy is holding it. Disabled here.
  * the size draw is a RESET event, so it has to be re-read after each reset
    rather than assumed: what is recorded per episode is the geom's actual
    half-edge, not the value that was asked for.

The output is one .npz per policy holding PER-EPISODE rows, not summaries --
peak height, hold length, spawn x/y, cube half-edge. The success bar is a choice
this script should not be making for the reader: the task's own bar is an
ABSOLUTE height of the cube CENTRE (100 mm above the work surface), so a 15 mm
cube has to climb 92.5 mm to clear it and a 60 mm cube only 70 mm. Keeping the
raw rows lets `wide_plot_sweep.py` show both that bar and a size-fair one.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import TaskGeometry, geometry_for  # noqa: E402
from wide_speed_audit import _load_policy  # noqa: E402

STEPS, DEV = 300, "cuda:0"
HOLD = 50  # steps above the bar that count as held, at 50 Hz -> 1 s
NOMINAL_MM = 50.0  # the cube the scale multiplier is relative to
SIZES_MM = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)
GRID_NX, GRID_NY = 5, 5


def _rollout(
  env: ManagerBasedRlEnv,
  wrapped: RslRlVecEnvWrapper,
  policy,
  geo: TaskGeometry,
  seed: int,
) -> dict[str, np.ndarray]:
  """One reset-to-end window under whatever the cfg currently says.

  Seeded identically at every sweep point on purpose. The DR draws are consumed
  in the same order regardless of the ranges, so cube yaw, mass and friction are
  PAIRED across sizes and across cells -- the difference between two points is
  then the thing that was swept, not which spawns happened to be lucky.
  """
  cube = env.scene["cube"]
  torch.manual_seed(seed)
  obs = wrapped.reset()[0]

  # After the reset: `dr_cube_scale` and the spawn both happen there.
  cube_geom = int(cube.indexing.geom_ids[0])
  half = env.sim.model.geom_size[:, cube_geom, 0].clone()
  origin = env.scene.env_origins
  spawn = (cube.data.root_link_pos_w[:, :2] - origin[:, :2]).clone()

  heights = []
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)
    heights.append(
      (cube.data.root_link_pos_w[:, 2] - origin[:, 2] - geo.surface_z).clone()
    )
  h = torch.stack(heights).cpu()  # [T, B], cube CENTRE above the work surface

  # Longest run above the bar, which is the task's own success test: peak alone
  # counts a cube that was flung past 100 mm and dropped.
  above = h > geo.lift_height
  run = torch.zeros(h.shape[1], dtype=torch.long)
  cur = torch.zeros_like(run)
  for t in range(h.shape[0]):
    cur = torch.where(above[t], cur + 1, torch.zeros_like(cur))
    run = torch.maximum(run, cur)
  return dict(
    peak=h.max(0).values.numpy(),
    hold=run.numpy(),
    half=half.cpu().numpy(),
    spawn_x=spawn[:, 0].cpu().numpy(),
    spawn_y=spawn[:, 1].cpu().numpy(),
    # The whole trace, so the SUCCESS BAR stays the reader's choice. The task's
    # own bar is absolute (cube CENTRE 100 mm up), which asks a 15 mm cube for
    # 92.5 mm of climb and a 60 mm one for 70 -- a size sweep scored on it alone
    # cannot separate "small cubes are harder to hold" from "small cubes are
    # scored harder". float16 resolves 0.12 mm at these heights, which is two
    # orders below anything read off it.
    h=h.numpy().astype(np.float16),
  )


def _succ(r: dict[str, np.ndarray]) -> float:
  return float((r["hold"] >= HOLD).mean())


def sweep(
  task: str,
  ckpt: str,
  role: str,
  num_envs: int,
  which: str,
  grid_box: tuple[float, float, float, float] | None = None,
  grid_shape: tuple[int, int] = (GRID_NX, GRID_NY),
) -> dict:
  geo = geometry_for(task)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  # One continuous window per env, and no mid-window respawn. See the module
  # docstring: the play cfg resamples at 5 s against a 6 s rollout, and a
  # resample teleports the cube.
  cfg.terminations = {}
  cfg.commands["lift_height"].resampling_time_range = (1e9, 1e9)

  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  policy = _load_policy(runner, ckpt, role)

  # The live cfg objects the two sweeps steer. Fetched through the managers
  # rather than off `cfg`, so this cannot quietly mutate a copy the running env
  # is not reading.
  scale = env.event_manager.get_term_cfg("dr_cube_scale").params
  pose = env.command_manager.get_term_cfg("lift_height").object_pose_range
  assert pose is not None
  box_x, box_y = tuple(pose.x), tuple(pose.y)
  out: dict[str, np.ndarray | float | str] = dict(
    task=task,
    ckpt=ckpt,
    num_envs=num_envs,
    steps=STEPS,
    hold_steps=HOLD,
    lift_height=geo.lift_height,
    surface_z=geo.surface_z,
    box_x=np.asarray(box_x),
    box_y=np.asarray(box_y),
  )

  if which in ("size", "both"):
    print(f"\n  size sweep, spawn over the task's own box x{box_x} y{box_y}")
    rows = []
    for mm in SIZES_MM:
      scale["scale_range"] = (mm / NOMINAL_MM, mm / NOMINAL_MM)
      r = _rollout(env, wrapped, policy, geo, seed=0)
      rows.append(r)
      print(
        f"    {mm:4.0f} mm  (half {r['half'].mean() * 1000:5.2f} mm)"
        f"  success {_succ(r) * 100:6.1f}%   median peak"
        f" {np.median(r['peak']) * 1000:6.1f} mm"
      )
    out["sizes_mm"] = np.asarray(SIZES_MM)
    for k in ("peak", "hold", "half", "spawn_x", "spawn_y"):
      out[f"size_{k}"] = np.stack([r[k] for r in rows])  # [S, B]
    out["size_h"] = np.stack([r["h"] for r in rows])  # [S, T, B]

  if which in ("grid", "both"):
    # 50 mm for every cell, so the map reads position and not size.
    scale["scale_range"] = (1.0, 1.0)
    nx, ny = grid_shape
    # `grid_box` sweeps somewhere the policy was never trained -- which is the
    # point of passing it, and also why the trained box is recorded above and
    # drawn on the figure. Outside it BOTH the policy and the bench change: the
    # D435's cone and its 200 mm minimum range cut off the near corners and the
    # far sides, and the arm runs past 1.5 sigma of action beyond ~720 mm. The
    # plot overlays those two boundaries from `wide_spawn_gate.feasibility_map`,
    # so a dead cell can be read as blind or out of reach rather than as a
    # policy that cannot grasp there.
    gx = (box_x[0], box_x[1], box_y[0], box_y[1]) if grid_box is None else grid_box
    xe = np.linspace(gx[0], gx[1], nx + 1)
    ye = np.linspace(gx[2], gx[3], ny + 1)
    print(
      f"\n  grid sweep, {nx}x{ny} cells over"
      f" x [{gx[0]:.3f},{gx[1]:.3f}] y [{gx[2]:.3f},{gx[3]:.3f}]"
      f"{'  (the trained box)' if grid_box is None else '  (WIDENED)'}"
    )
    rows = []
    for i in range(nx):
      for j in range(ny):
        # The cell, not its centre: a heatmap cell is an AREA, and sampling its
        # interior averages the cell the way the picture claims to.
        pose.x = (float(xe[i]), float(xe[i + 1]))
        pose.y = (float(ye[j]), float(ye[j + 1]))
        r = _rollout(env, wrapped, policy, geo, seed=0)
        rows.append(r)
        print(
          f"    x [{xe[i]:+.3f},{xe[i + 1]:+.3f}] y [{ye[j]:+.3f},{ye[j + 1]:+.3f}]"
          f"  success {_succ(r) * 100:6.1f}%   median peak"
          f" {np.median(r['peak']) * 1000:6.1f} mm"
        )
    pose.x, pose.y = box_x, box_y
    out["grid_x_edges"] = xe
    out["grid_y_edges"] = ye
    out["grid_shape"] = np.asarray((nx, ny))
    for k in ("peak", "hold", "half", "spawn_x", "spawn_y"):
      out[f"grid_{k}"] = np.stack([r[k] for r in rows]).reshape(nx, ny, num_envs)
    out["grid_h"] = np.stack([r["h"] for r in rows]).reshape(nx, ny, STEPS, num_envs)

  env.close()
  return out


def main() -> None:
  outdir = Path(sys.argv[1])
  outdir.mkdir(parents=True, exist_ok=True)
  args = sys.argv[2:]
  num_envs = next((int(a[5:]) for a in args if a.startswith("envs=")), 256)
  which = next((a[6:] for a in args if a.startswith("sweep=")), "both")
  assert which in ("size", "grid", "both"), which
  # box=x0,x1,y0,y1 in metres widens the grid past the trained spawn box;
  # grid=NXxNY sets its resolution. Both default to the trained box at 5x5.
  box_arg = next((a[4:] for a in args if a.startswith("box=")), None)
  grid_box = None
  if box_arg is not None:
    parts = tuple(float(v) for v in box_arg.split(","))
    assert len(parts) == 4, "box=x0,x1,y0,y1"
    grid_box = (parts[0], parts[1], parts[2], parts[3])
  shape_arg = next((a[5:] for a in args if a.startswith("grid=")), None)
  shape = (GRID_NX, GRID_NY)
  if shape_arg is not None:
    a_, b_ = shape_arg.lower().split("x")
    shape = (int(a_), int(b_))

  opts = ("envs=", "sweep=", "box=", "grid=")
  for spec in (a for a in args if "=" in a and not a.startswith(opts)):
    name, rest = spec.split("=", 1)
    task, ckpt = rest.split(":", 1)
    role = "policy"
    if ckpt.endswith((":teacher", ":student")):
      ckpt, role = ckpt.rsplit(":", 1)
      role = "teacher" if role == "teacher" else "policy"
    print(f"\n{'=' * 78}\n{name}   {task}\n  {ckpt}\n{'=' * 78}")
    out = sweep(task, ckpt, role, num_envs, which, grid_box, shape)
    out["name"] = name
    path = outdir / f"{name}.npz"
    np.savez(path, **out)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
  main()
