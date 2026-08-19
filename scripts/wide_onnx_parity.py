"""Is the exported ONNX the same policy that was evaluated? Check, don't assume.

`deploy/` runs the ONNX export and never loads mjlab. That is only safe if the
ONNX is numerically the checkpoint that scored on the cluster, and right now
that rests on a docstring ("export_policy_to_onnx exports alg.get_policy(),
which for distillation is the student"). A wrong export does not raise -- it
deploys as a policy that behaves plausibly and scores worse, which is
indistinguishable from a sim-to-real gap and would be blamed on one.

So: drive the real env with the torch student, and at every step feed the SAME
observation through onnxruntime. Report the per-step action difference. Anything
above float noise means the deployment path is running different weights than
the evaluation did.

Also checks that `deploy/policy.py`'s hand-assembled observation reproduces the
env's `student` group exactly -- that vector is the other thing that fails
silently, since a mis-ordered 34-vector is still a valid 34-vector.

That second check is the ONLY thing that verifies a history checkpoint's
stacking order end to end. `deploy.policy.stack_history` claims the layout is
term-major and oldest-first; here the env states what it actually is, at 136
numbers a step, and a wrong claim shows up as a large obs difference rather
than as a policy that merely does worse on the bench. Run this before flying
2026-08-11_00-52-56_s_g1_hist.

  uv run python scripts/wide_onnx_parity.py TASK CKPT.pt POLICY.onnx
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy import calib  # noqa: E402
from deploy.policy import StudentPolicy  # noqa: E402

N, STEPS, DEV = 32, 60, "cuda:0"

# TF32 OFF, or this check measures the wrong thing. PyTorch uses TF32 matmuls by
# default on Ampere and later -- 10 mantissa bits, so ~1e-3 relative error --
# while onnxruntime on CPU is true fp32. Left on, the comparison reports a
# ~5e-3 action difference that is entirely the GPU's arithmetic and says nothing
# about whether the export is the right policy.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _name_dim(w: int, history: int, profile) -> str:
  """Which term, which joint and how many steps back observation dim `w` is.

  Worth the arithmetic: under history the dimension numbers stop being
  readable by eye, and "dim 94" versus "actions right_1, 2 steps back" is
  the difference between a lead and a number.
  """
  offset = 0
  for term, width in profile.obs_terms:
    block = width * history
    if w < offset + block:
      i = w - offset
      lag = history - 1 - i // width  # 0 = this step
      label = calib.JOINT_NAMES[i % width] if width > 1 else ""
      age = "" if history == 1 else f", {lag} steps back"
      return f"{term} {label}{age}".replace("  ", " ")
    offset += block
  return "out of range"


def main() -> int:
  if len(sys.argv) != 4:
    raise SystemExit(__doc__)
  task, ckpt, onnx_path = sys.argv[1:4]

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  runner.load(ckpt, load_cfg={"student": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)

  sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  # The nominal cube keeps the constructor's range check happy; the per-env
  # sizes below are what actually go into the observation.
  deployed = StudentPolicy(onnx_path, cube_half_extent=calib.CUBE_EDGE_MM / 2000.0)
  print(deployed.describe())
  # The object's half-edge per env, read the way `object_half_extent` reads it.
  # Constant over the comparison: nothing resamples the cube inside 60 steps of
  # one episode, and `cfg.terminations = {}` above removes the early resets.
  cube = env.scene["cube"]
  half = env.sim.model.geom_size[:, int(cube.indexing.geom_ids[0]), 0].cpu().numpy()

  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  act_diff, obs_diff = [], []
  last_action = np.zeros((N, 11), dtype=np.float32)
  # (N, H, frame_dim), oldest first. H comes from the checkpoint, so a plain policy
  # runs the same code with H = 1 and no stacking to get wrong. Safe to fill
  # over the whole window rather than tracking resets: `cfg.terminations = {}`
  # above leaves only the time limit, and STEPS is 60 steps of a much longer
  # episode, so no env resets inside the comparison.
  hist = np.zeros((N, deployed.history_length, deployed.profile.frame_dim), np.float32)
  first = True

  for _ in range(STEPS):
    with torch.inference_mode():
      torch_act = policy(obs)

    groups = env.observation_manager.compute()
    student = groups["student"].cpu().numpy()
    camera = groups["camera"].cpu().numpy()

    # 1. Does deploy/policy.py's hand-assembled obs match the env's?
    robot = env.scene["robot"]
    # joint_pos_BIASED, not joint_pos. The DR env sets `biased=True` on this
    # term so the policy sees the same wrong angle the position command is
    # corrected by -- otherwise it observes the encoder error and cancels it,
    # which makes encoder_bias a free no-op. The bias is a MODEL of the real
    # robot's encoder error, so on hardware there is nothing to add: the real
    # reading already is the biased one, and deploy/policy.py feeding the raw
    # measurement is correct. Reading unbiased here compares the deployment
    # against a vector the policy never saw.
    jp = robot.data.joint_pos_biased.cpu().numpy()
    jv = robot.data.joint_vel.cpu().numpy()
    # The NEWEST goal_height, which is the last element of either layout.
    gh = student[:, -1]
    # Assembled by the DEPLOYMENT's own code, term list and all, so this check
    # covers round 2's extra `cube_size` term rather than a hand-written copy of
    # it. `frame()` reads `last_action` off the policy, so it is set per env
    # before each call; nothing else here is stateful.
    frames = np.zeros((N, deployed.profile.frame_dim), np.float32)
    for i in range(N):
      deployed.last_action = last_action[i]
      frames[i] = deployed.frame(jp[i], jv[i], float(gh[i]), float(half[i]))
    # Backfill then roll, exactly as `StudentPolicy.observe` does and as
    # `CircularBuffer` does on the push after a reset.
    if first:
      hist[:] = frames[:, None, :]
      first = False
    else:
      hist[:, :-1] = hist[:, 1:]
      hist[:, -1] = frames
    mine = np.stack([deployed.stack_history(hist[i]) for i in range(N)])
    obs_diff.append(np.abs(mine - student).max())

    # 2. Does the ONNX reproduce the torch student on the env's own obs?
    onnx_act = np.concatenate(
      [
        sess.run(
          ["actions"],
          {"obs": student[i : i + 1], "camera": camera[i : i + 1]},
        )[0]
        for i in range(N)
      ]
    )
    act_diff.append(np.abs(onnx_act - torch_act.cpu().numpy()).max())

    last_action = torch_act.cpu().numpy().astype(np.float32)
    obs = wrapped.step(torch_act)[0]

  env.close()
  ad, od = np.array(act_diff), np.array(obs_diff)
  print(f"\n{'=' * 62}")
  print(f"action  |onnx - torch|   max {ad.max():.3e}   mean {ad.mean():.3e}")
  print(f"obs     |deploy - env|   max {od.max():.3e}   mean {od.mean():.3e}")
  if od.max() >= 1e-4:
    # A constant offset here is almost always a rounded constant in calib.py,
    # so name the term rather than just failing.
    per = np.abs(mine - student).max(axis=0)
    w = int(per.argmax())
    print(
      f"worst obs dim {w} ({_name_dim(w, deployed.history_length, deployed.profile)})"
      f" by {per[w]:.3e}"
    )
  # float32 through two different runtimes; anything at 1e-3 is a real
  # difference in weights or preprocessing, not accumulated rounding.
  ok = ad.max() < 1e-3 and od.max() < 1e-4
  print(
    f"\n{'PASS' if ok else 'FAIL'}: the deployment path {'is' if ok else 'is NOT'} the evaluated policy."
  )
  print(f"deploy default_joint_pos check passed at load ({deployed.default[:3]}...)")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
