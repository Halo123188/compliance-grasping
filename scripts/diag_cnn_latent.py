"""Is the CNN encoder emitting anything, or a constant?

  uv run python scripts/diag_cnn_latent.py TASK STUDENT_CKPT

A behaviour loss that sits at the constant-predictor value tells you the student
learned nothing, but not WHY. This separates the two candidates:

  * the encoder outputs vary with the scene -> the image is being read, and the
    fault is downstream (head capacity, optimization, target).
  * the encoder outputs are near-constant -> the image is not being read at all,
    and no amount of training the head will help.

The specific failure this was written for: mjlab's ``SpatialSoftmax`` uses a
FIXED temperature of 1.0. A randomly-initialised conv trunk emits features of
order 0.1-1, and a softmax at temperature 1 over the 144 cells of a 16x9 map is
then nearly uniform -- so every channel's soft-argmax lands on the image centre
regardless of what is in the frame, and the keypoints carry no signal. It is the
same end state as global average pooling, which the sibling cubegrasp repo
measured stalling its behaviour loss at 1.0-1.7.

Reports, per keypoint dimension: the standard deviation ACROSS the batch (how
much it moves when the scene moves) and the softmax entropy as a fraction of the
uniform maximum (1.0 = fully collapsed, 0 = a single sharp peak).
"""

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, CKPT = sys.argv[1], sys.argv[2]
N, DEV = 64, "cuda:0"

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
a = load_rl_cfg(TASK)
wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
runner = load_runner_cls(TASK)(wrapped, asdict(a), device=DEV)
runner.load(CKPT, load_cfg={"student": True}, strict=True, map_location=DEV)
student = runner.alg.get_policy()

torch.manual_seed(0)
obs = wrapped.reset()[0]
for _ in range(10):
  with torch.inference_mode():
    obs = wrapped.step(student(obs))[0]

group = student.obs_groups_2d[0]
encoder = student.cnns[group]
with torch.inference_mode():
  image = obs[group]
  feats = encoder.cnn(image)  # (B, C, h, w), pre-softmax
  keypoints = encoder(image)  # (B, C*2)

f = feats.float()
B, C, h, w = f.shape
flat = f.reshape(B, C, -1)
weights = torch.softmax(flat / encoder.spatial_softmax.temperature, dim=-1)
entropy = -(weights * (weights + 1e-12).log()).sum(-1)
uniform = float(np.log(h * w))

k = keypoints.float().cpu().numpy()
print(f"task={TASK}\nstudent={CKPT}\n")
print(f"feature map: {C} channels x {h}x{w} = {h * w} cells")
print(f"softmax temperature: {encoder.spatial_softmax.temperature}")
print(f"pre-softmax feature std: {float(f.std()):.4f} (per-channel spread)")
print(
  f"\nsoftmax entropy / uniform: mean {float(entropy.mean()) / uniform:.4f} "
  f"min {float(entropy.min()) / uniform:.4f}"
)
print("  1.0 = uniform over the whole map, i.e. every keypoint pins to the centre")
print(
  f"\nkeypoint std ACROSS the {N} envs: mean {k.std(axis=0).mean():.5f} "
  f"max {k.std(axis=0).max():.5f}"
)
print(f"keypoint |mean| across envs:      {np.abs(k.mean(axis=0)).mean():.5f}")
print(
  "  keypoints live in [-1, 1]; a std of ~0 means the encoder returns the same"
  "\n  numbers no matter where the cube is."
)
env.close()
