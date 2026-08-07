#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=tf-smoke
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
cd /home/yiboc/compliance-grasping

TASK="${1:-Mjlab-Grasp-TwoFinger-Flexiv}"
echo "[smoke] task=$TASK  node=$(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Tiny run: prove the env instantiates on GPU and a few PPO iters complete.
uv run python src/mjlab/scripts/train.py "$TASK" \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard \
  --video False \
  --agent.run-name smoke
echo "[smoke] OK"
