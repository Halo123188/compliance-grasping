#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=tf-coll
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
cd /home/yiboc/compliance-grasping
uv run python scripts/show_colliders.py /work/yiboc
