#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=tf-render
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
cd /home/yiboc/compliance-grasping
mkdir -p /work/yiboc
uv run python scripts/render_scene.py && uv run python scripts/render_occlusion.py
echo "[render] OK"
