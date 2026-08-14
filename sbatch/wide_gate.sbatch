#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=spawn-gate
#SBATCH --exclude=bos14-node-086,bos14-node-092
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 UV_NO_SYNC=1
cd /home/yiboc/compliance-grasping
uv run python scripts/wide_spawn_gate.py "$@"
