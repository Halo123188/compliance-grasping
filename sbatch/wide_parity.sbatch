#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=wide-parity
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export UV_NO_SYNC=1
cd /home/yiboc/compliance-grasping
uv run python scripts/wide_onnx_parity.py "$@"
