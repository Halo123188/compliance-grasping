#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=wide-cases
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# uv re-syncs the venv on every `uv run`, taking an flock on .venv/.lock. That
# lock lives on NFS, and two jobs starting on different nodes at the same time
# deadlock on it -- silently, with no output at all, for as long as the job
# runs. mjlab is installed as a path .pth pointing at src/, so the sync has
# nothing to do anyway: source edits are already live without a rebuild.
export UV_NO_SYNC=1
cd /home/yiboc/compliance-grasping

# One clip per outcome category:
#   sbatch sbatch/wide_cases.sbatch OUTDIR NAME=TASK:CKPT:CASE [...]
uv run python scripts/render_cases.py "$@"
