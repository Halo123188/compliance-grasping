#!/usr/bin/env bash
# Generic job-array runner: one training command per line of the cmds file.
#   sbatch --array=0-<N-1>%<throttle> run_array.sbatch <cmds_file> <tag>
#SBATCH --job-name=compliance_arr
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cmds_file="$1"
tag="${2:-run}"
cmd="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$cmds_file")"
echo "[task $SLURM_ARRAY_TASK_ID] $cmd"
eval "$cmd" > "slurm_log/${tag}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
