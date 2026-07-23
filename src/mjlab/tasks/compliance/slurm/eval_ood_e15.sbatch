#!/usr/bin/env bash
# OOD generalization sweep for e15 (the run that fixed the success-farming
# exploit).  21 conditions across pull force, hand stiffness, repeat-pull rate,
# grab location and drag distance -- see scripts/eval_ood.py for the axes.
#
# Headline metrics are yield ratio + recovery time; success rate and loop
# completion both read ~1.0 for a fully braced arm and cannot be trusted alone.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/eval_ood_e15.sbatch
#SBATCH --job-name=compliance_ood_e15
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
# node-086's GPU is dead at the driver level (`nvidia-smi` itself reports
# "Unable to determine the device handle for gpu 0000:01:00.0"), which surfaces
# as a bare `CUDA unknown error` / IndexError in select_gpus.
#SBATCH --exclude=bos14-node-086
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# NB: do NOT set CUDA_VISIBLE_DEVICES — SLURM scopes the job to its GPU.

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi -L || echo "nvidia-smi FAILED on $(hostname)"

uv run python -m mjlab.tasks.compliance.scripts.eval_ood \
  --task Mjlab-Compliance-Reach-Flexiv-Full \
  --checkpoint logs/rsl_rl/compliance_reach_flexiv/2026-07-23_00-58-19_e15 \
  --device cuda:0 \
  --num-envs 256 \
  "$@"
