#!/usr/bin/env bash
# E3-force: does a much stronger force penalty produce directional compliance
# (K∥/K⊥ < 1, positive yield) once the push actually overlaps the reach?
#
# Builds on the timing/termination fix (push_time_range (0,0.3), success gated
# on ~is_pushing).  The just-finished baseline (force weight -0.001) hard-braces:
# K∥/K⊥≈1.10, yield≈-0.10, |F_ext| pinned at 40 N.  Here force weight is 4x
# (-0.004) so sustained max-force bracing is no longer affordable.  Curriculum
# maxes out by iter ~600, so 2500 iters is plenty (~20 min).
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_e3_force.sbatch
#SBATCH --job-name=compliance_e3f
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail

cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# NB: do NOT set CUDA_VISIBLE_DEVICES — SLURM already scopes the job to its
# allocated GPU; overriding it to 0 points at a GPU we may not own on a shared
# node and crashes with "CUDA unknown error" (killed the first e3f attempt).

uv run train Mjlab-Compliance-Reach-Flexiv \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 2500 \
  --agent.save-interval 200 \
  --agent.logger tensorboard \
  --agent.run-name e3_force0.004 \
  --env.rewards.force.weight -0.004 \
  --video False \
  "$@"
