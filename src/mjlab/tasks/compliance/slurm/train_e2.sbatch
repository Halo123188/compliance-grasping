#!/usr/bin/env bash
# E2 main training: PPO + GRU + displacement curriculum on the Stage-1
# compliance reach-under-disturbance task (plan §8 E2).
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_e2.sbatch
#SBATCH --job-name=compliance_e2
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
export CUDA_VISIBLE_DEVICES=0

uv run train Mjlab-Compliance-Reach-Flexiv \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 5000 \
  --agent.save-interval 100 \
  --agent.logger tensorboard \
  --agent.run-name e2_ppo_gru \
  --video False \
  "$@"
