#!/usr/bin/env bash
#SBATCH --job-name=compliance_torque_smooth_warmup
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
#SBATCH --exclude=bos14-node-086
set -euo pipefail
cd /home/yiboc/compliance-grasping; mkdir -p slurm_log
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "=== node: $(hostname) job: ${SLURM_JOB_ID} ==="; nvidia-smi -L || true
uv run train Mjlab-Compliance-Reach-Flexiv-TorqueSmooth-Warmup \
  --env.scene.num-envs 1024 --agent.num-steps-per-env 32 \
  --agent.max-iterations 2000 --agent.save-interval 200 \
  --agent.logger tensorboard --agent.run-name torque_smooth_warmup --video False "$@"
