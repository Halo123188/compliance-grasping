#!/usr/bin/env bash
# Direct joint-torque compliance (pure torque) on the e15 "full" disturbance.
# Rewards vs impedance: dropped stiffness/push_softness/push_anisotropy (they
# read a commanded stiffness that torque control has no analogue for); force
# penalty is now the sole compliance driver (weight doubled to -0.008, since
# yielding => small interaction force); added torque_effort (-0.002, mean
# (tau/tau_lim)^2) as the bang-bang regulariser.  Pure: the policy holds
# against gravity itself.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_torque_pure.sbatch
#SBATCH --job-name=compliance_torque_pure
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
#SBATCH --exclude=bos14-node-086
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi -L || echo "nvidia-smi FAILED on $(hostname)"

uv run train Mjlab-Compliance-Reach-Flexiv-TorquePure \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 3500 \
  --agent.save-interval 200 \
  --agent.logger tensorboard \
  --agent.run-name torque_pure \
  --video False \
  "$@"
