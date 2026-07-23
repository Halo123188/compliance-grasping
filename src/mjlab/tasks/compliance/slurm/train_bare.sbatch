#!/usr/bin/env bash
# Bare Rizon 4S (no UMI gripper) on the e15 setup.
#
# Identical training recipe to e15 (the run that fixed the success-farming
# exploit): the "full" env -- disturbance shape modes (hold/shake/drag), grip
# moment r x F + learned rotational stiffness, corrected harder-only
# randomisation, and the ~is_pushing-gated success bonus -- but the robot is the
# arm alone.  The end-effector is the link7 flange tool point instead of the
# grasp site between the UMI jaws, and the human can grab link5/6/7 (there is no
# gripper_base to grab).  Action dim (7) and observation dims (37 actor / 51
# critic) are unchanged, so the same policy architecture applies.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_bare.sbatch
#SBATCH --job-name=compliance_bare
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
# node-086's GPU is dead at the driver level (nvidia-smi itself fails there).
#SBATCH --exclude=bos14-node-086
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# NB: do NOT set CUDA_VISIBLE_DEVICES — SLURM scopes the job to its GPU.

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi -L || echo "nvidia-smi FAILED on $(hostname)"

uv run train Mjlab-Compliance-Reach-Flexiv-Bare-Full \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 3500 \
  --agent.save-interval 200 \
  --agent.logger tensorboard \
  --agent.run-name bare \
  --video False \
  "$@"
