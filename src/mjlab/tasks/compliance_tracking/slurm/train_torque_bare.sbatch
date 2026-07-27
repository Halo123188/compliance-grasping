#!/usr/bin/env bash
# Arm-only (no gripper) direct-torque compliance policy. Smoothing penalty on by
# default: the bare torque arm whips hard at reset (~100x joint speed, per the old
# compliance task) without it.
#   sbatch --job-name ct-bare-sm2e3 train_torque_bare.sbatch 0.002 0 5000
# args: SMOOTH_WEIGHT  SEED  ITERS
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh
SW="${1:-0.002}"; SEED="${2:-0}"; ITERS="${3:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-TorqueBare-Flexiv"
RUN="bare_A_sm${SW}_s${SEED}"
uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.rewards.joint-smooth.weight "-${SW}" \
  --agent.seed "${SEED}" --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" --agent.save-interval 250 \
  --agent.logger tensorboard --agent.run-name "${RUN}" --video False
echo "=== done: ${RUN} ==="; ls -td logs/rsl_rl/compliance_tracking_flexiv_a_torquebare/*"${RUN}"* | head -1
