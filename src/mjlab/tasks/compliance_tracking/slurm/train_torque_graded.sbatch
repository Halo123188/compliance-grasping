#!/usr/bin/env bash
# Graded multi-location compliance: push anywhere on the arm, softer toward the
# wrist. Smoothing on to tame the reset whip.
#   sbatch train_torque_graded.sbatch 0.002 0 5000
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
TASK="Mjlab-ComplianceTracking-StageA-TorqueGraded-Flexiv"
RUN="graded_A_sm${SW}_s${SEED}"
uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.rewards.joint-smooth.weight "-${SW}" \
  --agent.seed "${SEED}" --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" --agent.save-interval 250 \
  --agent.logger tensorboard --agent.run-name "${RUN}" --video False
echo "=== done: ${RUN} ==="; ls -td logs/rsl_rl/compliance_tracking_flexiv_a_torquegraded/*"${RUN}"* | head -1
