#!/usr/bin/env bash
# Tame the reset "whip to the goal" structurally: delta_frac caps the per-step
# torque change at delta_frac*effort_limit, so the arm cannot whip -- no reward
# penalty to tune (the joint_acc_l2 penalty was mis-scaled and diverged). On the
# soft as100 config.
#   sbatch train_torque_deltafrac.sbatch 0.1 0 5000
# args: DELTA_FRAC  SEED  ITERS
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh
DF="${1:-0.1}"; SEED="${2:-0}"; ITERS="${3:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-Torque-Flexiv"
RUN="torque_A_as100_df${DF}_s${SEED}"
uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.commands.teacher.admittance-stiffness 100 \
  --env.commands.teacher.k-soft 100 \
  --env.actions.arm-torque.delta-frac "${DF}" \
  --agent.seed "${SEED}" --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" --agent.save-interval 250 \
  --agent.logger tensorboard --agent.run-name "${RUN}" --video False
echo "=== done: ${RUN} ==="; ls -td logs/rsl_rl/compliance_tracking_flexiv_a_torque/*"${RUN}"* | head -1
