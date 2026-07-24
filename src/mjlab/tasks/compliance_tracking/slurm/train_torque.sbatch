#!/usr/bin/env bash
# EXP-2: direct joint-torque policy tracking the teacher's compliant torque
# target. Success = low effective_k_pull (arm goes soft along a generic push,
# which the diagonal-K action space structurally could not do).
#   sbatch --job-name ct-torque-A train_torque.sbatch Torque 1.0 0 5000
# args: VARIANT(Torque|TorqueDiag)  TORQUE_W  SEED  ITERS
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh

VARIANT="${1:-Torque}"
TW="${2:-1.0}"
SEED="${3:-0}"
ITERS="${4:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-${VARIANT}-Flexiv"
SUFFIX=$(echo "$VARIANT" | tr 'A-Z' 'a-z')
RUN="${SUFFIX}_A_tw${TW}_s${SEED}"

uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.rewards.torque-tracking.weight "${TW}" \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False

LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_a_${SUFFIX}/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR} (variant=${VARIANT} tw=${TW}) ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage A --checkpoint "${LOGDIR}" --num-envs 256 --steps 800
