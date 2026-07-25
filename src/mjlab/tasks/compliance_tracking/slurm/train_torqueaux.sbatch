#!/usr/bin/env bash
# STEP B: proprioception-only direct-torque policy + auxiliary force-estimation
# loss. The actor learns to infer F_ext from its own joint-torque history via a
# train-time head (discarded at deploy). Success = LOW and RELIABLE (low-variance
# across seeds) effective_k_pull -- the deployable version of what TorqueDiag got
# with the privileged pull direction fed in.
#   sbatch --job-name ct-taux-c0.5-s0 train_torqueaux.sbatch 0.5 1.0 0 5000
# args: AUX_COEF  TORQUE_W  SEED  ITERS
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

AUX="${1:-0.5}"
TW="${2:-1.0}"
SEED="${3:-0}"
ITERS="${4:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-TorqueAux-Flexiv"
RUN="torqueaux_A_aux${AUX}_tw${TW}_s${SEED}"

uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.rewards.torque-tracking.weight "${TW}" \
  --agent.algorithm.aux-coef "${AUX}" \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False

LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_a_torqueaux/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR} (aux=${AUX} tw=${TW}) ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage A --task "${TASK}" --checkpoint "${LOGDIR}" --num-envs 256 --steps 800
