#!/usr/bin/env bash
# EXP-1 diagnostic: anisotropic stiffness supervision + privileged pull direction
# in the actor obs. Does directional compliance appear once the direction is
# observable? K∥/K⊥ < 1 in the eval => observability was the binding wall.
#   sbatch --job-name ct-supdiag-A train_supdiag.sbatch 1.0 0.10 0 5000
# args: W_K  SIGMA_PULL  SEED  ITERS
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh

W_K="${1:-1.0}"
SP="${2:-0.10}"
SEED="${3:-0}"
ITERS="${4:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-SupDiag-Flexiv"
RUN="supdiag_A_wk${W_K}_sp${SP}_s${SEED}"

uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.rewards.stiffness-tracking.weight "${W_K}" \
  --env.rewards.position-tracking.params.sigma-pull "${SP}" \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False

LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_a_supdiag/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR}  (w_K=${W_K} sigma_pull=${SP}, privileged pull dir in actor) ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage A --checkpoint "${LOGDIR}" --num-envs 256 --steps 800
