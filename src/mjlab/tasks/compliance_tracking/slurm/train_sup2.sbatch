#!/usr/bin/env bash
# Round-2 stiffness supervision: anisotropic K_target (soft along the pull,
# stiff perpendicular), baseline freeze band, sweeping w_K and the pull-time
# position sigma. Then the §3.4 diagnostic eval (student + oracle).
#   sbatch --job-name ct-sup2-A-wk1.0-sp0.10 train_sup2.sbatch A 1.0 0.10 0 5000
# args: STAGE  W_K  SIGMA_PULL  SEED  ITERS
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

STAGE="${1:-A}"
W_K="${2:-1.0}"
SP="${3:-0.10}"
SEED="${4:-0}"
ITERS="${5:-5000}"
TASK="Mjlab-ComplianceTracking-Stage${STAGE}-Sup-Flexiv"
RUN="sup2_${STAGE}_wk${W_K}_sp${SP}_s${SEED}"

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

LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_$(echo "${STAGE}" | tr 'A-Z' 'a-z')_sup/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR}  (w_K=${W_K} sigma_pull=${SP}) ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --checkpoint "${LOGDIR}" --num-envs 256 --steps 800

uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --baseline --num-envs 256 --steps 800
