#!/usr/bin/env bash
# Full training for a stiffness-supervised stage at one w_K, then the §3.4
# diagnostic eval (student + oracle) on the final checkpoint.
#   sbatch --job-name ct-sup-B-wk1.0 train_sup.sbatch B 1.0 0 6000
# args: STAGE  W_K  SEED  ITERS
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

STAGE="${1:-B}"
W_K="${2:-1.0}"
SEED="${3:-0}"
ITERS="${4:-6000}"
TASK="Mjlab-ComplianceTracking-Stage${STAGE}-Sup-Flexiv"
# run name encodes w_K so the sweep's runs never collide on disk
RUN="sup_${STAGE}_wk${W_K}_s${SEED}"

uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.rewards.stiffness-tracking.weight "${W_K}" \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False

LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_$(echo "${STAGE}" | tr 'A-Z' 'a-z')_sup/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR}  (w_K=${W_K}) ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --checkpoint "${LOGDIR}" --num-envs 256 --steps 800

uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --baseline --num-envs 256 --steps 800
