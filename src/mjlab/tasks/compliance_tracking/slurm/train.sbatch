#!/usr/bin/env bash
# Full training run for one compliance-tracking stage + seed, followed by the
# §3.4 diagnostic eval on the final checkpoint.
#   sbatch --job-name ct-A-s0 train.sbatch A 0 5000
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
SEED="${2:-0}"
ITERS="${3:-5000}"
RUN="stage${STAGE}_s${SEED}"

uv run train "Mjlab-ComplianceTracking-Stage${STAGE}-Flexiv" \
  --env.scene.num-envs 1024 \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False

# Evaluate the final checkpoint on the fixed diagnostic batch, and the analytic
# oracle alongside it so the student is always read against its own ceiling.
LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_$(echo "${STAGE}" | tr 'A-Z' 'a-z')/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR} ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --checkpoint "${LOGDIR}" --num-envs 256 --steps 800

uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --baseline --num-envs 256 --steps 800
