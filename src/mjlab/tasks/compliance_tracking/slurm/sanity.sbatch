#!/usr/bin/env bash
# Short end-to-end training sanity run: does the PPO+GRU loop actually turn on
# for this task, on GPU, with a real env count?  Not a training run — 30
# iterations only.  Usage:  sbatch --job-name ct-sanity-a sanity.sbatch A
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh

STAGE="${1:-A}"
shift || true

echo "=== smoke test, stage ${STAGE} ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.smoke_test \
  --stage "${STAGE}" --num-envs 64 --steps 800

echo "=== 30 training iterations, stage ${STAGE} ==="
uv run train "Mjlab-ComplianceTracking-Stage${STAGE}-Flexiv" \
  --env.scene.num-envs 512 \
  --agent.max-iterations 30 \
  --agent.save-interval 30 \
  --agent.logger tensorboard \
  --agent.run-name "sanity_${STAGE}" \
  --video False \
  "$@"

echo "=== baseline eval, stage ${STAGE} ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage "${STAGE}" --baseline --num-envs 128 --steps 800
