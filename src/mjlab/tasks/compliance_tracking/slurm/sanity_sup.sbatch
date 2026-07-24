#!/usr/bin/env bash
# Smoke test for the stiffness-supervised variants: does the env with the extra
# teacher target-impedance signal + stiffness_tracking reward + widened freeze
# band actually construct and train?  30 iterations only.  Usage:
#   sbatch --job-name ct-sup-a sanity_sup.sbatch A
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh

STAGE="${1:-A}"
shift || true
TASK="Mjlab-ComplianceTracking-Stage${STAGE}-Sup-Flexiv"

echo "=== 30 training iterations, ${TASK} ==="
uv run train "${TASK}" \
  --env.scene.num-envs 512 \
  --agent.max-iterations 30 \
  --agent.save-interval 30 \
  --agent.logger tensorboard \
  --agent.run-name "sanity_sup_${STAGE}" \
  --video False \
  "$@"

echo "=== DONE ${TASK} ==="
