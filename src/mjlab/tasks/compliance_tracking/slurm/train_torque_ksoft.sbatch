#!/usr/bin/env bash
# Push k_par (softness along the pull) as low as possible, ideally below the
# k_soft=200 default. k_soft sets the commanded along-pull stiffness in BOTH the
# torque target and the admittance x_t (perpendicular stays k_precision=900, so
# lower k_soft only softens the pull axis, no task cost). torque_weight closes the
# ~2x gap between commanded k_soft and the ~390 the student actually reached.
#   sbatch --job-name ct-ks50-tw3 train_torque_ksoft.sbatch 50 3.0 0 5000
# args: K_SOFT  TORQUE_W  SEED  ITERS
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh

KS="${1:-50}"
TW="${2:-1.0}"
SEED="${3:-0}"
ITERS="${4:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-Torque-Flexiv"
RUN="torque_A_ks${KS}_tw${TW}_s${SEED}"

uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.commands.teacher.k-soft "${KS}" \
  --env.rewards.torque-tracking.weight "${TW}" \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False
echo "=== done: ${RUN} ==="
ls -td logs/rsl_rl/compliance_tracking_flexiv_a_torque/*"${RUN}"* | head -1
