#!/usr/bin/env bash
# The REAL softness lever: admittance_stiffness. The dominant position reward
# tracks the admittance target x_t, which yields F/admittance_stiffness along the
# push (isotropic in the admittance; default 300 -> ~13 cm -> k_par ~380). k_soft
# only touched the secondary torque target and was overridden. Lower
# admittance_stiffness -> x_t yields more -> the arm goes softer along the pull.
# k_soft is matched so the torque target agrees.
#   sbatch --job-name ct-as100 train_torque_soft.sbatch 100 100 1.0 0 5000
# args: ADMITTANCE_STIFFNESS  K_SOFT  TORQUE_W  SEED  ITERS
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh
AS="${1:-100}"; KS="${2:-100}"; TW="${3:-1.0}"; SEED="${4:-0}"; ITERS="${5:-5000}"
TASK="Mjlab-ComplianceTracking-StageA-Torque-Flexiv"
RUN="torque_A_as${AS}_ks${KS}_tw${TW}_s${SEED}"
uv run train "${TASK}" \
  --env.scene.num-envs 1024 \
  --env.commands.teacher.admittance-stiffness "${AS}" \
  --env.commands.teacher.k-soft "${KS}" \
  --env.rewards.torque-tracking.weight "${TW}" \
  --agent.seed "${SEED}" --agent.num-steps-per-env 32 \
  --agent.max-iterations "${ITERS}" --agent.save-interval 250 \
  --agent.logger tensorboard --agent.run-name "${RUN}" --video False
echo "=== done: ${RUN} ==="; ls -td logs/rsl_rl/compliance_tracking_flexiv_a_torque/*"${RUN}"* | head -1
