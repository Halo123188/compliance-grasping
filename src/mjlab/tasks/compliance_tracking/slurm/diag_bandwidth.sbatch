#!/usr/bin/env bash
# Diagnostic for the K_parallel > K_perp result (see README "Known ceilings").
#
# Hypothesis: the policy commands MORE stiffness along the pull because yielding
# is a *fast* motion (admittance omega_n ~ 12 rad/s vs ~0.14 m/s nominal path
# speed), and in this action space tracking bandwidth IS stiffness.
#
# Test: slow the yield transient without changing its magnitude. Raising the
# admittance mass lowers omega_n = sqrt(k/m) while leaving the steady-state
# yield F/k untouched, so the target goes to the same place, just gently.
#   m = 2 -> 8 kg  halves omega_n (12.2 -> 6.1 rad/s); damping d = 2*zeta*sqrt(k*m)
#   is derived from m in the teacher, so zeta stays 0.9.
#
# Prediction: if the bandwidth story is right, K_anisotropy_pull falls toward 1.
# If it stays ~1.5, the story is wrong and the cause is elsewhere.
#
#   sbatch --job-name ct-diag-m8 diag_bandwidth.sbatch 8.0 0
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

MASS="${1:-8.0}"
SEED="${2:-0}"
RUN="diag_m${MASS}_s${SEED}"

uv run train Mjlab-ComplianceTracking-StageA-Flexiv \
  --env.commands.teacher.admittance-mass "${MASS}" \
  --env.scene.num-envs 1024 \
  --agent.seed "${SEED}" \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 5000 \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.run-name "${RUN}" \
  --video False

LOGDIR=$(ls -td logs/rsl_rl/compliance_tracking_flexiv_a/*"${RUN}"* | head -1)
echo "=== eval: ${LOGDIR} (admittance mass ${MASS}) ==="
uv run python -m mjlab.tasks.compliance_tracking.scripts.eval \
  --stage A --checkpoint "${LOGDIR}" --num-envs 256 --steps 800
