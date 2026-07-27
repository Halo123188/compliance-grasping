#!/usr/bin/env bash
# Generalization: apply the sustained push at link7 (wrist, in-distribution),
# link5 (forearm) and link4 (elbow) -- all OOD except link7 -- and measure whether
# the end-effector still yields compliantly (k_par along the push) to a push
# applied further up the arm. Trained only on wrist pushes.
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh
TQ=Mjlab-ComplianceTracking-StageA-Torque-Flexiv
run() { echo; echo "########## $2  push@$3"; uv run python _stiffness_probe2.py "$TQ" "$1" 0.40 "$3"; }
echo "===== as100 policy (k_par~190 at wrist) ====="
for B in link7 link5 link4; do run "logs/rsl_rl/compliance_tracking_flexiv_a_torque/2026-07-25_05-23-10_torque_A_as100_ks100_tw1.0_s0" "as100" "$B"; done
echo "===== baseline policy (k_par~380 at wrist) ====="
for B in link7 link5 link4; do run "logs/rsl_rl/compliance_tracking_flexiv_a_torque/2026-07-24_01-36-16_torque_A_tw1.0_s0" "baseline" "$B"; done
echo; echo "########## GENERALIZE PROBE COMPLETE"
