#!/usr/bin/env bash
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
B=logs/rsl_rl/compliance_tracking_flexiv_a_torque
run() { local D; D=$(ls -td $B/*"$1"* 2>/dev/null | head -1); echo; echo "########## $1"; [ -n "$D" ] && uv run python _stiffness_probe2.py "$TQ" "$D" 0.60 || echo "  no dir"; }
run "torque_A_as150_ks150_tw1.0_s0"
run "torque_A_as100_ks100_tw1.0_s0"
run "torque_A_as60_ks60_tw1.0_s0"
echo; echo "########## reference admittance=300 k_soft=200 (baseline)"
uv run python _stiffness_probe2.py "$TQ" "$B/2026-07-24_01-36-16_torque_A_tw1.0_s0" 0.60
echo; echo "########## AS PROBE COMPLETE"
