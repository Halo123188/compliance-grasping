#!/usr/bin/env bash
# Two-direction probe of the k_soft sweep, with a large anchor displacement
# (0.50 m) so the saturated push stays on across a big yield -> low k_par is
# measurable. Reports k_par (along pull) + yield; lower/bigger == softer.
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
run() {
  local D; D=$(ls -td $B/*"$1"* 2>/dev/null | head -1)
  echo; echo "########## $1  ($D)"
  [ -n "$D" ] && uv run python _stiffness_probe2.py "$TQ" "$D" 0.50 || echo "  no run dir"
}
run "torque_A_ks100_tw1.0_s0"
run "torque_A_ks50_tw1.0_s0"
run "torque_A_ks25_tw1.0_s0"
run "torque_A_ks50_tw3.0_s0"
run "torque_A_ks25_tw3.0_s0"
# reference: the k_soft=200 default (probe1 checkpoint), same big-displacement probe
echo; echo "########## reference k_soft=200 (tw1.0)"
uv run python _stiffness_probe2.py "$TQ" "$B/2026-07-24_01-36-16_torque_A_tw1.0_s0" 0.50
echo; echo "########## KSOFT PROBE COMPLETE"
