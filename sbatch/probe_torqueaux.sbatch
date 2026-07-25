#!/usr/bin/env bash
# Steady-state stiffness probe over the TorqueAux sweep + the Torque (no sensing)
# and TorqueDiag (privileged direction) baselines. One sustained generic push,
# measured only near force saturation -> honest k_par along the pull. Lower=softer.
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh

AUX=Mjlab-ComplianceTracking-StageA-TorqueAux-Flexiv
TQ=Mjlab-ComplianceTracking-StageA-Torque-Flexiv
TD=Mjlab-ComplianceTracking-StageA-TorqueDiag-Flexiv
B=logs/rsl_rl

run() { echo; echo "########## $1"; uv run python _stiffness_probe.py "$2" "$3"; }

run "TorqueAux aux0.5 s0" "$AUX" "$B/compliance_tracking_flexiv_a_torqueaux/2026-07-24_20-31-40_torqueaux_A_aux0.5_tw1.0_s0"
run "TorqueAux aux0.5 s1" "$AUX" "$B/compliance_tracking_flexiv_a_torqueaux/2026-07-25_00-32-10_torqueaux_A_aux0.5_tw1.0_s1"
run "TorqueAux aux2.0 s0" "$AUX" "$B/compliance_tracking_flexiv_a_torqueaux/2026-07-24_20-31-18_torqueaux_A_aux2.0_tw1.0_s0"
run "TorqueAux aux2.0 s1" "$AUX" "$B/compliance_tracking_flexiv_a_torqueaux/2026-07-24_20-31-40_torqueaux_A_aux2.0_tw1.0_s1"
run "Torque baseline tw1.0 s0 (no sensing)" "$TQ" "$B/compliance_tracking_flexiv_a_torque/2026-07-24_01-36-16_torque_A_tw1.0_s0"
run "TorqueDiag tw1.0 s0 (privileged dir)"  "$TD" "$B/compliance_tracking_flexiv_a_torquediag/2026-07-24_01-35-25_torquediag_A_tw1.0_s0"
echo; echo "########## PROBE COMPLETE"
