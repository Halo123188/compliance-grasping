#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
source src/mjlab/tasks/compliance_tracking/slurm/_preflight.sh
AUX=Mjlab-ComplianceTracking-StageA-TorqueAux-Flexiv
B=logs/rsl_rl/compliance_tracking_flexiv_a_torqueaux
# Softest aux run (probe1 k_par=350); has the force head -> the sensing panel shows.
uv run python -m mjlab.tasks.compliance_tracking.scripts.render_torque \
  --task "$AUX" \
  --checkpoint "$B/2026-07-24_20-31-40_torqueaux_A_aux0.5_tw1.0_s0" \
  --out compliance_torqueaux_aux0.5_s0.mp4 \
  --steps 420 --fps 40 --device cuda:0
echo "=== RENDER COMPLETE ==="; ls -la compliance_torqueaux_aux0.5_s0.mp4
