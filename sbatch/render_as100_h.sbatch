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
uv run python -m mjlab.tasks.compliance_tracking.scripts.render_torque \
  --task Mjlab-ComplianceTracking-StageA-Torque-Flexiv \
  --checkpoint "logs/rsl_rl/compliance_tracking_flexiv_a_torque/2026-07-25_05-23-10_torque_A_as100_ks100_tw1.0_s0" \
  --out compliance_torque_as100_horizontal.mp4 \
  --steps 420 --fps 40 --device cuda:0 \
  --displacement 0.28 --hold-s 2.6 --stiffness 800 \
  --push-dir 0.0 1.0 0.0
echo "=== RENDER COMPLETE ==="; ls -la compliance_torque_as100_horizontal.mp4
