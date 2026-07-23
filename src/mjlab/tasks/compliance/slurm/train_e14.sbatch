#!/usr/bin/env bash
# E14: items 6+5 on the CORRECTED randomisation (harder-only: f_max 35-80,
#  (1) push_softness = absolute K_par (-0.08) + secondary push_anisotropy (-0.02)
#      -- ratio-only penalty was gameable ("stiff everywhere" costs 0).
#  (2) K_min 50 -> 15: let the arm truly slacken at large yields (the +-3cm x_ref
#      cap otherwise forces a structural force floor).
#  (3) asymmetric critic: privileged human state (F_ext, u, hand, K_h, phase) on
#      the critic only (obs 36 actor / 47 critic) -- free variance reduction.
#  (4) force deadband 5N -> 1.5N: stop giving small resistance a free pass.
# Weights are now in the cfg file, so no CLI reward overrides needed.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_e10.sbatch
#SBATCH --job-name=compliance_e14
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# NB: do NOT set CUDA_VISIBLE_DEVICES — SLURM scopes the job to its GPU.

uv run train Mjlab-Compliance-Reach-Flexiv-Full \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 3500 \
  --agent.save-interval 200 \
  --agent.logger tensorboard \
  --agent.run-name e14 \
  --video False \
  "$@"
