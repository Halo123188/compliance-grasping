#!/usr/bin/env bash
# E4-A1: does an unpredictable orthogonal environment load make directional
# compliance (K∥/K⊥ < 1) the optimum?
#
# In pure free space, reaching the goal needs no stiffness (x_ref alone gets you
# there), so isotropic softness dominates and K∥/K⊥→1 is reward-optimal (see the
# e2/e3f runs: K∥/K⊥≈1.04-1.11).  A1 adds a small unpredictable force
# (env_load_std=4 N/axis, set in reach_env_cfg) the arm must reject with
# stiffness to hold the goal.  Now: soft along the hand push (force penalty) yet
# stiff elsewhere (to reject the load) — only arbitration satisfies both.
#
# Single variable vs the e3f run: A1 added, force weight kept at -0.004.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_e4_a1.sbatch
#SBATCH --job-name=compliance_e4a1
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
# NB: do NOT set CUDA_VISIBLE_DEVICES — SLURM already scopes the job to its GPU.

uv run train Mjlab-Compliance-Reach-Flexiv \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 2500 \
  --agent.save-interval 200 \
  --agent.logger tensorboard \
  --agent.run-name e4_a1_f0.004 \
  --env.rewards.force.weight -0.004 \
  --video False \
  "$@"
