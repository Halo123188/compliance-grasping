#!/usr/bin/env bash
# E9-aniso: push_softness = penalise anisotropy log(Kpar/Kperp) (cannot be gamed by isotropic softening); weight -0.2.
#
# Root cause found in e2/e3f/e4a1: the progress reward penalises every cm the
# hand drags the arm off-goal, so the arm resists the pull, never reaches the
# hand target, and the human never releases (recovery = n/a in all runs).  Fix:
# gate progress to zero while the human is pulling (mdp/rewards.py).  Now the arm
# has no reason to fight the pull -> yields to the hand target -> force drops ->
# human releases -> progress resumes -> arm returns to the goal.  A1 load stays
# on so the return-and-hold needs stiffness.
#
# Single variable vs e4a1: progress gated during the pull.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_e5_yield.sbatch
#SBATCH --job-name=compliance_e9a
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
  --agent.run-name e9_aniso \
  --env.rewards.force.weight -0.004 \
  --env.rewards.push_softness.weight -0.2 \
  --video False \
  "$@"
