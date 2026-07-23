#!/usr/bin/env bash
# E15: close the success-farming exploit that degenerated e11-e14.
#
# Diagnosis (e14 @ iter 1800, eval level 3): yield 0.24, peak |F| 45N,
# recovery-time median 0.00s, K_par/K_perp 1.000 -- the arm braced at the goal
# and never moved.  Per-episode reward accounting showed why:
#
#     success  +328   <- +1 EVERY step, ungated
#     force     -71
#     softness  -32
#     progress   +1.3 <- the ONLY term paying for the return trip
#
# Three individually-reasonable settings closed a loop: (a) success_bonus is
# per-step and ungated, (b) the success TERMINATION is gated on ~is_pushing so a
# braced arm never ends its episode, (c) the human only releases once the arm
# yields.  Net: brace at the goal -> never released -> farm +1/step for ~600
# steps, netting +225 over the cost of bracing.  e10 (5s episodes, shorter
# pulls) had little to farm and scored success +7.4; e11+ (7s, d up to 0.30m)
# multiplied the payoff ~44x, which is what actually caused the regression --
# not the f_max/K_h range, which is what e14 tried to fix.
#
# Change (single, minimal, so attribution stays clean):
#   success_bonus is now gated on ~is_pushing, matching `progress` and
#   `reached_goal`.  Holding the goal under load earns nothing; the only route
#   back to earning is yield -> be released -> return.
#
# Everything else is identical to e14 (harder-only randomisation, shape modes,
# grip torque + learned rotational stiffness, randomised grab body / release
# rule).  Watch yield ratio and recovery time -- success rate and loop
# completion both read ~1.0 / 0.59 for the braced e14 policy and are not
# trustworthy on their own.
#
# Submit:  sbatch src/mjlab/tasks/compliance/slurm/train_e15.sbatch
#SBATCH --job-name=compliance_e15
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

# Node diagnostics: two e15 submissions died with `IndexError` in select_gpus
# because torch.cuda.device_count() returned 0 (NVML/CUDA failed to init on the
# allocated node).  Record which node so a persistently bad one can be excluded.
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  gpus: ${SLURM_JOB_GPUS:-unset} ==="
nvidia-smi -L || echo "nvidia-smi FAILED on $(hostname)"

uv run train Mjlab-Compliance-Reach-Flexiv-Full \
  --env.scene.num-envs 1024 \
  --agent.num-steps-per-env 32 \
  --agent.max-iterations 3500 \
  --agent.save-interval 200 \
  --agent.logger tensorboard \
  --agent.run-name e15 \
  --video False \
  "$@"
