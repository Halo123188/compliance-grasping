#!/usr/bin/env bash
# v18 = the bug-fix batch on top of v17b's recipe. Changed vs v17b:
#   1. grip_force_base passed explicitly (8.0 N, was silently 22.5 N) and
#      _GRIP_FORCE_SCALE 8 -> 11 so the reachable force is [0, 30] N.
#   2. grasp force measured as the contact-NORMAL component, not net-force
#      magnitude (sensor reduce netforce -> maxforce, 8 slots/finger).
#   3. force_margin_penalty clamped to [0, 1].
#   4. lift_success now time_out=True (bootstraps) + one-shot bonus, weight 100.
#   5. crush_penalty genuinely one-shot, weight -5 -> -15 (same total price).
#   6. ee_lift / ee_anchor reference re-captured per episode.
#   7. play/render no longer teleports the cube every 5 s.
#   8. curriculum merged onto the base instead of replacing it.
# Everything else (reward set, weights, clip_actions=2.0, obs history, curricula)
# is identical to v17b. Usage: sbatch fragile_v18.sbatch <seed>
#SBATCH --job-name=fragile_v18
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --exclude=bos14-node-086
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/reflex-grasp-wt/slurm_log/%x-%j.log
set -euo pipefail
cd /home/yiboc/reflex-grasp-wt          # isolated worktree, NOT the main repo
mkdir -p slurm_log
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
SEED="${1:?seed}"
uv run train Mjlab-Lift-Fragile-Flexiv \
  --env.scene.num-envs 4096 \
  --agent.seed "$SEED" \
  --agent.run-name "fragile_v18_s${SEED}" \
  --agent.num-steps-per-env 24 \
  --agent.max-iterations 9000 \
  --agent.save-interval 100 \
  --agent.logger tensorboard \
  --video False
