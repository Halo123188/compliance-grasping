#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=tf-distill
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
cd /home/yiboc/compliance-grasping

# DAgger: distil the state teacher into a camera-only student.
#
#   sbatch sbatch/twofinger_distill.sbatch <TASK> <TEACHER_CKPT> <NENV> <ITERS> <RUN>
#
# Defaults distil v11_align (run 46984, 80.5% deployment success) into the depth
# student. Fewer envs than a PPO run on purpose: every env renders an image every
# control step, and the rollout buffer holds num_envs x num_steps whole images.
TASK="${1:-Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth}"
TEACHER="${2:-/work/yiboc/mjlab_logs/rsl_rl/flexiv_two_finger_grasp/2026-07-31_19-50-25_v11_align/model_2999.pt}"
NENV="${3:-1024}"
ITERS="${4:-3000}"
RUN="${5:-$SLURM_JOB_ID}"
echo "[distill] task=$TASK envs=$NENV iters=$ITERS run=$RUN node=$(hostname)"
echo "[distill] teacher=$TEACHER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

test -f "$TEACHER" || { echo "[distill] teacher checkpoint not found"; exit 1; }

# See twofinger_train.sbatch: /home is full, compute nodes have /work.
LOG_ROOT="${LOG_ROOT:-/work/yiboc/mjlab_logs/rsl_rl}"
mkdir -p "$LOG_ROOT"
echo "[distill] log_root=$LOG_ROOT"

uv run python src/mjlab/scripts/train.py "$TASK" \
  --env.scene.num-envs "$NENV" \
  --agent.max-iterations "$ITERS" \
  --agent.logger tensorboard \
  --log-root "$LOG_ROOT" \
  --video False \
  --teacher-checkpoint "$TEACHER" \
  --agent.run-name "$RUN" \
  "${@:6}"
echo "[distill] done"
