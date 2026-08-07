#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=wide-distill
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# uv re-syncs the venv on every `uv run`, taking an flock on .venv/.lock. That
# lock lives on NFS, and two jobs starting on different nodes at the same time
# deadlock on it -- silently, with no output at all, for as long as the job
# runs. mjlab is installed as a path .pth pointing at src/, so the sync has
# nothing to do anyway: source edits are already live without a rebuild.
export UV_NO_SYNC=1
cd /home/yiboc/compliance-grasping

# DAgger: distil the state teacher into the camera-only student.
#
#   sbatch sbatch/wide_distill.sbatch <TASK> <TEACHER_CKPT> <NENV> <ITERS> <RUN>
#
# Fewer envs than a PPO run on purpose: every env renders an image every control
# step, and the rollout buffer holds num_envs x num_steps whole images.
TASK="${1:-Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth}"
TEACHER="${2:?teacher checkpoint required}"
NENV="${3:-1024}"
ITERS="${4:-3000}"
RUN="${5:-$SLURM_JOB_ID}"
echo "[distill] task=$TASK envs=$NENV iters=$ITERS run=$RUN node=$(hostname)"
echo "[distill] teacher=$TEACHER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

test -f "$TEACHER" || { echo "[distill] teacher checkpoint not found"; exit 1; }

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
