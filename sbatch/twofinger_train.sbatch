#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=tf-train
#SBATCH --exclude=bos14-node-086
#SBATCH --output=sbatch/logs/%x-%j.log
set -euo pipefail
export MUJOCO_GL=egl
cd /home/yiboc/compliance-grasping

TASK="${1:-Mjlab-Grasp-TwoFinger-Flexiv}"
NENV="${2:-4096}"
ITERS="${3:-3000}"
RUN="${4:-$SLURM_JOB_ID}"
echo "[train] task=$TASK envs=$NENV iters=$ITERS run=$RUN node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# /home is a shared partition that is currently 100% full. On the COMPUTE nodes
# the /export/work NFS share (1+ TB free) is mounted at /work, so write all
# training output there. It appears at /export/work/yiboc/... on the login node.
LOG_ROOT="${LOG_ROOT:-/work/yiboc/mjlab_logs/rsl_rl}"
mkdir -p "$LOG_ROOT"
echo "[train] log_root=$LOG_ROOT"

uv run python src/mjlab/scripts/train.py "$TASK" \
  --env.scene.num-envs "$NENV" \
  --agent.max-iterations "$ITERS" \
  --agent.logger tensorboard \
  --log-root "$LOG_ROOT" \
  --video False \
  --agent.run-name "$RUN" \
  "${@:5}"
echo "[train] done"
