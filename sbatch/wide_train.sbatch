#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=wide-train
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

# State-based PPO teacher on the WIDE claw + foam bench.
#
#   sbatch sbatch/wide_train.sbatch <TASK> <NENV> <ITERS> <RUN> [extra args]
#
# --mem is explicit and NOT optional: mujoco-warp JIT-compiles several kernels
# specialised on this scene, and NVRTC needs the headroom (see arm_cfg.py's note
# on the D435i capsule for what running short of it looks like).
TASK="${1:-Mjlab-Grasp-TwoFingerWide-Flexiv}"
NENV="${2:-4096}"
ITERS="${3:-4000}"
RUN="${4:-$SLURM_JOB_ID}"
echo "[train] task=$TASK envs=$NENV iters=$ITERS run=$RUN node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# /home is shared; compute nodes mount /export/work at /work.
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
