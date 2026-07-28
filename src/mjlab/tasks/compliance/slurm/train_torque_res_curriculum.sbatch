#!/usr/bin/env bash
# Option-3 curriculum, PHASE B: resume the reach-and-hold policy from phase A and
# turn the disturbance on.  Pass the phase-A run dir name via LOAD_RUN env var.
#SBATCH --job-name=compliance_torque_res_curriculum
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/yiboc/compliance-grasping/slurm_log/%x-%j.log
#SBATCH --exclude=bos14-node-086
set -euo pipefail
cd /home/yiboc/compliance-grasping
mkdir -p slurm_log
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  resume from: ${LOAD_RUN} ==="; nvidia-smi -L || true
uv run train Mjlab-Compliance-Reach-Flexiv-TorqueResidual \
  --env.scene.num-envs 1024 --agent.num-steps-per-env 32 \
  --agent.max-iterations 6000 --agent.save-interval 200 \
  --agent.logger tensorboard --agent.run-name torque_res_curriculum \
  --agent.resume True --agent.load-run "${LOAD_RUN}" \
  --video False "$@"
