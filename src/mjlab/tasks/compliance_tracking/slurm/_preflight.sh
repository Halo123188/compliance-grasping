# Shared preflight for compliance-tracking SLURM jobs. Source, do not execute.
#
# Some nodes in this cluster come up with a broken driver state: NVML fails to
# initialize and torch sees zero GPUs, so `train` dies in select_gpus() before
# any task code runs. SLURM does not mark those nodes down, so a plain resubmit
# can land on the same one. Detect it here and requeue instead of burning the
# job -- and always log the hostname, so a failure that slips through is
# diagnosable after the fact (the earlier ones were not).

echo "[preflight] host=$(hostname) job=${SLURM_JOB_ID:-?} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "[preflight] FAIL: nvidia-smi unusable on $(hostname); requeueing job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 1
fi

if ! uv run python -c "import torch; assert torch.cuda.device_count() > 0" 2>/dev/null; then
  echo "[preflight] FAIL: torch sees no CUDA device on $(hostname); requeueing job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 1
fi

echo "[preflight] ok: $(nvidia-smi -L | head -1)"
