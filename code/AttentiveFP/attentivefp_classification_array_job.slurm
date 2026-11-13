#!/usr/bin/env bash
#SBATCH --job-name=afp_cls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

activate_conda_env() {
  local env_name="$1"
  local conda_sh="${CONDA_SH:-}"
  if [[ -z "${conda_sh}" ]]; then
    for candidate in \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
      if [[ -f "${candidate}" ]]; then
        conda_sh="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
    echo "ERROR: conda.sh not found. Set CONDA_SH to your conda initialization script." >&2
    exit 1
  fi
  source "${conda_sh}"
  conda activate "${env_name}"
}

: "${PROJECT_DIR:?Set PROJECT_DIR to the AttentiveFP code directory or use the submit script}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared benchmark split directory}"
: "${OUT_ROOT:?Set OUT_ROOT to the output directory}"
cd "${PROJECT_DIR}"
mkdir -p "${OUT_ROOT}/slurm_logs"
exec > >(tee -a "${OUT_ROOT}/slurm_logs/attentivefp_cls_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") \
     2> >(tee -a "${OUT_ROOT}/slurm_logs/attentivefp_cls_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)

cancel_array_on_error() {
  local exit_code=$?
  if [[ "${CANCEL_ARRAY_ON_FAILURE:-1}" == "1" && -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    echo "ERROR: task failed; cancelling array ${SLURM_ARRAY_JOB_ID}" >&2
    scancel "${SLURM_ARRAY_JOB_ID}" || true
  fi
  exit "${exit_code}"
}
trap cancel_array_on_error ERR

activate_conda_env "${CONDA_ENV}"
python check_attentivefp_environment.py

python train_attentivefp_classification_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --dataset-config "${CONFIG}" \
  --tasks classification \
  --split-types "${SPLIT_TYPES}" \
  --seeds "${SEEDS}" \
  --single-run-index "${SLURM_ARRAY_TASK_ID}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip "${GRAD_CLIP}" \
  --lr-reduce-factor "${LR_REDUCE_FACTOR}" \
  --lr-patience "${LR_PATIENCE}" \
  --dropout "${DROPOUT}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --num-timesteps "${NUM_TIMESTEPS}" \
  --select-metric val_pr_auc \
  --threshold-mode fixed_0p5 \
  --num-workers "${NUM_WORKERS}" \
  --torch-num-threads "${TORCH_NUM_THREADS}" \
  --log-every-batches "${LOG_EVERY_BATCHES}"
