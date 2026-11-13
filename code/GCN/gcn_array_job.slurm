#!/usr/bin/env bash
#SBATCH --job-name=gcn_ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR via the submit script}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared split directory}"
: "${OUT_ROOT:?Set OUT_ROOT to the output directory}"
cd "${PROJECT_DIR}"
RUN_ROOT="${OUT_ROOT}"
mkdir -p "${RUN_ROOT}/slurm_logs"

# Slurm itself writes stdout/stderr to PROJECT_DIR unless --output path is absolute.
# PROJECT_DIR is the code folder, while RUN_ROOT is the output folder.
exec > >(tee -a "${RUN_ROOT}/slurm_logs/gcn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") \
  2> >(tee -a "${RUN_ROOT}/slurm_logs/gcn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)

activate_conda() {
  if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda initialization script not found." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
}

activate_conda

python --version
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count', torch.cuda.device_count(), 'current_name', torch.cuda.get_device_name(0))
PY

STRICT_PERIMETER_SEEDS="${STRICT_PERIMETER_SEEDS:-0}"
SELECT_METRIC="${SELECT_METRIC:-val_pr_auc}"

EXTRA_ARGS=(--threshold-mode fixed_0p5 --select-metric "${SELECT_METRIC}")
if [[ ",${TASKS}," == *",classification,"* ]]; then
  EXTRA_ARGS+=(--use-pos-weight)
fi

python train_gcn_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --dataset-config "${CONFIG}" \
  --tasks "${TASKS}" \
  --split-types "${SPLIT_TYPES}" \
  --seeds "${SEEDS}" \
  --strict-perimeter-seeds "${STRICT_PERIMETER_SEEDS}" \
  --single-run-index "${SLURM_ARRAY_TASK_ID}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --dropout "${DROPOUT}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --mlp-hidden-dim "${MLP_HIDDEN_DIM}" \
  --log-every-batches "${LOG_EVERY_BATCHES}" \
  "${EXTRA_ARGS[@]}"
