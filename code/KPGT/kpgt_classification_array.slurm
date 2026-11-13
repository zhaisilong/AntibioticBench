#!/bin/bash
#SBATCH --job-name=kpgt_cls
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:?Set SCRIPT_DIR or use the supplied submit script}"
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${PWD}/${SCRIPT_DIR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
[[ "${BENCHMARK_ROOT}" = /* ]] || BENCHMARK_ROOT="${PWD}/${BENCHMARK_ROOT}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared split-data directory}"
[[ "${DATA_ROOT}" = /* ]] || DATA_ROOT="${PWD}/${DATA_ROOT}"
KPGT_ROOT="${KPGT_ROOT:?Set KPGT_ROOT to the external KPGT source directory}"
[[ "${KPGT_ROOT}" = /* ]] || KPGT_ROOT="${PWD}/${KPGT_ROOT}"
if [[ -n "${KPGT_PRETRAINED_MODEL:-}" && "${KPGT_PRETRAINED_MODEL}" != /* ]]; then
  KPGT_PRETRAINED_MODEL="${PWD}/${KPGT_PRETRAINED_MODEL}"
  export KPGT_PRETRAINED_MODEL
fi
export KPGT_ROOT
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT}/outputs/KPGT/classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
DATASET_CONFIG="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_config.csv}"
[[ "${DATASET_CONFIG}" = /* ]] || DATASET_CONFIG="${PWD}/${DATASET_CONFIG}"
CONDA_ENV="${CONDA_ENV:-KPGT}"
RUN_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

# Fixed benchmark hyperparameters. They can be overridden at submission time.
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
PREPROCESS_JOBS="${PREPROCESS_JOBS:-${SLURM_CPUS_PER_TASK}}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
USE_POS_WEIGHT="${USE_POS_WEIGHT:-0}"
SELECT_METRIC="${SELECT_METRIC:-val_pr_auc}"

mkdir -p "${OUT_ROOT}/slurm_logs"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda initialization script was not found." >&2
  exit 2
fi
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "===== KPGT classification array task ====="
echo "date=$(date -Is)"
echo "run started"
echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID} array_task_id=${RUN_INDEX}"
echo "Python environment activated"
python --version
nvidia-smi || true

a=(
  python -u train_kpgt_antibioticbench.py
  --task classification
  --data-root "${DATA_ROOT}"
  --out-root "${OUT_ROOT}"
  --dataset-config "${DATASET_CONFIG}"
  --single-run-index "${RUN_INDEX}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --lr "${LR}"
  --dropout "${DROPOUT}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --eval-num-workers "${EVAL_NUM_WORKERS}"
  --preprocess-jobs "${PREPROCESS_JOBS}"
  --selection-metric "${SELECT_METRIC}"
  --threshold 0.5
  --resume "${RESUME}"
)
[[ "${FORCE}" == "1" ]] && a+=(--force)
[[ "${FORCE_PREPROCESS}" == "1" ]] && a+=(--force-preprocess)
[[ "${USE_POS_WEIGHT}" == "1" ]] && a+=(--use-pos-weight)
[[ -n "${KPGT_ROOT:-}" ]] && a+=(--kpgt-root "${KPGT_ROOT}")
[[ -n "${KPGT_PRETRAINED_MODEL:-}" ]] && a+=(--model-path "${KPGT_PRETRAINED_MODEL}")

"${a[@]}"
echo "===== classification task ${RUN_INDEX} completed at $(date -Is) ====="
