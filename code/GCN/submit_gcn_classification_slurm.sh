#!/usr/bin/env bash
set -euo pipefail

# Slurm submitter for GCN classification benchmark.

BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing prepared benchmark splits}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GCN/output_classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME:-GCN}"
CONFIG="${CONFIG:-${PROJECT_DIR}/dataset_config.csv}"
[[ "${CONFIG}" = /* ]] || CONFIG="${PWD}/${CONFIG}"
CONDA_ENV="${CONDA_ENV:-gcn_ab}"
RUN_ROOT="${OUT_ROOT}"

TASKS="classification"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"
STRICT_PERIMETER_SEEDS="${STRICT_PERIMETER_SEEDS:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
SELECT_METRIC="${SELECT_METRIC:-val_pr_auc}"

EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DROPOUT="${DROPOUT:-0.15}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-256}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"

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

cd "${PROJECT_DIR}"
activate_conda

PLAN_N=$(python train_gcn_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --dataset-config "${CONFIG}" \
  --tasks "${TASKS}" \
  --split-types "${SPLIT_TYPES}" \
  --seeds "${SEEDS}" \
  --strict-perimeter-seeds "${STRICT_PERIMETER_SEEDS}" \
  --print-run-plan | python -c 'import json,sys; print(len(json.load(sys.stdin)))')

LAST=$((PLAN_N - 1))
mkdir -p "${RUN_ROOT}/slurm_logs"

echo "TASKS=${TASKS}"
echo "SEEDS=${SEEDS}; STRICT_PERIMETER_SEEDS=${STRICT_PERIMETER_SEEDS}"
echo "Total classification runs: ${PLAN_N}; submitting Slurm array 0-${LAST}%${MAX_CONCURRENT}"

export PROJECT_DIR DATA_ROOT OUT_ROOT MODEL_NAME CONFIG CONDA_ENV TASKS SPLIT_TYPES SEEDS STRICT_PERIMETER_SEEDS EPOCHS PATIENCE BATCH_SIZE EVAL_BATCH_SIZE LR WEIGHT_DECAY DROPOUT HIDDEN_DIM NUM_LAYERS MLP_HIDDEN_DIM LOG_EVERY_BATCHES SELECT_METRIC
export CONDA_SH="${CONDA_SH:-}"
sbatch \
  --array=0-${LAST}%${MAX_CONCURRENT} \
  --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL \
  gcn_array_job.slurm

echo "Classification submitted.."
echo "After the jobs finish, run the corresponding postprocessing script with OUT_ROOT set."
