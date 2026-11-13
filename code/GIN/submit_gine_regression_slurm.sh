#!/usr/bin/env bash
set -euo pipefail

# ========= 按服务器目录结构设置 =========
BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing prepared benchmark splits}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GIN/output_regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME:-GINE}"
CONFIG="${CONFIG:-${PROJECT_DIR}/dataset_config.csv}"
[[ "${CONFIG}" = /* ]] || CONFIG="${PWD}/${CONFIG}"
CONDA_ENV="${CONDA_ENV:-chemprop-repro}"   # 需包含 torch/rdkit/sklearn/scipy
RUN_ROOT="${OUT_ROOT}"

# ========= benchmark 设置 =========
TASKS="regression"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# ========= GINE 回归固定 benchmark 超参数（不要按 test set 修改） =========
EPOCHS="300"
PATIENCE="40"
BATCH_SIZE="64"
EVAL_BATCH_SIZE="256"
LR="5e-4"
WEIGHT_DECAY="1e-5"
DROPOUT="0.20"
HIDDEN_DIM="300"
NUM_LAYERS="5"
MLP_HIDDEN_DIM="300"
POOLING="mean"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-20}"

cd "${PROJECT_DIR}"
if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
fi
conda activate "${CONDA_ENV}"
mkdir -p "${RUN_ROOT}/slurm_logs"

PLAN_N=$(python train_gine_regression_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --dataset-config "${CONFIG}" \
  --tasks "${TASKS}" \
  --split-types "${SPLIT_TYPES}" \
  --seeds "${SEEDS}" \
  --print-run-plan | python -c 'import json,sys; print(len(json.load(sys.stdin)))')

LAST=$((PLAN_N - 1))
echo "WARNING: GINE results may overwrite same-path vanilla GIN run files."
echo "Total regression runs: ${PLAN_N}; submitting Slurm array 0-${LAST}%${MAX_CONCURRENT}"

SBATCH_EXTRA=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  SBATCH_EXTRA+=(--exclude="${EXCLUDE_NODES}")
fi

export PROJECT_DIR DATA_ROOT OUT_ROOT MODEL_NAME CONFIG CONDA_ENV TASKS SPLIT_TYPES SEEDS EPOCHS PATIENCE BATCH_SIZE EVAL_BATCH_SIZE LR WEIGHT_DECAY DROPOUT HIDDEN_DIM NUM_LAYERS MLP_HIDDEN_DIM POOLING LOG_EVERY_BATCHES
export CONDA_SH="${CONDA_SH:-}"
sbatch "${SBATCH_EXTRA[@]}" \
  --array=0-${LAST}%${MAX_CONCURRENT} \
  --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL \
  gine_regression_array_job.slurm

echo "Slurm submission completed."
echo "作业全部完成后运行：bash postprocess_gine_regression_outputs.sh"
