#!/usr/bin/env bash
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

# Resolve code from this launcher; user-supplied relative paths use the caller's directory.
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared benchmark split directory}"
if [[ ! -d "${DATA_ROOT}" ]]; then echo "ERROR: DATA_ROOT is not a directory." >&2; exit 1; fi
DATA_ROOT="$(cd -- "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${REPO_ROOT}}/outputs/AttentiveFP/regression}"
case "${OUT_ROOT}" in /*) ;; *) OUT_ROOT="${PWD}/${OUT_ROOT}" ;; esac
MODEL_NAME="${MODEL_NAME:-AttentiveFP}"
CONFIG="${CONFIG:-${PROJECT_DIR}/dataset_config.csv}"
case "${CONFIG}" in /*) ;; *) CONFIG="${PWD}/${CONFIG}" ;; esac
CONDA_ENV="${CONDA_ENV:-AttentiveFP}"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
CANCEL_ARRAY_ON_FAILURE="${CANCEL_ARRAY_ON_FAILURE:-1}"

EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
LR_REDUCE_FACTOR="${LR_REDUCE_FACTOR:-0.5}"
LR_PATIENCE="${LR_PATIENCE:-10}"
DROPOUT="${DROPOUT:-0.20}"
HIDDEN_DIM="${HIDDEN_DIM:-200}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-20}"

cd "${PROJECT_DIR}"
activate_conda_env "${CONDA_ENV}"
mkdir -p "${OUT_ROOT}/slurm_logs"
python check_attentivefp_environment.py

PLAN_N=$(python train_attentivefp_regression_antibioticbench.py \
  --data-root "${DATA_ROOT}" --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" --dataset-config "${CONFIG}" \
  --tasks regression --split-types "${SPLIT_TYPES}" --seeds "${SEEDS}" \
  --print-run-plan | python -c 'import json,sys; print(len(json.load(sys.stdin)))')
LAST=$((PLAN_N - 1))
ARRAY_SPEC="0-${LAST}%${MAX_CONCURRENT}"
if [[ -n "${RUN_RANGE:-}" ]]; then ARRAY_SPEC="${RUN_RANGE}%${MAX_CONCURRENT}"; fi

SBATCH_EXTRA=()
if [[ -n "${EXCLUDE_NODES}" ]]; then SBATCH_EXTRA+=(--exclude="${EXCLUDE_NODES}"); fi

export PROJECT_DIR DATA_ROOT OUT_ROOT MODEL_NAME CONFIG CONDA_ENV SPLIT_TYPES SEEDS
export EPOCHS PATIENCE BATCH_SIZE EVAL_BATCH_SIZE LR MIN_LR WEIGHT_DECAY GRAD_CLIP
export LR_REDUCE_FACTOR LR_PATIENCE DROPOUT HIDDEN_DIM NUM_LAYERS NUM_TIMESTEPS
export NUM_WORKERS TORCH_NUM_THREADS LOG_EVERY_BATCHES CANCEL_ARRAY_ON_FAILURE
export CONDA_SH="${CONDA_SH:-}"

JOB_ID=$(sbatch --parsable "${SBATCH_EXTRA[@]}" \
  --array="${ARRAY_SPEC}" \
  --output="${OUT_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${OUT_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL \
  attentivefp_regression_array_job.slurm)

echo "Full run plan: ${PLAN_N}; submitted array ${ARRAY_SPEC} as job ${JOB_ID}"
echo "Output directory configured."
echo "After completion: bash postprocess_attentivefp_regression_outputs.sh"
