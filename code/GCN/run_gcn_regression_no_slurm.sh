#!/usr/bin/env bash
set -euo pipefail

# GCN regression benchmark runner for servers without Slurm.

BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing prepared benchmark splits}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GCN/output_regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME:-GCN}"
CONFIG="${CONFIG:-${PROJECT_DIR}/dataset_config.csv}"
[[ "${CONFIG}" = /* ]] || CONFIG="${PWD}/${CONFIG}"
CONDA_ENV="${CONDA_ENV:-gcn_ab}"
RUN_ROOT="${OUT_ROOT}"

TASKS="regression"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"
STRICT_PERIMETER_SEEDS="${STRICT_PERIMETER_SEEDS:-0}"
GPUS="${GPUS:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-0}"

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
SELECT_METRIC="${SELECT_METRIC:-val_rmse}"

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
    echo "ERROR: conda initialization script not found. Set CONDA_ENV after activating manually, or install conda." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
}

cd "${PROJECT_DIR}"
activate_conda
mkdir -p "${RUN_ROOT}/local_logs"

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

echo "TASKS=${TASKS}"
echo "SPLIT_TYPES=${SPLIT_TYPES}"
echo "SEEDS=${SEEDS}; STRICT_PERIMETER_SEEDS=${STRICT_PERIMETER_SEEDS}"
echo "Total regression runs: ${PLAN_N}"

# Each slot is permanently assigned to one GPU; child failures prevent summarization.
source "${PROJECT_DIR}/gpu_slots.sh"
gpu_slots_init "${GPUS}" "${MAX_JOBS_PER_GPU}" "${PLAN_N}"

for ((IDX = 0; IDX < PLAN_N; IDX++)); do
  if ! gpu_slots_acquire; then break; fi
  GPU="${GPU_SLOT_DEVICE}"
  echo "Launching regression run index ${IDX}/${PLAN_N} on GPU ${GPU}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    python train_gcn_antibioticbench.py \
      --data-root "${DATA_ROOT}" \
      --out-root "${OUT_ROOT}" \
      --model-name "${MODEL_NAME}" \
      --dataset-config "${CONFIG}" \
      --tasks "${TASKS}" \
      --split-types "${SPLIT_TYPES}" \
      --seeds "${SEEDS}" \
      --strict-perimeter-seeds "${STRICT_PERIMETER_SEEDS}" \
      --single-run-index "${IDX}" \
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
      --select-metric "${SELECT_METRIC}" \
      --threshold-mode fixed_0p5 \
      --log-every-batches "${LOG_EVERY_BATCHES}" \
      > "${RUN_ROOT}/local_logs/gcn_regression_run_${IDX}.out" \
      2> "${RUN_ROOT}/local_logs/gcn_regression_run_${IDX}.err"
  ) &
  GPU_SLOT_PIDS[GPU_SLOT_INDEX]=$!
done

gpu_slots_finish
echo "All regression runs finished.."

if [[ "${RUN_POSTPROCESS}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT}" CONDA_ENV="${CONDA_ENV}" bash postprocess_gcn_regression.sh
else
  echo "Postprocess not run. Set OUT_ROOT when running the corresponding postprocessing script."
fi
