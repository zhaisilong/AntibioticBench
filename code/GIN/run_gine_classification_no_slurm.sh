#!/usr/bin/env bash
set -euo pipefail

# 无 Slurm 版本：用 tmux/nohup 运行。
# 例：tmux new -s gine_cls
#     bash run_gine_classification_no_slurm.sh

BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing prepared benchmark splits}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GIN/output_classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME:-GINE}"
CONFIG="${CONFIG:-${PROJECT_DIR}/dataset_config.csv}"
[[ "${CONFIG}" = /* ]] || CONFIG="${PWD}/${CONFIG}"
CONDA_ENV="${CONDA_ENV:-chemprop-repro}"
RUN_ROOT="${OUT_ROOT}"

TASKS="classification"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"
GPUS="${GPUS:-0}"                         # 多卡可写 GPUS="0,1,2,3"；如需避开 6，就不要写 6
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

EPOCHS="300"
PATIENCE="40"
BATCH_SIZE="128"
EVAL_BATCH_SIZE="512"
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
mkdir -p "${RUN_ROOT}/local_logs"

PLAN_N=$(python train_gine_classification_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --dataset-config "${CONFIG}" \
  --tasks "${TASKS}" \
  --split-types "${SPLIT_TYPES}" \
  --seeds "${SEEDS}" \
  --print-run-plan | python -c 'import json,sys; print(len(json.load(sys.stdin)))')

echo "Total classification runs: ${PLAN_N}"
# Each slot is permanently assigned to one GPU; child failures prevent summarization.
source "${PROJECT_DIR}/gpu_slots.sh"
gpu_slots_init "${GPUS}" "${MAX_JOBS_PER_GPU}" "${PLAN_N}"

for ((IDX = 0; IDX < PLAN_N; IDX++)); do
  if ! gpu_slots_acquire; then break; fi
  GPU="${GPU_SLOT_DEVICE}"
  echo "Launching classification run index ${IDX}/${PLAN_N} on GPU ${GPU}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    python train_gine_classification_antibioticbench.py \
      --data-root "${DATA_ROOT}" \
      --out-root "${OUT_ROOT}" \
      --model-name "${MODEL_NAME}" \
      --dataset-config "${CONFIG}" \
      --tasks classification \
      --split-types "${SPLIT_TYPES}" \
      --seeds "${SEEDS}" \
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
      --pooling "${POOLING}" \
      --select-metric val_pr_auc \
      --use-pos-weight \
      --threshold-mode fixed_0p5 \
      --log-every-batches "${LOG_EVERY_BATCHES}" \
      > "${RUN_ROOT}/local_logs/gine_cls_run_${IDX}.out" 2> "${RUN_ROOT}/local_logs/gine_cls_run_${IDX}.err"
  ) &
  GPU_SLOT_PIDS[GPU_SLOT_INDEX]=$!
done

gpu_slots_finish
echo "All classification runs finished. Summarizing and running AC pair analysis..."
bash postprocess_gine_classification_outputs.sh
