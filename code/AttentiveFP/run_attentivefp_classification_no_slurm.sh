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
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${REPO_ROOT}}/outputs/AttentiveFP/classification}"
case "${OUT_ROOT}" in /*) ;; *) OUT_ROOT="${PWD}/${OUT_ROOT}" ;; esac
MODEL_NAME="${MODEL_NAME:-AttentiveFP}"
CONFIG="${CONFIG:-${PROJECT_DIR}/dataset_config.csv}"
case "${CONFIG}" in /*) ;; *) CONFIG="${PWD}/${CONFIG}" ;; esac
CONDA_ENV="${CONDA_ENV:-AttentiveFP}"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"
GPUS="${GPUS:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
POSTPROCESS="${POSTPROCESS:-1}"

EPOCHS="${EPOCHS:-300}"; PATIENCE="${PATIENCE:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"; EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
LR="${LR:-5e-4}"; MIN_LR="${MIN_LR:-1e-6}"; WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"; LR_REDUCE_FACTOR="${LR_REDUCE_FACTOR:-0.5}"; LR_PATIENCE="${LR_PATIENCE:-10}"
DROPOUT="${DROPOUT:-0.20}"; HIDDEN_DIM="${HIDDEN_DIM:-200}"; NUM_LAYERS="${NUM_LAYERS:-2}"; NUM_TIMESTEPS="${NUM_TIMESTEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"; TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"; LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-20}"

cd "${PROJECT_DIR}"; activate_conda_env "${CONDA_ENV}"; mkdir -p "${OUT_ROOT}/local_logs"
python check_attentivefp_environment.py
PLAN_N=$(python train_attentivefp_classification_antibioticbench.py --data-root "${DATA_ROOT}" --out-root "${OUT_ROOT}" --model-name "${MODEL_NAME}" --dataset-config "${CONFIG}" --tasks classification --split-types "${SPLIT_TYPES}" --seeds "${SEEDS}" --print-run-plan | python -c 'import json,sys; print(len(json.load(sys.stdin)))')
[[ "${PLAN_N}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: run plan is empty or invalid" >&2; exit 1; }
START_IDX=0; END_IDX=$((PLAN_N - 1))
if [[ -n "${RUN_RANGE:-}" ]]; then
  if [[ "${RUN_RANGE}" =~ ^([0-9]+)-([0-9]+)$ ]]; then START_IDX=$((10#${BASH_REMATCH[1]})); END_IDX=$((10#${BASH_REMATCH[2]}));
  elif [[ "${RUN_RANGE}" =~ ^[0-9]+$ ]]; then START_IDX=$((10#${RUN_RANGE})); END_IDX="${START_IDX}";
  else echo "ERROR: RUN_RANGE must be N or START-END" >&2; exit 1; fi
fi
if (( START_IDX > END_IDX || END_IDX >= PLAN_N )); then
  echo "ERROR: RUN_RANGE is outside the run plan" >&2; exit 1
fi
[[ "${MAX_JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: MAX_JOBS_PER_GPU must be a positive integer" >&2; exit 1;
}
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
(( NUM_GPUS > 0 )) || { echo "ERROR: GPUS must list at least one GPU" >&2; exit 1; }
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_ARR[$i]="${GPU_ARR[$i]//[[:space:]]/}"
  [[ -n "${GPU_ARR[$i]}" ]] || { echo "ERROR: GPUS contains an empty entry" >&2; exit 1; }
  for ((j=0; j<i; j++)); do
    [[ "${GPU_ARR[$i]}" != "${GPU_ARR[$j]}" ]] || { echo "ERROR: GPUS contains a duplicate GPU" >&2; exit 1; }
  done
done
MAX_TOTAL=$((NUM_GPUS * MAX_JOBS_PER_GPU))
# Each slot is permanently assigned to one GPU. Reap its previous child before reuse.
# This preserves per-GPU limits even when jobs finish out of order.
slot_pids=()
failed=0
for ((IDX=START_IDX; IDX<=END_IDX; IDX++)); do
  slot=$(((IDX - START_IDX) % MAX_TOTAL))
  if [[ -n "${slot_pids[$slot]:-}" ]]; then
    if ! wait "${slot_pids[$slot]}"; then failed=1; fi
    unset "slot_pids[$slot]"
    if (( failed )); then break; fi
  fi
  GPU="${GPU_ARR[$((slot % NUM_GPUS))]}"
  echo "Launching AttentiveFP classification index ${IDX}/${PLAN_N} on GPU ${GPU}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    python train_attentivefp_classification_antibioticbench.py \
      --data-root "${DATA_ROOT}" --out-root "${OUT_ROOT}" --model-name "${MODEL_NAME}" --dataset-config "${CONFIG}" \
      --tasks classification --split-types "${SPLIT_TYPES}" --seeds "${SEEDS}" --single-run-index "${IDX}" \
      --epochs "${EPOCHS}" --patience "${PATIENCE}" --batch-size "${BATCH_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay "${WEIGHT_DECAY}" --grad-clip "${GRAD_CLIP}" \
      --lr-reduce-factor "${LR_REDUCE_FACTOR}" --lr-patience "${LR_PATIENCE}" --dropout "${DROPOUT}" \
      --hidden-dim "${HIDDEN_DIM}" --num-layers "${NUM_LAYERS}" --num-timesteps "${NUM_TIMESTEPS}" \
      --select-metric val_pr_auc \
          --threshold-mode fixed_0p5 \
      --num-workers "${NUM_WORKERS}" --torch-num-threads "${TORCH_NUM_THREADS}" --log-every-batches "${LOG_EVERY_BATCHES}" \
      > "${OUT_ROOT}/local_logs/attentivefp_cls_run_${IDX}.out" 2> "${OUT_ROOT}/local_logs/attentivefp_cls_run_${IDX}.err"
  ) & slot_pids[$slot]=$!
done
# Drain all started children even after a failure; never treat partial success as complete.
for ((slot=0; slot<MAX_TOTAL; slot++)); do
  if [[ -n "${slot_pids[$slot]:-}" ]]; then
    if ! wait "${slot_pids[$slot]}"; then failed=1; fi
  fi
done
if (( failed )); then echo "ERROR: at least one run failed; postprocessing skipped" >&2; exit 1; fi
if [[ "${POSTPROCESS}" == "1" ]]; then
  if (( START_IDX == 0 && END_IDX == PLAN_N - 1 )); then
    bash postprocess_attentivefp_classification_outputs.sh
  else
    echo "Partial RUN_RANGE: automatic postprocessing skipped."
  fi
fi
