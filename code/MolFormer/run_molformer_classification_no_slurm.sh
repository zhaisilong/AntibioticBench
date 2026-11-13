#!/usr/bin/env bash
# MoLFormer AntibioticBench classification launcher for a server WITHOUT Slurm.
# One worker processes one run at a time on its assigned physical GPU.
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${PWD}/${SCRIPT_DIR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
[[ "${BENCHMARK_ROOT}" = /* ]] || BENCHMARK_ROOT="${PWD}/${BENCHMARK_ROOT}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared split-data directory}"
[[ "${DATA_ROOT}" = /* ]] || DATA_ROOT="${PWD}/${DATA_ROOT}"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT}/outputs/MolFormer/classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
DATASET_CONFIG="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_config.csv}"
[[ "${DATASET_CONFIG}" = /* ]] || DATASET_CONFIG="${PWD}/${DATASET_CONFIG}"
MOLFORMER_ROOT="${MOLFORMER_ROOT:-${BENCHMARK_ROOT}/.cache/molformer}"
[[ "${MOLFORMER_ROOT}" = /* ]] || MOLFORMER_ROOT="${PWD}/${MOLFORMER_ROOT}"
CONDA_ENV="${CONDA_ENV:-molformer}"
MODEL_ID="${MODEL_ID:-ibm-research/MoLFormer-XL-both-10pct}"
MODEL_REVISION="${MODEL_REVISION:-compat-v4}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

# Shared Hugging Face cache.
export HF_HOME="${HF_HOME:-${MOLFORMER_ROOT}/hf_cache}"
[[ "${HF_HOME}" = /* ]] || export HF_HOME="${PWD}/${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
[[ "${HF_HUB_CACHE}" = /* ]] || export HF_HUB_CACHE="${PWD}/${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}"

# Execution controls.
RUN_MODE="${RUN_MODE:-batch}"                 # test | batch
GPUS="${GPUS:-0}"                             # e.g. "0", "0 1", or "0,1"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-191}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
POSTPROCESS="${POSTPROCESS:-1}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
PREFETCH_MODEL="${PREFETCH_MODEL:-1}"
SHOW_RUN_OUTPUT="${SHOW_RUN_OUTPUT:-0}"

# Fixed MoLFormer benchmark hyperparameters.
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.10}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
SELECT_METRIC="${SELECT_METRIC:-val_pr_auc}"
MAX_LENGTH="${MAX_LENGTH:-202}"
CANONICALIZE_NON_ISOMERIC="${CANONICALIZE_NON_ISOMERIC:-1}"

init_conda() {
  if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "ERROR: conda initialization script was not found." >&2
    exit 2
  fi
  conda activate "${CONDA_ENV}"
}

[[ -d "${SCRIPT_DIR}" ]] || { echo "ERROR: SCRIPT_DIR not found: ${SCRIPT_DIR}" >&2; exit 2; }
[[ -f "${SCRIPT_DIR}/train_molformer_classification_antibioticbench.py" ]] || {
  echo "ERROR: classification training script not found in ${SCRIPT_DIR}" >&2; exit 2;
}
[[ -f "${DATASET_CONFIG}" ]] || { echo "ERROR: dataset config not found: ${DATASET_CONFIG}" >&2; exit 2; }

case "${RUN_MODE}" in
  test) START_INDEX=0; END_INDEX=0; POSTPROCESS=0 ;;
  batch) ;;
  *) echo "ERROR: RUN_MODE must be test or batch." >&2; exit 2 ;;
esac
[[ "${START_INDEX}" =~ ^[0-9]+$ && "${END_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: START_INDEX and END_INDEX must be integers." >&2; exit 2;
}
(( START_INDEX >= 0 && END_INDEX <= 191 && START_INDEX <= END_INDEX )) || {
  echo "ERROR: valid run-index range is 0-191; received ${START_INDEX}-${END_INDEX}." >&2; exit 2;
}
[[ "${MAX_JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: MAX_JOBS_PER_GPU must be a positive integer." >&2; exit 2;
}

init_conda
cd "${SCRIPT_DIR}"
mkdir -p "${OUT_ROOT}/no_slurm_logs"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"

GPUS="${GPUS//,/ }"
read -r -a GPU_ARRAY <<< "${GPUS}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "ERROR: no GPU was specified." >&2; exit 2; }
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid GPU id: ${gpu}" >&2; exit 2; }
done

RUN_TAG="${RUN_TAG:-classification_$(date +%Y%m%d_%H%M%S)}"
SESSION_LOG_DIR="${OUT_ROOT}/no_slurm_logs/${RUN_TAG}"
mkdir -p "${SESSION_LOG_DIR}"
FAIL_FLAG="${SESSION_LOG_DIR}/.failed"
FAIL_LIST="${SESSION_LOG_DIR}/failed_runs.txt"
: > "${FAIL_LIST}"
rm -f "${FAIL_FLAG}"

cleanup() {
  local rc=$?
  if (( rc != 0 )); then touch "${FAIL_FLAG}" 2>/dev/null || true; fi
  for pid in $(jobs -pr); do
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit "${rc}"
}
trap cleanup INT TERM

if [[ "${VALIDATE_INPUTS}" == "1" ]]; then
  echo "[preflight] validating all 192 classification inputs..."
  python -u train_molformer_classification_antibioticbench.py \
    --data-root "${DATA_ROOT}" --dataset-config "${DATASET_CONFIG}" --validate-inputs \
    | tee "${SESSION_LOG_DIR}/validate_inputs.log"
fi
if [[ "${PREFETCH_MODEL}" == "1" ]]; then
  echo "[preflight] validating/caching MoLFormer checkpoint..."
  python -u prefetch_molformer_model.py \
    --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    --local-files-only "${LOCAL_FILES_ONLY}" \
    | tee "${SESSION_LOG_DIR}/prefetch_model.log"
fi

nvidia-smi || true
echo "===== MoLFormer classification: no-Slurm launcher ====="
python --version
echo "run_mode=${RUN_MODE} range=${START_INDEX}-${END_INDEX}"
echo "gpus=${GPU_ARRAY[*]} max_jobs_per_gpu=${MAX_JOBS_PER_GPU}"
echo "model configured; revision=${MODEL_REVISION}"
echo "epochs=${EPOCHS} patience=${PATIENCE} lr=${LR} constant batch=${BATCH_SIZE}/${EVAL_BATCH_SIZE}"

run_one() {
  local run_index="$1" gpu="$2" log_file
  log_file=$(printf "%s/run_%03d_gpu%s.log" "${SESSION_LOG_DIR}" "${run_index}" "${gpu}")
  if [[ "${STOP_ON_ERROR}" == "1" && -f "${FAIL_FLAG}" ]]; then
    echo "[worker gpu=${gpu}] stop flag exists; run ${run_index} will not start."
    return 0
  fi
  local -a cmd=(
    python -u train_molformer_classification_antibioticbench.py
    --data-root "${DATA_ROOT}"
    --out-root "${OUT_ROOT}"
    --dataset-config "${DATASET_CONFIG}"
    --single-run-index "${run_index}"
    --model-id "${MODEL_ID}"
    --model-revision "${MODEL_REVISION}"
    --local-files-only "${LOCAL_FILES_ONLY}"
    --molformer-repo "${MOLFORMER_ROOT}"
    --canonicalize-non-isomeric "${CANONICALIZE_NON_ISOMERIC}"
    --max-length "${MAX_LENGTH}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --batch-size "${BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --head-dropout "${HEAD_DROPOUT}"
    --grad-clip "${GRAD_CLIP}"
    --num-workers "${NUM_WORKERS}"
    --eval-num-workers "${EVAL_NUM_WORKERS}"
    --selection-metric "${SELECT_METRIC}"
    --threshold 0.5
    --resume "${RESUME}"
  )
  [[ "${FORCE}" == "1" ]] && cmd+=(--force)

  echo "[start] $(date -Is) task=classification run=${run_index} gpu=${gpu}"
  {
    echo "===== run metadata ====="
    echo "date=$(date -Is) gpu=${gpu}"
    nvidia-smi -i "${gpu}" || true
    echo "===== training ====="
  } > "${log_file}"

  local status=0
  set +e
  if [[ "${SHOW_RUN_OUTPUT}" == "1" ]]; then
    RUN_LOG_FILE="${log_file}" CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
    status=${PIPESTATUS[0]}
  else
    RUN_LOG_FILE="${log_file}" CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >> "${log_file}" 2>&1
    status=$?
  fi
  set -e
  if (( status == 0 )); then
    echo "[done]  $(date -Is) task=classification run=${run_index} gpu=${gpu}"
  else
    echo "[FAIL]  $(date -Is) task=classification run=${run_index} gpu=${gpu} status=${status}" | tee -a "${log_file}" >&2
    printf "%s\tgpu=%s\tstatus=%s\tlog=%s\n" "${run_index}" "${gpu}" "${status}" "$(basename -- "${log_file}")" >> "${FAIL_LIST}"
    touch "${FAIL_FLAG}"
    return "${status}"
  fi
}

worker_loop() {
  local worker_id="$1" total_workers="$2" gpu="$3"
  local index=$(( START_INDEX + worker_id )) worker_status=0
  while (( index <= END_INDEX )); do
    if [[ "${STOP_ON_ERROR}" == "1" && -f "${FAIL_FLAG}" ]]; then
      echo "[worker ${worker_id}, gpu=${gpu}] detected failure flag; stopping before run ${index}."
      break
    fi
    if ! run_one "${index}" "${gpu}"; then
      worker_status=1
      [[ "${STOP_ON_ERROR}" == "1" ]] && break
    fi
    index=$(( index + total_workers ))
  done
  return "${worker_status}"
}

TOTAL_WORKERS=$(( ${#GPU_ARRAY[@]} * MAX_JOBS_PER_GPU ))
declare -a PIDS=()
worker_id=0
for gpu in "${GPU_ARRAY[@]}"; do
  for ((slot=0; slot<MAX_JOBS_PER_GPU; slot++)); do
    worker_loop "${worker_id}" "${TOTAL_WORKERS}" "${gpu}" &
    PIDS+=("$!")
    worker_id=$((worker_id + 1))
  done
done

launcher_status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then launcher_status=1; fi
done
if [[ -s "${FAIL_LIST}" ]]; then
  launcher_status=1
  echo "Failed runs:" >&2
  cat "${FAIL_LIST}" >&2
fi
if (( launcher_status != 0 )); then
  echo "Classification launcher finished with failures. Postprocessing was not run." >&2
  exit 1
fi

if [[ "${POSTPROCESS}" == "1" ]]; then
  if [[ "${START_INDEX}" == "0" && "${END_INDEX}" == "191" ]]; then
    echo "[postprocess] building complete classification summaries..."
    BENCHMARK_ROOT="${BENCHMARK_ROOT}" SCRIPT_DIR="${SCRIPT_DIR}" DATA_ROOT="${DATA_ROOT}" \
      OUT_ROOT="${OUT_ROOT}" DATASET_CONFIG="${DATASET_CONFIG}" CONDA_ENV="${CONDA_ENV}" \
      ALLOW_INCOMPLETE=0 bash "${SCRIPT_DIR}/postprocess_molformer_classification.sh" \
      2>&1 | tee "${SESSION_LOG_DIR}/postprocess.log"
  elif [[ "${ALLOW_INCOMPLETE}" == "1" ]]; then
    echo "[postprocess] building partial classification summaries..."
    BENCHMARK_ROOT="${BENCHMARK_ROOT}" SCRIPT_DIR="${SCRIPT_DIR}" DATA_ROOT="${DATA_ROOT}" \
      OUT_ROOT="${OUT_ROOT}" DATASET_CONFIG="${DATASET_CONFIG}" CONDA_ENV="${CONDA_ENV}" \
      ALLOW_INCOMPLETE=1 bash "${SCRIPT_DIR}/postprocess_molformer_classification.sh" \
      2>&1 | tee "${SESSION_LOG_DIR}/postprocess.log"
  else
    echo "[postprocess] skipped because only run indices ${START_INDEX}-${END_INDEX} were requested."
  fi
fi

echo "===== all requested MoLFormer classification runs completed: $(date -Is) ====="
