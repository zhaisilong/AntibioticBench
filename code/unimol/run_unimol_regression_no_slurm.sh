#!/usr/bin/env bash
set -Eeuo pipefail

# Uni-Mol v1 AntibioticBench regression launcher for a server without Slurm.
# Runs 192 jobs sequentially on one physical GPU.

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset-split directory}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/unimol/regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${PROJECT_ROOT}/outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}"
[[ "${SHARED_CACHE_ROOT}" = /* ]] || SHARED_CACHE_ROOT="${PWD}/${SHARED_CACHE_ROOT}"
UNIMOL_REPO="${UNIMOL_REPO:-}"
if [[ -n "${UNIMOL_REPO}" ]]; then
  UNIMOL_REPO="$(cd "${UNIMOL_REPO}" && pwd)"
fi
CONDA_ENV="${CONDA_ENV:-unimol}"
GPU_ID="${GPU_ID:-2}"

# CPU_LIST is optional. Empty means no taskset binding.
CPU_LIST="${CPU_LIST:-8-19}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CPU_THREADS="${CPU_THREADS:-1}"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
POOLER_DROPOUT="${POOLER_DROPOUT:-0.10}"
MAX_NORM="${MAX_NORM:-1.0}"
CONF_SIZE="${CONF_SIZE:-11}"
CONFORMER_SEED="${CONFORMER_SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
USE_AMP="${USE_AMP:-1}"

# Examples: 0-191, 0-9, or 37
RUN_RANGE="${RUN_RANGE:-0-191}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
POSTPROCESS="${POSTPROCESS:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
SHOW_RUN_OUTPUT="${SHOW_RUN_OUTPUT:-1}"

cd "${SCRIPT_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Cannot find conda initialization script." >&2
  exit 1
fi
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}${UNIMOL_REPO:+:${UNIMOL_REPO}}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT_ROOT}/_manifests" "${OUT_ROOT}/_local_logs" "${OUT_ROOT}/_incomplete_backups"
SESSION_STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_LOG_DIR="${OUT_ROOT}/_local_logs/session_${SESSION_STAMP}"
mkdir -p "${SESSION_LOG_DIR}"

CACHE_COMPLETED_COUNT=$(find "${SHARED_CACHE_ROOT}" -mindepth 2 -maxdepth 2 -name completed.flag | wc -l)
if [[ "${CACHE_COMPLETED_COUNT}" -ne 12 ]]; then
  echo "[ERROR] Expected 12 completed shared caches, found ${CACHE_COMPLETED_COUNT}." >&2
  exit 1
fi

MANIFEST="${OUT_ROOT}/_manifests/unimol_regression_manifest.tsv"
python -u "${SCRIPT_DIR}/build_unimol_manifest.py" \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --task regression \
  --manifest "${MANIFEST}" \
  --expected-datasets 12

N_RUNS=$(( $(wc -l < "${MANIFEST}") - 1 ))
if [[ "${N_RUNS}" -ne 192 ]]; then
  echo "[ERROR] Expected 192 regression runs, found ${N_RUNS}." >&2
  exit 1
fi

if [[ "${RUN_RANGE}" == *-* ]]; then
  START_INDEX="${RUN_RANGE%-*}"
  END_INDEX="${RUN_RANGE#*-}"
else
  START_INDEX="${RUN_RANGE}"
  END_INDEX="${RUN_RANGE}"
fi
if ! [[ "${START_INDEX}" =~ ^[0-9]+$ && "${END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Invalid RUN_RANGE=${RUN_RANGE}; use forms such as 0-191 or 12." >&2
  exit 1
fi
if (( START_INDEX < 0 || END_INDEX >= N_RUNS || START_INDEX > END_INDEX )); then
  echo "[ERROR] RUN_RANGE=${RUN_RANGE} is outside 0-$((N_RUNS - 1))." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available =", torch.cuda.is_available())
print("visible GPU count =", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one visible CUDA GPU")
print("logical cuda:0 =", torch.cuda.get_device_name(0))
PY

if [[ -n "${CPU_LIST}" ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "[ERROR] CPU_LIST is set but taskset is unavailable." >&2
    exit 1
  fi
  CPU_PREFIX=(taskset -c "${CPU_LIST}")
else
  CPU_PREFIX=()
fi

is_completed() {
  local status_file="$1"
  [[ -f "${status_file}" ]] || return 1
  python - "${status_file}" <<'PY'
import json, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        obj = json.load(f)
    raise SystemExit(0 if obj.get("status") == "completed" else 1)
except Exception:
    raise SystemExit(1)
PY
}

AMP_ARGS=(--use-amp)
if [[ "${USE_AMP}" != "1" ]]; then
  AMP_ARGS=(--no-use-amp)
fi

printf '%s\n' \
  "===== Uni-Mol regression: no Slurm =====" \
  "date=$(date -Is)" \
  "GPU_ID=${GPU_ID}" \
  "CPU_LIST=${CPU_LIST:-disabled}" \
  "NUM_WORKERS=${NUM_WORKERS}" \
  "CPU_THREADS=${CPU_THREADS}" \
  "RUN_RANGE=${START_INDEX}-${END_INDEX}" \
  "RESUME=${RESUME} FORCE=${FORCE} POSTPROCESS=${POSTPROCESS}"

failed_runs=0
for ((IDX=START_INDEX; IDX<=END_INDEX; IDX++)); do
  LINE_NO=$((IDX + 2))
  ROW=$(sed -n "${LINE_NO}p" "${MANIFEST}")
  if [[ -z "${ROW}" ]]; then
    echo "[ERROR] No manifest row for run index ${IDX}." >&2
    exit 1
  fi

  IFS=$'\t' read -r RUN_INDEX TASK DATASET_NAME SPLIT_TYPE SEED TRAIN_CSV VALIDATION_CSV TEST_CSV OUT_DIR <<< "${ROW}"
  SHARED_CACHE_DIR="${SHARED_CACHE_ROOT}/${DATASET_NAME}"
  STATUS_FILE="${OUT_DIR}/status.json"
  RUN_LOG=$(printf "%s/run_%03d.log" "${SESSION_LOG_DIR}" "${IDX}")

  if [[ "${RESUME}" == "1" ]] && is_completed "${STATUS_FILE}"; then
    echo "[SKIP] run=${IDX} completed: ${DATASET_NAME} | ${SPLIT_TYPE} | seed_${SEED}"
    continue
  fi

  if [[ -d "${OUT_DIR}" && -n "$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if [[ "${FORCE}" == "1" || "${RESUME}" == "1" ]]; then
      SAFE_DATASET=${DATASET_NAME//\//_}
      BACKUP_DIR="${OUT_ROOT}/_incomplete_backups/run_${IDX}_${SAFE_DATASET}_${SPLIT_TYPE}_seed_${SEED}_${SESSION_STAMP}"
      echo "[BACKUP] Moving existing incomplete output to the backup directory"
      mkdir -p "$(dirname "${BACKUP_DIR}")"
      mv "${OUT_DIR}" "${BACKUP_DIR}"
    else
      echo "[ERROR] Output exists but is not completed" >&2
      echo "Set RESUME=1 or FORCE=1 to rerun it." >&2
      exit 1
    fi
  fi
  mkdir -p "${OUT_DIR}"

  CMD=(
    python -u "${SCRIPT_DIR}/train_unimol_regression_antibioticbench.py"
    --train-csv "${TRAIN_CSV}"
    --validation-csv "${VALIDATION_CSV}"
    --test-csv "${TEST_CSV}"
    --out-dir "${OUT_DIR}"
    --shared-cache-dir "${SHARED_CACHE_DIR}"
    --dataset-name "${DATASET_NAME}"
    --split-type "${SPLIT_TYPE}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --batch-size "${BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --learning-rate "${LEARNING_RATE}"
    --warmup-ratio "${WARMUP_RATIO}"
    --pooler-dropout "${POOLER_DROPOUT}"
    --max-norm "${MAX_NORM}"
    --conf-size "${CONF_SIZE}"
    --conformer-seed "${CONFORMER_SEED}"
    --num-workers "${NUM_WORKERS}"
    --cpu-threads "${CPU_THREADS}"
    --log-interval "${LOG_INTERVAL}"
    "${AMP_ARGS[@]}"
  )

  echo "[START] $(date -Is) run=${IDX}/$((N_RUNS - 1)) GPU=${GPU_ID} ${DATASET_NAME} | ${SPLIT_TYPE} | seed_${SEED}"
  {
    echo "date=$(date -Is)"
    echo "run_index=${IDX}"
    echo "GPU_ID=${GPU_ID}"
    echo "CPU_LIST=${CPU_LIST:-disabled}"
    echo
  } > "${RUN_LOG}"

  set +e
  if [[ "${SHOW_RUN_OUTPUT}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CPU_PREFIX[@]}" "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
    RC=${PIPESTATUS[0]}
  else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CPU_PREFIX[@]}" "${CMD[@]}" >> "${RUN_LOG}" 2>&1
    RC=$?
  fi
  set -e

  if [[ "${RC}" -ne 0 ]] || ! is_completed "${STATUS_FILE}"; then
    ((failed_runs+=1))
    echo "[FAILED] run=${IDX} exit_code=${RC}; check the local run log" >&2
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      if [[ "${RC}" -eq 0 ]]; then
        exit 1
      else
        exit "${RC}"
      fi
    fi
  else
    echo "[DONE] $(date -Is) run=${IDX}: ${DATASET_NAME} | ${SPLIT_TYPE} | seed_${SEED}"
  fi
done

if (( failed_runs > 0 )); then
  echo "[ERROR] ${failed_runs} run(s) failed; postprocessing skipped." >&2
  exit 1
fi

if [[ "${POSTPROCESS}" == "1" && "${START_INDEX}" -eq 0 && "${END_INDEX}" -eq $((N_RUNS - 1)) ]]; then
  OUT_ROOT="${OUT_ROOT}" CONDA_ENV="${CONDA_ENV}" bash "${SCRIPT_DIR}/postprocess_unimol_regression.sh"
else
  echo "[INFO] Postprocess skipped because POSTPROCESS=${POSTPROCESS} or RUN_RANGE is partial."
fi

echo "===== Regression workflow finished: $(date -Is) ====="
