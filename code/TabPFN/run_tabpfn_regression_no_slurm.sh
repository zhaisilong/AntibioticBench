#!/usr/bin/env bash
set -euo pipefail

# AntibioticBench TabPFN-3 regression launcher (no Slurm)

CONDA_ENV="${CONDA_ENV:-tabpfn}"
GPU_ID="${GPU_ID:-0}"
CPU_THREADS="${CPU_THREADS:-8}"
RUN_MODE="${RUN_MODE:-test}"          # test | batch
RUN_RANGE="${RUN_RANGE:-}"
SINGLE_RUN_INDEX="${SINGLE_RUN_INDEX:-}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-256}"
POSTPROCESS="${POSTPROCESS:-1}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd -- "${SCRIPT_DIR}" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared benchmark split directory}"
if [[ ! -d "$DATA_ROOT" ]]; then echo "ERROR: DATA_ROOT is not a directory." >&2; exit 1; fi
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${REPO_ROOT}}/outputs/TabPFN/regression}"
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="${PWD}/$OUT_ROOT" ;; esac
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-$OUT_ROOT/feature_cache}"
KV_CACHE_ROOT="${KV_CACHE_ROOT:-$OUT_ROOT/kv_cache}"
case "$FEATURE_CACHE_ROOT" in /*) ;; *) FEATURE_CACHE_ROOT="${PWD}/$FEATURE_CACHE_ROOT" ;; esac
case "$KV_CACHE_ROOT" in /*) ;; *) KV_CACHE_ROOT="${PWD}/$KV_CACHE_ROOT" ;; esac
SAVE_FITTED_KV_CACHE="${SAVE_FITTED_KV_CACHE:-0}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda is unavailable; set CONDA_SH to the conda initialization script." >&2
  exit 1
fi
conda activate "$CONDA_ENV"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/regression_launcher_$(date +%Y%m%d_%H%M%S).log"

args=(
  python -u train_tabpfn_regression_antibioticbench.py
  --data-root "$DATA_ROOT"
  --out-root "$OUT_ROOT"
  --run-mode "$RUN_MODE"
  --predict-batch-size "$PRED_BATCH_SIZE"
  --feature-cache-root "$FEATURE_CACHE_ROOT"
  --kv-cache-root "$KV_CACHE_ROOT"
  --postprocess "$POSTPROCESS"
  --expected-datasets 12
  --split-types random,scaffold,repeated_stratified_perimeter,strict_perimeter
  --seeds 0,1,2,3,4
  --strict-perimeter-seeds 0
)

[[ -n "$RUN_RANGE" ]] && args+=(--run-range "$RUN_RANGE")
[[ -n "$SINGLE_RUN_INDEX" ]] && args+=(--single-run-index "$SINGLE_RUN_INDEX")
[[ "$RESUME" == "1" ]] && args+=(--resume)
[[ "$FORCE" == "1" ]] && args+=(--force)
[[ "$FAIL_FAST" == "1" ]] && args+=(--fail-fast)
[[ "$ALLOW_MISSING" == "1" ]] && args+=(--allow-missing)
[[ "$SAVE_FITTED_KV_CACHE" == "1" ]] && args+=(--save-fitted-kv-cache)

mkdir -p "$OUT_ROOT" "$FEATURE_CACHE_ROOT" "$KV_CACHE_ROOT"

echo "================================================================================" | tee -a "$LOG"
echo "TabPFN-3 regression | no fine-tuning | fit_with_cache | no train prediction" | tee -a "$LOG"
echo "GPU physical id : $GPU_ID  (inside process this becomes cuda:0)" | tee -a "$LOG"
echo "CPU threads     : $CPU_THREADS" | tee -a "$LOG"
echo "RUN_MODE        : $RUN_MODE" | tee -a "$LOG"
echo "RUN_RANGE       : ${RUN_RANGE:-ALL}" | tee -a "$LOG"
echo "PRED_BATCH_SIZE : $PRED_BATCH_SIZE" | tee -a "$LOG"
echo "SAVE_KV_STATE   : $SAVE_FITTED_KV_CACHE" | tee -a "$LOG"
echo "FIT_MODE        : fit_with_cache (live KV cache is process-local memory)" | tee -a "$LOG"
echo "TRAIN_PRED      : disabled (avoid self-context evaluation)" | tee -a "$LOG"
echo "CUBLAS_WORKSPACE_CONFIG: $CUBLAS_WORKSPACE_CONFIG" | tee -a "$LOG"
echo "================================================================================" | tee -a "$LOG"

"${args[@]}" 2>&1 | tee -a "$LOG"
