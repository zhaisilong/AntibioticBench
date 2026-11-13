#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset-split directory}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/unimol/regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
CONDA_ENV="${CONDA_ENV:-unimol}"
UNIMOL_REPO="${UNIMOL_REPO:-}"
if [[ -n "${UNIMOL_REPO}" ]]; then
  UNIMOL_REPO="$(cd "${UNIMOL_REPO}" && pwd)"
fi
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${PROJECT_ROOT}/outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}"
[[ "${SHARED_CACHE_ROOT}" = /* ]] || SHARED_CACHE_ROOT="${PWD}/${SHARED_CACHE_ROOT}"

PARTITION="${PARTITION:-gpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEMORY="${MEMORY:-128G}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
SBATCH_GPU_ARGS="${SBATCH_GPU_ARGS:---gpus=1}"
read -r -a GPU_ARGS <<< "${SBATCH_GPU_ARGS}"

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
NUM_WORKERS="${NUM_WORKERS:-4}"
CPU_THREADS="${CPU_THREADS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
USE_AMP="${USE_AMP:-1}"

mkdir -p "${OUT_ROOT}/_manifests" "${OUT_ROOT}/_slurm_logs"

if [[ ! -d "${SHARED_CACHE_ROOT}" ]]; then
  echo "Shared 3D cache root does not exist." >&2
  echo "Run submit_unimol_3d_cache_slurm.sh first." >&2
  exit 1
fi
CACHE_COMPLETED_COUNT=$(find "${SHARED_CACHE_ROOT}" -mindepth 2 -maxdepth 2 -name completed.flag | wc -l)
if [[ "${CACHE_COMPLETED_COUNT}" -ne 12 ]]; then
  echo "Expected 12 completed dataset caches, found ${CACHE_COMPLETED_COUNT}." >&2
  echo "Run or finish submit_unimol_3d_cache_slurm.sh before training." >&2
  exit 1
fi
if (( NUM_WORKERS + 1 > CPUS_PER_TASK )); then
  echo "NUM_WORKERS=${NUM_WORKERS} plus main process exceeds CPUS_PER_TASK=${CPUS_PER_TASK}." >&2
  exit 1
fi
MANIFEST="${OUT_ROOT}/_manifests/unimol_regression_manifest.tsv"

python "${SCRIPT_DIR}/build_unimol_manifest.py" \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --task regression \
  --manifest "${MANIFEST}" \
  --expected-datasets 12

N_RUNS=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( N_RUNS <= 0 )); then
  echo "No runs in manifest." >&2
  exit 1
fi

echo "Submitting regression: ${N_RUNS} runs; strict_perimeter seed_0 only."
TASK=regression
export MANIFEST TASK SCRIPT_DIR CONDA_ENV UNIMOL_REPO EPOCHS PATIENCE BATCH_SIZE EVAL_BATCH_SIZE LEARNING_RATE WARMUP_RATIO POOLER_DROPOUT MAX_NORM CONF_SIZE CONFORMER_SEED SHARED_CACHE_ROOT NUM_WORKERS CPU_THREADS LOG_INTERVAL USE_AMP
JOB_ID=$(sbatch \
  --parsable \
  --job-name=unimol_reg \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  "${GPU_ARGS[@]}" \
  --array="0-$((N_RUNS-1))%${MAX_CONCURRENT}" \
  --output="${OUT_ROOT}/_slurm_logs/unimol_reg_%A_%a.out" \
  --error="${OUT_ROOT}/_slurm_logs/unimol_reg_%A_%a.err" \
  --export=ALL \
  "${SCRIPT_DIR}/unimol_array_job.slurm"
)

echo "Submitted regression job: ${JOB_ID}"
echo "Manifest created under the output root."

echo "After completion:"
echo "  Run postprocess_unimol_regression.sh with the same OUT_ROOT."
