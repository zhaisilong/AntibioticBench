#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset-split directory}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${PROJECT_ROOT}/outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}"
[[ "${SHARED_CACHE_ROOT}" = /* ]] || SHARED_CACHE_ROOT="${PWD}/${SHARED_CACHE_ROOT}"
CONDA_ENV="${CONDA_ENV:-unimol}"
UNIMOL_REPO="${UNIMOL_REPO:-}"
if [[ -n "${UNIMOL_REPO}" ]]; then
  UNIMOL_REPO="$(cd "${UNIMOL_REPO}" && pwd)"
fi

PARTITION="${PARTITION:-gpu}"
CACHE_CPUS_PER_TASK="${CACHE_CPUS_PER_TASK:-16}"
CACHE_CPU_WORKERS="${CACHE_CPU_WORKERS:-15}"
CACHE_MEMORY="${CACHE_MEMORY:-128G}"
CACHE_TIME_LIMIT="${CACHE_TIME_LIMIT:-72:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"

CONF_SIZE="${CONF_SIZE:-11}"
CONFORMER_SEED="${CONFORMER_SEED:-0}"
CACHE_MODE="${CACHE_MODE:-fast}"
CACHE_MAX_ATOMS="${CACHE_MAX_ATOMS:-256}"
CACHE_CHUNKSIZE="${CACHE_CHUNKSIZE:-8}"
CACHE_FORCE="${CACHE_FORCE:-0}"

if (( CACHE_CPU_WORKERS + 1 > CACHE_CPUS_PER_TASK )); then
  echo "CACHE_CPU_WORKERS=${CACHE_CPU_WORKERS} plus parent process exceeds CACHE_CPUS_PER_TASK=${CACHE_CPUS_PER_TASK}." >&2
  exit 1
fi

mkdir -p "${SHARED_CACHE_ROOT}/_manifests" "${SHARED_CACHE_ROOT}/_slurm_logs"
CACHE_MANIFEST="${SHARED_CACHE_ROOT}/_manifests/unimol_shared_cache_manifest.tsv"

python "${SCRIPT_DIR}/build_unimol_cache_manifest.py" \
  --data-root "${DATA_ROOT}" \
  --cache-root "${SHARED_CACHE_ROOT}" \
  --manifest "${CACHE_MANIFEST}" \
  --expected-datasets 12

N_DATASETS=$(( $(wc -l < "${CACHE_MANIFEST}") - 1 ))
if (( N_DATASETS <= 0 )); then
  echo "No datasets in cache manifest." >&2
  exit 1
fi

echo "Submitting shared 3D cache: ${N_DATASETS} datasets."
echo "This is CPU preprocessing; no GPU is requested."
export CACHE_MANIFEST SCRIPT_DIR CONDA_ENV UNIMOL_REPO CONF_SIZE CONFORMER_SEED CACHE_CPU_WORKERS CACHE_MODE CACHE_MAX_ATOMS CACHE_CHUNKSIZE CACHE_FORCE
JOB_ID=$(sbatch \
  --parsable \
  --job-name=unimol_3dcache \
  --partition="${PARTITION}" \
  --cpus-per-task="${CACHE_CPUS_PER_TASK}" \
  --mem="${CACHE_MEMORY}" \
  --time="${CACHE_TIME_LIMIT}" \
  --array="0-$((N_DATASETS-1))%${MAX_CONCURRENT}" \
  --output="${SHARED_CACHE_ROOT}/_slurm_logs/unimol_cache_%A_%a.out" \
  --error="${SHARED_CACHE_ROOT}/_slurm_logs/unimol_cache_%A_%a.err" \
  --export=ALL \
  "${SCRIPT_DIR}/unimol_cache_job.slurm"
)

echo "Submitted shared cache job: ${JOB_ID}"

echo "After all 12 cache jobs complete, submit classification and regression."
