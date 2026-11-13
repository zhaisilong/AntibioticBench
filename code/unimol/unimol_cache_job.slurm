#!/bin/bash
#SBATCH --ntasks=1

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}" >&2; exit ${rc}' ERR

: "${CACHE_MANIFEST:?CACHE_MANIFEST is required}"
: "${SCRIPT_DIR:?SCRIPT_DIR is required}"
: "${CONDA_ENV:=unimol}"
: "${CONF_SIZE:=11}"
: "${CONFORMER_SEED:=0}"
: "${CACHE_CPU_WORKERS:=15}"
: "${CACHE_MODE:=fast}"
: "${CACHE_MAX_ATOMS:=256}"
: "${CACHE_CHUNKSIZE:=8}"
: "${CACHE_FORCE:=0}"

if (( CACHE_CPU_WORKERS + 1 > ${SLURM_CPUS_PER_TASK:-1} )); then
  echo "CACHE_CPU_WORKERS=${CACHE_CPU_WORKERS} plus parent process exceeds allocated CPUs=${SLURM_CPUS_PER_TASK:-1}" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LINE_NO=$((TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "${CACHE_MANIFEST}")
if [[ -z "${ROW}" ]]; then
  echo "No cache-manifest row for array task ${TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r RUN_INDEX DATASET_NAME DATASET_DIR CACHE_DIR <<< "${ROW}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Cannot find conda initialization script." >&2
  exit 3
fi
conda activate "${CONDA_ENV}"

# Slurm may run from a spool directory, so resolve relative source paths from
# the submit directory. The submit wrapper normally passes an absolute path.
UNIMOL_REPO="${UNIMOL_REPO:-}"
if [[ -n "${UNIMOL_REPO}" ]]; then
  if [[ "${UNIMOL_REPO}" != /* ]]; then
    UNIMOL_REPO="${SLURM_SUBMIT_DIR:-$PWD}/${UNIMOL_REPO}"
  fi
  UNIMOL_REPO="$(cd "${UNIMOL_REPO}" && pwd)"
fi
export PYTHONPATH="${SCRIPT_DIR}${UNIMOL_REPO:+:${UNIMOL_REPO}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "================================================================================================"
echo "Uni-Mol shared 3D cache job"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "array_task_id=${TASK_ID}"
echo "dataset_name=${DATASET_NAME}"
echo "conf_size=${CONF_SIZE}"
echo "conformer_seed=${CONFORMER_SEED}"
echo "cache_cpu_workers=${CACHE_CPU_WORKERS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "================================================================================================"

FORCE_ARGS=()
if [[ "${CACHE_FORCE}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

srun --unbuffered python "${SCRIPT_DIR}/precompute_unimol_shared_3d_cache.py" \
  --dataset-dir "${DATASET_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --conf-size "${CONF_SIZE}" \
  --conformer-seed "${CONFORMER_SEED}" \
  --cpu-workers "${CACHE_CPU_WORKERS}" \
  --mode "${CACHE_MODE}" \
  --max-atoms "${CACHE_MAX_ATOMS}" \
  --chunksize "${CACHE_CHUNKSIZE}" \
  "${FORCE_ARGS[@]}"

echo "Finished: $(date --iso-8601=seconds)"
