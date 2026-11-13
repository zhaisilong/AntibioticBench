#!/bin/bash
#SBATCH --ntasks=1

set -Eeuo pipefail

trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}" >&2; exit ${rc}' ERR

: "${MANIFEST:?MANIFEST is required}"
: "${TASK:?TASK is required}"
: "${SCRIPT_DIR:?SCRIPT_DIR is required}"
: "${CONDA_ENV:=unimol}"
: "${EPOCHS:=100}"
: "${PATIENCE:=20}"
: "${BATCH_SIZE:=32}"
: "${EVAL_BATCH_SIZE:=64}"
: "${LEARNING_RATE:=1e-4}"
: "${WARMUP_RATIO:=0.06}"
: "${POOLER_DROPOUT:=0.10}"
: "${MAX_NORM:=1.0}"
: "${CONF_SIZE:=11}"
: "${CONFORMER_SEED:=0}"
: "${SHARED_CACHE_ROOT:?SHARED_CACHE_ROOT is required}"
: "${NUM_WORKERS:=4}"
: "${CPU_THREADS:=1}"
: "${LOG_INTERVAL:=20}"
: "${USE_AMP:=1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LINE_NO=$((TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "${MANIFEST}")
if [[ -z "${ROW}" ]]; then
  echo "No manifest row for array task ${TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r RUN_INDEX ROW_TASK DATASET_NAME SPLIT_TYPE SEED TRAIN_CSV VALIDATION_CSV TEST_CSV OUT_DIR <<< "${ROW}"
SHARED_CACHE_DIR="${SHARED_CACHE_ROOT}/${DATASET_NAME}"

if [[ "${ROW_TASK}" != "${TASK}" ]]; then
  echo "Manifest task mismatch: ROW_TASK=${ROW_TASK}, TASK=${TASK}" >&2
  exit 3
fi
if [[ "${SPLIT_TYPE}" == "strict_perimeter" && "${SEED}" != "0" ]]; then
  echo "Invalid strict_perimeter seed=${SEED}; only seed_0 is allowed." >&2
  exit 4
fi
if [[ ! -f "${SHARED_CACHE_DIR}/completed.flag" ]]; then
  echo "Shared cache missing or incomplete for ${DATASET_NAME}" >&2
  echo "Run submit_unimol_3d_cache_slurm.sh first." >&2
  exit 5
fi
if (( NUM_WORKERS + 1 > ${SLURM_CPUS_PER_TASK:-1} )); then
  echo "NUM_WORKERS=${NUM_WORKERS} plus main process exceeds allocated CPUs=${SLURM_CPUS_PER_TASK:-1}" >&2
  exit 6
fi

mkdir -p "${OUT_DIR}"

# Activate conda robustly.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Cannot find conda initialization script." >&2
  exit 5
fi
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}${UNIMOL_REPO:+:${UNIMOL_REPO}}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

echo "================================================================================================"
echo "Uni-Mol AntibioticBench array run"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "array_task_id=${TASK_ID}"
echo "run_index=${RUN_INDEX}"
echo "task=${TASK}"
echo "dataset_name=${DATASET_NAME}"
echo "split_type=${SPLIT_TYPE}"
echo "seed=${SEED}"
echo "conda_env=${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "CPU_THREADS=${CPU_THREADS}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-NA}"
echo "================================================================================================"
python --version
nvidia-smi || true

if [[ "${TASK}" == "classification" ]]; then
  TRAIN_SCRIPT="${SCRIPT_DIR}/train_unimol_classification_antibioticbench.py"
else
  TRAIN_SCRIPT="${SCRIPT_DIR}/train_unimol_regression_antibioticbench.py"
fi

AMP_ARGS=(--use-amp)
if [[ "${USE_AMP}" != "1" ]]; then
  AMP_ARGS=(--no-use-amp)
fi

srun --unbuffered python "${TRAIN_SCRIPT}" \
  --train-csv "${TRAIN_CSV}" \
  --validation-csv "${VALIDATION_CSV}" \
  --test-csv "${TEST_CSV}" \
  --out-dir "${OUT_DIR}" \
  --shared-cache-dir "${SHARED_CACHE_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --split-type "${SPLIT_TYPE}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --pooler-dropout "${POOLER_DROPOUT}" \
  --max-norm "${MAX_NORM}" \
  --conf-size "${CONF_SIZE}" \
  --conformer-seed "${CONFORMER_SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --cpu-threads "${CPU_THREADS}" \
  --log-interval "${LOG_INTERVAL}" \
  "${AMP_ARGS[@]}"

echo "Finished: $(date --iso-8601=seconds)"
