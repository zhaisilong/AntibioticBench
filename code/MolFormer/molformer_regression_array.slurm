#!/usr/bin/env bash
#SBATCH --job-name=molformer_reg
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -Eeuo pipefail
SCRIPT_DIR="${SCRIPT_DIR:?Set SCRIPT_DIR or use the supplied submit script}"
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${PWD}/${SCRIPT_DIR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
[[ "${BENCHMARK_ROOT}" = /* ]] || BENCHMARK_ROOT="${PWD}/${BENCHMARK_ROOT}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared split-data directory}"
[[ "${DATA_ROOT}" = /* ]] || DATA_ROOT="${PWD}/${DATA_ROOT}"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT}/outputs/MolFormer/regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
DATASET_CONFIG="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_config.csv}"
[[ "${DATASET_CONFIG}" = /* ]] || DATASET_CONFIG="${PWD}/${DATASET_CONFIG}"
MOLFORMER_ROOT="${MOLFORMER_ROOT:-${BENCHMARK_ROOT}/.cache/molformer}"
[[ "${MOLFORMER_ROOT}" = /* ]] || MOLFORMER_ROOT="${PWD}/${MOLFORMER_ROOT}"
CONDA_ENV="${CONDA_ENV:-molformer}"
MODEL_ID="${MODEL_ID:-ibm-research/MoLFormer-XL-both-10pct}"
MODEL_REVISION="${MODEL_REVISION:-compat-v4}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
RUN_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

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
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
SELECT_METRIC="${SELECT_METRIC:-val_rmse}"
MAX_LENGTH="${MAX_LENGTH:-202}"
CANONICALIZE_NON_ISOMERIC="${CANONICALIZE_NON_ISOMERIC:-1}"

export HF_HOME="${HF_HOME:-${MOLFORMER_ROOT}/hf_cache}"
[[ "${HF_HOME}" = /* ]] || export HF_HOME="${PWD}/${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
[[ "${HF_HUB_CACHE}" = /* ]] || export HF_HUB_CACHE="${PWD}/${HF_HUB_CACHE}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p "${OUT_ROOT}/slurm_logs" "${HF_HUB_CACHE}"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"

echo "===== MoLFormer regression array task ====="
echo "date=$(date -Is) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID} task=${RUN_INDEX}"
python --version
nvidia-smi || true

cmd=(
  python -u train_molformer_regression_antibioticbench.py
  --data-root "${DATA_ROOT}"
  --out-root "${OUT_ROOT}"
  --dataset-config "${DATASET_CONFIG}"
  --single-run-index "${RUN_INDEX}"
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
  --resume "${RESUME}"
)
[[ "${FORCE}" == "1" ]] && cmd+=(--force)
"${cmd[@]}"
echo "===== regression task ${RUN_INDEX} completed at $(date -Is) ====="
