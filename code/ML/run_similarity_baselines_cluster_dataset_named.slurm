#!/usr/bin/env bash
#SBATCH --job-name=ab_sim
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
## 如果你的集群必须指定分区，请取消下一行注释并改成真实分区名，例如 cpu 或 gpu
## #SBATCH --partition=cpu

set -euo pipefail

echo "============================================================"
echo "Job name       : ${SLURM_JOB_NAME:-NA}"
echo "Job ID         : ${SLURM_JOB_ID:-NA}"
echo "Start time     : $(date)"
echo "============================================================"

# Slurm may execute a copied script; locate code from the submit directory or SCRIPT_DIR.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -z "${SCRIPT_DIR:-}" ]]; then
  submit_dir="${SUBMIT_DIR}"
  if [[ -f "${submit_dir}/run_ml.sh" ]]; then
    SCRIPT_DIR="${submit_dir}"
  elif [[ -f "${submit_dir}/code/ML/run_ml.sh" ]]; then
    SCRIPT_DIR="${submit_dir}/code/ML"
  else
    echo "ERROR: set SCRIPT_DIR to the code/ML directory" >&2; exit 1
  fi
fi
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${SUBMIT_DIR}/${SCRIPT_DIR}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing prepared benchmark splits}"
[[ "${DATA_ROOT}" = /* ]] || DATA_ROOT="${SUBMIT_DIR}/${DATA_ROOT}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/ML/output_similarity}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${SUBMIT_DIR}/${OUT_ROOT}"
RUN_TAG=${RUN_TAG:-""}
CONDA_ENV=${CONDA_ENV:-myenv}

# Modes:
#   test            one small test: Staphylococcus, random, seed_0, classification+regression, 1NN+5NN
#   batch           all requested datasets/splits/seeds
#   test_then_batch test first, then batch
RUN_MODE=${RUN_MODE:-batch}

TASKS=${TASKS:-"classification regression"}
MODELS=${MODELS:-"1nn 5nn"}
SPLIT_TYPES=${SPLIT_TYPES:-"random scaffold repeated_stratified_perimeter strict_perimeter"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-""}
WEIGHTING=${WEIGHTING:-"similarity"}
CLASSIFICATION_TARGET_COL=${CLASSIFICATION_TARGET_COL:-""}
REGRESSION_TARGET_COL=${REGRESSION_TARGET_COL:-""}
VALUE_COL=${VALUE_COL:-""}
ACTIVE_COL=${ACTIVE_COL:-""}
CLASSIFICATION_RANK_METRIC=${CLASSIFICATION_RANK_METRIC:-"test_pr_auc"}
REGRESSION_RANK_METRIC=${REGRESSION_RANK_METRIC:-"test_rmse"}

TEST_TASKS=${TEST_TASKS:-"classification regression"}
TEST_MODELS=${TEST_MODELS:-"1nn 5nn"}
TEST_DATASETS=${TEST_DATASETS:-Staphylococcus}
TEST_SPLIT_TYPES=${TEST_SPLIT_TYPES:-random}
TEST_SEEDS=${TEST_SEEDS:-0}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

if [[ -n "${CONDA_SH:-}" ]]; then
  [[ "${CONDA_SH}" = /* ]] || CONDA_SH="${SUBMIT_DIR}/${CONDA_SH}"
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
fi
conda activate "${CONDA_ENV}"

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs"
cd "${SCRIPT_DIR}"

python --version
echo "Conda env      : ${CONDA_DEFAULT_ENV:-NA}"
echo "RUN_MODE       : ${RUN_MODE}"
echo "TASKS          : ${TASKS}"
echo "MODELS         : ${MODELS}"
echo "SPLIT_TYPES    : ${SPLIT_TYPES}"
echo "SEEDS          : ${SEEDS}"
echo "DATASETS       : ${DATASETS:-ALL}"
echo "WEIGHTING      : ${WEIGHTING}"
echo "RUN_TAG        : ${RUN_TAG:-AUTO}"

for f in run_similarity_baselines.sh train_similarity_baselines.py summarize_similarity_baselines.py; do
  if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
    echo "ERROR: Missing required similarity baseline file: ${f}"
    echo "Files in SCRIPT_DIR:"
    ls -lh "${SCRIPT_DIR}"
    exit 1
  fi
done

sanitize_tag() {
  local raw="$*"
  if [[ -z "${raw}" ]]; then
    echo ""
  else
    echo "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//'
  fi
}

dataset_tag() {
  if [[ -n "${1:-}" ]]; then
    sanitize_tag "$1"
  else
    echo "all_datasets"
  fi
}

SNAPSHOT_DIR="${OUT_ROOT}/_run_snapshot_${SLURM_JOB_ID:-manual}"
mkdir -p "${SNAPSHOT_DIR}"
cp -f "${SCRIPT_DIR}/run_similarity_baselines.sh" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/train_similarity_baselines.py" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/summarize_similarity_baselines.py" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/privacy_utils.py" "${SNAPSHOT_DIR}/"
cp -f "$0" "${SNAPSHOT_DIR}/$(basename "$0")" 2>/dev/null || true

run_test() {
  local dtag mtag ttag stag setag auto_tag test_out
  dtag=$(dataset_tag "${TEST_DATASETS}")
  mtag=$(sanitize_tag "${TEST_MODELS}")
  ttag=$(sanitize_tag "${TEST_TASKS}")
  stag=$(sanitize_tag "${TEST_SPLIT_TYPES}")
  setag=$(sanitize_tag "${TEST_SEEDS}")
  auto_tag="similarity_test_${dtag}_${ttag}_${mtag}_${stag}_seeds_${setag}"
  test_out="${OUT_ROOT}/${RUN_TAG:-${auto_tag}}"

  echo "============================================================"
  echo "Running SIMILARITY TEST mode"
  echo "TEST_TASKS       : ${TEST_TASKS}"
  echo "TEST_MODELS      : ${TEST_MODELS}"
  echo "TEST_DATASETS    : ${TEST_DATASETS}"
  echo "TEST_SPLIT_TYPES : ${TEST_SPLIT_TYPES}"
  echo "TEST_SEEDS       : ${TEST_SEEDS}"
  echo "============================================================"

  TASKS="${TEST_TASKS}" \
  MODELS="${TEST_MODELS}" \
  SPLIT_TYPES="${TEST_SPLIT_TYPES}" \
  SEEDS="${TEST_SEEDS}" \
  DATASETS="${TEST_DATASETS}" \
  WEIGHTING="${WEIGHTING}" \
  CLASSIFICATION_TARGET_COL="${CLASSIFICATION_TARGET_COL}" \
  REGRESSION_TARGET_COL="${REGRESSION_TARGET_COL}" \
  VALUE_COL="${VALUE_COL}" \
  ACTIVE_COL="${ACTIVE_COL}" \
  CLASSIFICATION_RANK_METRIC="${CLASSIFICATION_RANK_METRIC}" \
  REGRESSION_RANK_METRIC="${REGRESSION_RANK_METRIC}" \
  bash "${SCRIPT_DIR}/run_similarity_baselines.sh" "${DATA_ROOT}" "${test_out}"

  echo "Checking required SIMILARITY TEST outputs ..."
  for name in metrics_full.csv predictions_test.csv diagnostics_metrics.csv diagnostic_subgroup_metrics.csv train_log.csv training_curve.csv train_log_long.csv model.joblib run_info.json; do
    if ! find "${test_out}" -name "${name}" | grep -q .; then
      echo "ERROR: TEST output missing: ${name}"
      exit 1
    fi
  done
  echo "SIMILARITY TEST mode passed."
}

run_batch() {
  local dtag mtag ttag stag setag auto_tag batch_out
  dtag=$(dataset_tag "${DATASETS}")
  mtag=$(sanitize_tag "${MODELS}")
  ttag=$(sanitize_tag "${TASKS}")
  stag=$(sanitize_tag "${SPLIT_TYPES}")
  setag=$(sanitize_tag "${SEEDS}")
  auto_tag="similarity_batch_${dtag}_${ttag}_${mtag}_${stag}_seeds_${setag}"
  batch_out="${OUT_ROOT}/${RUN_TAG:-${auto_tag}}"

  echo "============================================================"
  echo "Running SIMILARITY BATCH mode"
  echo "============================================================"

  TASKS="${TASKS}" \
  MODELS="${MODELS}" \
  SPLIT_TYPES="${SPLIT_TYPES}" \
  SEEDS="${SEEDS}" \
  DATASETS="${DATASETS}" \
  WEIGHTING="${WEIGHTING}" \
  CLASSIFICATION_TARGET_COL="${CLASSIFICATION_TARGET_COL}" \
  REGRESSION_TARGET_COL="${REGRESSION_TARGET_COL}" \
  VALUE_COL="${VALUE_COL}" \
  ACTIVE_COL="${ACTIVE_COL}" \
  CLASSIFICATION_RANK_METRIC="${CLASSIFICATION_RANK_METRIC}" \
  REGRESSION_RANK_METRIC="${REGRESSION_RANK_METRIC}" \
  bash "${SCRIPT_DIR}/run_similarity_baselines.sh" "${DATA_ROOT}" "${batch_out}"
}

case "${RUN_MODE}" in
  test)
    run_test
    ;;
  batch)
    run_batch
    ;;
  test_then_batch)
    run_test
    run_batch
    ;;
  *)
    echo "ERROR: Unknown RUN_MODE=${RUN_MODE}"
    exit 1
    ;;
esac

echo "============================================================"
echo "End time: $(date)"
echo "Similarity baseline job finished."
echo "============================================================"
