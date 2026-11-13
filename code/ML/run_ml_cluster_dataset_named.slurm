#!/usr/bin/env bash
#SBATCH --job-name=ab_ml
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
## 如果你的集群必须指定分区，请取消下一行注释并改成真实分区名，例如 cpu 或 gpu
## #SBATCH --partition=cpu

set -euo pipefail

echo "============================================================"
echo "Job name       : ${SLURM_JOB_NAME:-NA}"
echo "Job ID         : ${SLURM_JOB_ID:-NA}"
echo "Start time     : $(date)"
echo "============================================================"

# ===== 1. 固定项目路径 =====
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

# OUT_ROOT 现在只作为“总输出根目录”。
# 每次 test / batch / svm 会在 OUT_ROOT 下再创建一个有含义的结果目录，
# 不再用 test_${SLURM_JOB_ID} 或 svm_${SLURM_JOB_ID} 这种任务号命名结果。
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/ML/output}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${SUBMIT_DIR}/${OUT_ROOT}"

# 可选：手动指定本次运行结果目录名。
# 例如：
#   RUN_TAG=Staphylococcus_random_seed0_rf_test sbatch run_ml_cluster.slurm
RUN_TAG=${RUN_TAG:-""}

# ===== 2. 运行模式 =====
# 可选：
#   test            只跑一个模型、一个菌属、一个 split、一个 seed
#   batch           批量跑 RF + XGB + LightGBM
#   svm             单独跑 SVM
#   test_then_batch 先测试，成功后再批量跑 RF + XGB + LightGBM
RUN_MODE=${RUN_MODE:-batch}

# ===== 3. 默认训练范围，可在 sbatch 前用环境变量覆盖 =====
MODELS=${MODELS:-"rf xgb lightgbm"}
SPLIT_TYPES=${SPLIT_TYPES:-"random scaffold repeated_stratified_perimeter strict_perimeter"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-""}
COMPUTE_LANDSCAPE=${COMPUTE_LANDSCAPE:-0}

# test 模式默认值
TEST_MODEL=${TEST_MODEL:-rf}
TEST_DATASETS=${TEST_DATASETS:-Staphylococcus}
TEST_SPLIT_TYPES=${TEST_SPLIT_TYPES:-random}
TEST_SEEDS=${TEST_SEEDS:-0}

# ===== 4. 多线程设置：让底层数学库使用 Slurm 分配的 CPU 核心 =====
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# ===== 5. 激活 conda 环境 =====
if [[ -n "${CONDA_SH:-}" ]]; then
  [[ "${CONDA_SH}" = /* ]] || CONDA_SH="${SUBMIT_DIR}/${CONDA_SH}"
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
fi
conda activate "${CONDA_ENV:-myenv}"

python --version
echo "Conda env      : ${CONDA_DEFAULT_ENV:-NA}"

# ===== 6. 检查脚本和目录 =====
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs"

cd "${SCRIPT_DIR}"

echo "RUN_MODE       : ${RUN_MODE}"
echo "MODELS         : ${MODELS}"
echo "SPLIT_TYPES    : ${SPLIT_TYPES}"
echo "SEEDS          : ${SEEDS}"
echo "DATASETS       : ${DATASETS:-ALL}"
echo "COMPUTE_LANDSCAPE: ${COMPUTE_LANDSCAPE}"
echo "RUN_TAG        : ${RUN_TAG:-AUTO}"

for f in run_ml.sh train_ml.py summarize_ml.py roughness.py ml_notes.md; do
  if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
    echo "ERROR: Missing required file: ${f}"
    echo "Files in SCRIPT_DIR:"
    ls -lh "${SCRIPT_DIR}"
    exit 1
  fi
done


sanitize_tag() {
  # 把空格、斜杠、逗号等不适合做目录名的字符统一替换为下划线。
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

# 保存本次运行脚本快照，便于复现。
# 快照仍保留 job ID，便于和 Slurm 日志对应；正式结果目录不再用 job ID 命名。
SNAPSHOT_DIR="${OUT_ROOT}/_run_snapshot_${SLURM_JOB_ID:-manual}"
mkdir -p "${SNAPSHOT_DIR}"
cp -f "${SCRIPT_DIR}/run_ml.sh" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/train_ml.py" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/summarize_ml.py" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/roughness.py" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/ml_notes.md" "${SNAPSHOT_DIR}/"
cp -f "${SCRIPT_DIR}/privacy_utils.py" "${SNAPSHOT_DIR}/"
cp -f "$0" "${SNAPSHOT_DIR}/$(basename "$0")" 2>/dev/null || true

run_test() {
  local dtag
  local stag
  local setag
  local auto_tag
  local test_out

  dtag=$(dataset_tag "${TEST_DATASETS}")
  stag=$(sanitize_tag "${TEST_SPLIT_TYPES}")
  setag=$(sanitize_tag "${TEST_SEEDS}")
  auto_tag="test_${dtag}_${TEST_MODEL}_${stag}_seeds_${setag}"
  test_out="${OUT_ROOT}/${RUN_TAG:-${auto_tag}}"

  echo "============================================================"
  echo "Running TEST mode"
  echo "TEST_MODEL       : ${TEST_MODEL}"
  echo "TEST_DATASETS    : ${TEST_DATASETS}"
  echo "TEST_SPLIT_TYPES : ${TEST_SPLIT_TYPES}"
  echo "TEST_SEEDS       : ${TEST_SEEDS}"
  echo "============================================================"

  MODELS="${TEST_MODEL}" \
  SPLIT_TYPES="${TEST_SPLIT_TYPES}" \
  SEEDS="${TEST_SEEDS}" \
  DATASETS="${TEST_DATASETS}" \
  COMPUTE_LANDSCAPE=0 \
  bash "${SCRIPT_DIR}/run_ml.sh" "${DATA_ROOT}" "${test_out}"

  echo "Checking required TEST outputs ..."
  for name in metrics_full.csv predictions_test.csv diagnostics_metrics.csv train_log.csv training_curve.csv train_log_long.csv model.joblib; do
    if ! find "${test_out}" -name "${name}" | grep -q .; then
      echo "ERROR: TEST output missing: ${name}"
      exit 1
    fi
  done
  echo "TEST mode passed."
}

run_batch() {
  local dtag
  local mtag
  local stag
  local setag
  local auto_tag
  local batch_out

  dtag=$(dataset_tag "${DATASETS}")
  mtag=$(sanitize_tag "${MODELS}")
  stag=$(sanitize_tag "${SPLIT_TYPES}")
  setag=$(sanitize_tag "${SEEDS}")
  auto_tag="batch_${dtag}_${mtag}_${stag}_seeds_${setag}"
  batch_out="${OUT_ROOT}/${RUN_TAG:-${auto_tag}}"

  echo "============================================================"
  echo "Running BATCH mode"
  echo "============================================================"

  MODELS="${MODELS}" \
  SPLIT_TYPES="${SPLIT_TYPES}" \
  SEEDS="${SEEDS}" \
  DATASETS="${DATASETS}" \
  COMPUTE_LANDSCAPE="${COMPUTE_LANDSCAPE}" \
  bash "${SCRIPT_DIR}/run_ml.sh" "${DATA_ROOT}" "${batch_out}"
}

run_svm() {
  local dtag
  local stag
  local setag
  local auto_tag
  local svm_out

  dtag=$(dataset_tag "${DATASETS}")
  stag=$(sanitize_tag "${SPLIT_TYPES}")
  setag=$(sanitize_tag "${SEEDS}")
  auto_tag="svm_${dtag}_${stag}_seeds_${setag}"
  svm_out="${OUT_ROOT}/${RUN_TAG:-${auto_tag}}"

  echo "============================================================"
  echo "Running SVM mode"
  echo "============================================================"

  MODELS="svm" \
  SPLIT_TYPES="${SPLIT_TYPES}" \
  SEEDS="${SEEDS}" \
  DATASETS="${DATASETS}" \
  COMPUTE_LANDSCAPE=0 \
  bash "${SCRIPT_DIR}/run_ml.sh" "${DATA_ROOT}" "${svm_out}"
}

case "${RUN_MODE}" in
  test)
    run_test
    ;;
  batch)
    run_batch
    ;;
  svm)
    run_svm
    ;;
  test_then_batch)
    run_test
    run_batch
    ;;
  *)
    echo "ERROR: Unknown RUN_MODE=${RUN_MODE}"
    echo "Allowed: test, batch, svm, test_then_batch"
    exit 1
    ;;
esac

echo "============================================================"
echo "Finished time   : $(date)"
echo "============================================================"
