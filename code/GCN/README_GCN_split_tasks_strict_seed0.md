# GCN AntibioticBench：分类/回归分开提交版

以下命令从仓库根目录开始。先将 `path/to/prepared-splits` 替换为预先划分的数据目录；仓库中的原始 CSV 并非训练/验证/测试划分。

```bash
export REPO_ROOT="$(pwd)"
export DATA_ROOT="$(cd path/to/prepared-splits && pwd)"
```

默认输出位于仓库的 `outputs/GCN/`，可通过 `OUT_ROOT` 覆盖。脚本、依赖与集群参数需按环境检查。

无 Slurm 启动器通过 `GPUS` 和 `MAX_JOBS_PER_GPU` 控制每张 GPU 的并发任务数。如果任一任务失败，启动器会等待已启动任务结束、返回非零状态并跳过自动汇总。`gpu_slots.sh` 是本包启动脚本的必要组成部分。

本版本做了两类修改：

1. 分类和回归分开提交、分开输出：
   - 分类输出：`${REPO_ROOT}/outputs/GCN/output_classification`
   - 回归输出：`${REPO_ROOT}/outputs/GCN/output_regression`
2. `strict_perimeter` 默认只生成 `seed_0` 的任务。`random`、`scaffold`、`repeated_stratified_perimeter` 仍使用 `seed_0` 到 `seed_4`。

因此每个任务的 run 数为：

```text
random: 12 datasets × 5 seeds = 60
scaffold: 12 datasets × 5 seeds = 60
repeated_stratified_perimeter: 12 datasets × 5 seeds = 60
strict_perimeter: 12 datasets × 1 seed = 12
合计：192 runs / task
```

分类和回归合计为 384 runs，而不是旧版 480 runs。

## 1. 需要替换/新增的文件

把本包里的文件放到：

```bash
${REPO_ROOT}/code/GCN
```

核心文件：

```text
train_gcn_antibioticbench.py          # 已新增 --strict-perimeter-seeds，默认 strict_perimeter 只跑 seed_0
gcn_array_job.slurm                   # Slurm array 子任务脚本，已支持 strict seed 规则
run_gcn_classification_no_slurm.sh    # 无 Slurm：分类任务
run_gcn_regression_no_slurm.sh        # 无 Slurm：回归任务
submit_gcn_classification_slurm.sh    # Slurm：分类任务
submit_gcn_regression_slurm.sh        # Slurm：回归任务
postprocess_gcn_classification.sh     # 分类汇总
postprocess_gcn_regression.sh         # 回归汇总
```

`summarize_gcn_outputs.py`、`activity_cliff_pair_analysis.py`、`dataset_config.csv` 保持兼容。

## 2. 无 Slurm 服务器运行

### 分类任务

```bash
cd "${REPO_ROOT}/code/GCN"

GPUS=0,1,2 \
MAX_JOBS_PER_GPU=1 \
CONDA_ENV=gcn_ab \
bash run_gcn_classification_no_slurm.sh
```

分类结果会写入：

```text
${REPO_ROOT}/outputs/GCN/output_classification
```

### 回归任务

```bash
cd "${REPO_ROOT}/code/GCN"

GPUS=0,1,2 \
MAX_JOBS_PER_GPU=1 \
CONDA_ENV=gcn_ab \
bash run_gcn_regression_no_slurm.sh
```

回归结果会写入：

```text
${REPO_ROOT}/outputs/GCN/output_regression
```

## 3. Slurm 服务器运行

### 分类任务

```bash
cd "${REPO_ROOT}/code/GCN"
CONDA_ENV=gcn_ab bash submit_gcn_classification_slurm.sh
```

### 回归任务

```bash
cd "${REPO_ROOT}/code/GCN"
CONDA_ENV=gcn_ab bash submit_gcn_regression_slurm.sh
```

## 4. 汇总结果

分类全部完成后：

```bash
cd "${REPO_ROOT}/code/GCN"
OUT_ROOT="${REPO_ROOT}/outputs/GCN/output_classification" CONDA_ENV=gcn_ab bash postprocess_gcn_classification.sh
```

回归全部完成后：

```bash
cd "${REPO_ROOT}/code/GCN"
OUT_ROOT="${REPO_ROOT}/outputs/GCN/output_regression" CONDA_ENV=gcn_ab bash postprocess_gcn_regression.sh
```

如果 activity-cliff pair 文件太大，可以只输出 summary 和 molecule-level 文件：

```bash
SKIP_PAIR_FILES=1 OUT_ROOT="${REPO_ROOT}/outputs/GCN/output_classification" CONDA_ENV=gcn_ab bash postprocess_gcn_classification.sh
SKIP_PAIR_FILES=1 OUT_ROOT="${REPO_ROOT}/outputs/GCN/output_regression" CONDA_ENV=gcn_ab bash postprocess_gcn_regression.sh
```

## 5. 检查完成数量

分类目标：192 个 `metrics_full.csv`

```bash
find "${REPO_ROOT}/outputs/GCN/output_classification" -path "*/classification/*/seed_*/*metrics_full.csv" | wc -l
```

回归目标：192 个 `metrics_full.csv`

```bash
find "${REPO_ROOT}/outputs/GCN/output_regression" -path "*/regression/*/seed_*/*metrics_full.csv" | wc -l
```

检查是否错误地生成了 strict_perimeter 的 seed_1 到 seed_4：

```bash
find "${REPO_ROOT}/outputs/GCN/output_classification" -path "*/strict_perimeter/seed_[1-4]" -type d
find "${REPO_ROOT}/outputs/GCN/output_regression" -path "*/strict_perimeter/seed_[1-4]" -type d
```

正常情况下应该没有输出。

## 6. 重要参数

`train_gcn_antibioticbench.py` 新增参数：

```bash
--strict-perimeter-seeds 0
```

默认就是 `0`。所以即使命令里写：

```bash
--split-types random,scaffold,repeated_stratified_perimeter,strict_perimeter \
--seeds 0,1,2,3,4
```

脚本也只会为 `strict_perimeter` 生成 `seed_0`，不会再生成 `strict_perimeter/seed_1`、`seed_2`、`seed_3`、`seed_4`。
