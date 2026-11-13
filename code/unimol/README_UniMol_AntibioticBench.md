# Uni-Mol v1 AntibioticBench：共享 3D 缓存版

本脚本包用于 12 个抗菌数据集的 Uni-Mol v1 分类和回归 benchmark。

## 1. 核心修改

不再在每个 split/seed/run 中重复生成 3D 构象。

新的流程是：

```text
每个数据集的 classification + regression 全部 split CSV
        ↓
仅提取 exact SMILES，不读取标签进入缓存
        ↓
每个唯一 SMILES 固定生成 11 个 no-H 3D 构象
        ↓
每个数据集保存一份共享缓存
        ↓
classification / regression / 所有 split / 所有训练 seed 共同读取
        ↓
每个 run 仍然端到端微调 Uni-Mol，embedding 每次重新计算
```

共享的是原子 token、坐标、距离和 edge type 等静态 3D 输入，不共享微调后的模型 embedding。

## 2. 构象协议

```text
model               Uni-Mol v1
hydrogen            no-H
remove_hs           True
conf_size           11
conformer_seed      0
RDKit embedding seed 0,1,2,...,10
```

同一数据集只生成一次。缓存目录默认：

```bash
../../outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0
```

每个数据集目录包含：

```text
dataset_name/
├── unimol_v1_noh_conformers.pkl
├── molecules.csv
├── cache_info.json
└── completed.flag
```

`molecules.csv` 会记录每个分子的 3D/2D fallback 情况。

## 3. 11 个构象在训练和评估中的使用

训练：

```text
每个 molecule 每个 epoch 只取 1 个缓存构象
构象索引按 molecule index + epoch + training seed 确定性循环
```

这样不会把训练集直接扩大 11 倍。

Validation：

```text
每个 epoch 对 11 个构象分别预测
按 molecule 对 11 个预测求平均
用平均预测计算 val PR-AUC / val RMSE
并据此选择最佳 checkpoint
```

最终 train / validation / test：

```text
均输出 11 构象平均预测
test 仍只在最佳 checkpoint 确定后最终评估一次
```

## 4. 固定训练参数

```text
optimizer             Adam
betas                 (0.9, 0.99)
epsilon               1e-6
learning_rate         1e-4
weight_decay          0
scheduler             polynomial decay
warmup_ratio          0.06
batch_size            32
eval_batch_size       64
epochs                 100
patience               20
pooler_dropout         0.1
gradient_clip_norm     1.0
mixed_precision        enabled
classification loss    BCEWithLogitsLoss
class weight           none
classification select  max validation PR-AUC
regression select      min validation RMSE
classification cutoff  0.5
```

## 5. CPU 控制

### 5.1 3D 缓存预处理

默认：

```text
CACHE_CPUS_PER_TASK=16
CACHE_CPU_WORKERS=15
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

- `CACHE_CPUS_PER_TASK`：Slurm 分配的 CPU 核心数。
- `CACHE_CPU_WORKERS`：实际并行生成 RDKit 构象的进程数。
- 默认分配 16 核，其中 15 个 RDKit worker，保留 1 核给父进程进行特征整理和写盘。
- 每个 worker 的数学库线程固定为 1，防止 16 个进程各自再启动多线程。

### 5.2 GPU 模型训练

默认：

```text
CPUS_PER_TASK=12
NUM_WORKERS=4
CPU_THREADS=1
```

- `CPUS_PER_TASK`：Slurm 给每个 GPU run 分配 12 个 CPU 核心。
- `NUM_WORKERS=4`：PyTorch DataLoader 使用 4 个 CPU 子进程准备 batch。
- `CPU_THREADS=1`：训练主进程的 OMP/MKL/PyTorch CPU 线程为 1。

共享缓存已经生成后，训练阶段不需要大量 CPU。默认 12 核比较保守；稳定测试后可降到 6–8 核。

必须满足：

```text
NUM_WORKERS + 1 <= CPUS_PER_TASK
```

## 6. 路径配置

从仓库根目录开始执行以下配置，将示例路径替换为实际路径；随后只进入一次 `code/unimol`。脚本位置自动识别；必须将 `DATA_ROOT` 设为已准备好的分类、回归和各 split/seed 划分的根目录。仓库数据压缩包内的原始表不是预先划分的数据。

```bash
export DATA_ROOT="$(cd path/to/prepared/splits && pwd)"
# 可选：使用本地源码时指定外部仓库，否则使用环境中安装的 unimol_tools。
# export UNIMOL_REPO="$(cd path/to/unimol && pwd)"
# 可选：覆盖输出或缓存位置。
# export OUT_ROOT="${PWD}/path/to/output"
# export SHARED_CACHE_ROOT="${PWD}/path/to/cache"
cd code/unimol
```

相对路径相对于执行命令时所在目录解析。默认输出和缓存位于仓库的 `outputs/unimol/` 下。现有 conda 环境和模型权重必须由使用者另行准备。Slurm 提交器会向作业显式传入脚本目录，不能直接从 Slurm 临时脚本位置推断仓库。

作业和缓存清单包含本地文件路径，仅供执行使用，不应直接公开分享。日志和可分享的结果元数据不主动记录主机名或完整本地路径；上传前仍需检查其他依赖产生的日志。

## 7. 第一步：环境预检查

```bash
conda activate unimol

python preflight_unimol.py \
  --data-root "${DATA_ROOT}"
```

末尾应出现：

```text
PREFLIGHT_OK
```

并确认：

```text
dataset_count: 12
CUDA available: True
```

## 8. 第二步：先测试一个数据集的共享缓存

```bash
CONDA_ENV=unimol \
CPU_WORKERS=8 \
bash run_single_cache_test.sh
```

默认测试第一个数据集。

缓存生成后检查：

```bash
python validate_unimol_shared_cache.py \
  --cache-root "${SHARED_CACHE_ROOT:-../../outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}" \
  --expected-datasets 1 \
  --conf-size 11 \
  --conformer-seed 0 \
  --deep
```

## 9. 第三步：正式预生成 12 个数据集缓存

```bash
CONDA_ENV=unimol \
PARTITION=gpu \
CACHE_CPUS_PER_TASK=16 \
CACHE_CPU_WORKERS=15 \
MAX_CONCURRENT=1 \
bash submit_unimol_3d_cache_slurm.sh
```

这个任务只申请 CPU 和内存，不申请 GPU。

如果集群有单独 CPU 分区：

```bash
PARTITION=cpu \
CACHE_CPUS_PER_TASK=16 \
CACHE_CPU_WORKERS=15 \
bash submit_unimol_3d_cache_slurm.sh
```

12 个任务全部完成后验证：

```bash
python validate_unimol_shared_cache.py \
  --cache-root "${SHARED_CACHE_ROOT:-../../outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}" \
  --expected-datasets 12 \
  --conf-size 11 \
  --conformer-seed 0
```

应出现：

```text
CACHE_VALIDATION_OK
```

## 10. 第四步：单 run GPU 测试

分类：

```bash
CONDA_ENV=unimol \
CUDA_VISIBLE_DEVICES=0 \
bash run_single_test_classification.sh
```

回归：

```bash
CONDA_ENV=unimol \
CUDA_VISIBLE_DEVICES=0 \
bash run_single_test_regression.sh
```

测试脚本只跑 3 epochs。

## 11. 第五步：正式提交分类任务

```bash
CONDA_ENV=unimol \
PARTITION=gpu \
CPUS_PER_TASK=12 \
NUM_WORKERS=4 \
CPU_THREADS=1 \
MAX_CONCURRENT=1 \
bash submit_unimol_classification_slurm.sh
```

## 12. 第六步：正式提交回归任务

```bash
CONDA_ENV=unimol \
PARTITION=gpu \
CPUS_PER_TASK=12 \
NUM_WORKERS=4 \
CPU_THREADS=1 \
MAX_CONCURRENT=1 \
bash submit_unimol_regression_slurm.sh
```

每类任务均为：

```text
random                         12 × 5 = 60
scaffold                       12 × 5 = 60
repeated_stratified_perimeter  12 × 5 = 60
strict_perimeter               12 × 1 = 12
总计                                   192
```

`strict_perimeter` 强制只使用 `seed_0`。

## 13. Slurm GPU 申请方式

默认：

```bash
--gpus=1
```

如果集群要求：

```bash
--gres=gpu:1
```

提交时使用：

```bash
SBATCH_GPU_ARGS="--gres=gpu:1" \
bash submit_unimol_classification_slurm.sh
```

## 14. 每个 run 输出

```text
best_model.pth
model_0.pth
metrics_full.csv
predictions_train.csv
predictions_validation.csv
predictions_test.csv
diagnostics_metrics.csv
diagnostic_subgroup_metrics.csv
train_log.csv
training_curve.csv
train_log_long.csv
feature_summary.json
model_params.json
run_info.json
run.log
status.json
```

不会再在每个 run 中创建 `_conformer_cache/` 或 `_test_conformer_cache/`。

## 15. 输出协议

- train：整体预测和整体指标。
- validation：整体预测和整体指标，并用于最佳 checkpoint 选择。
- test：整体预测、整体指标、NN 诊断和 Activity Cliff 诊断。
- 分类阈值固定为 0.5。
- 分类不使用类别权重。
- NN reference 固定为当前 run 的 train set。
- NN 指纹固定 Morgan radius=2、2048 bits、Tanimoto。
- 模型 embedding 在每个 run 中随端到端微调重新计算，绝不缓存或跨 run 复用。

## 16. 汇总

分类：

```bash
OUT_ROOT=../../outputs/unimol/classification \
bash postprocess_unimol_classification.sh
```

回归：

```bash
OUT_ROOT=../../outputs/unimol/regression \
bash postprocess_unimol_regression.sh
```

## 17. 重要说明

1. 缓存按 exact SMILES 字符串索引，不会重新进行去盐、中和、互变异构体标准化或 canonical SMILES 重写。
2. 分类和回归可共享缓存，因为缓存不包含 `active` 或 `log10_mic` 标签。
3. test SMILES 参与无监督的几何预处理不构成标签泄漏；test 标签仍不参与训练和模型选择。
4. 如修改 `conf_size`、`conformer_seed`、`remove_hs`、Uni-Mol版本或输入 SMILES，必须重新生成缓存。
5. 不要同时启动两个任务写同一个数据集缓存。
6. 本地运行时设置 `STOP_ON_ERROR=0` 可在单个 run 失败后继续剩余任务；只要有失败，最终仍返回非零并跳过自动汇总。
