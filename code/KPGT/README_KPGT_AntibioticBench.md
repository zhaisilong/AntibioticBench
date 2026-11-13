# KPGT AntibioticBench：分类与回归分开提交版

## 1. 本脚本包完成的任务

本包用于评估 12 个抗菌数据集在 KPGT 上的分类与回归表现，并按两份 AntibioticBench 统一输出规范保存结果。

- 分类和回归分别提交、分别输出。
- `random`、`scaffold`、`repeated_stratified_perimeter`：每个数据集运行 `seed_0`–`seed_4`。
- `strict_perimeter`：每个数据集只运行 `seed_0`。
- 每类任务共 `12 × (5 + 5 + 5 + 1) = 192` 个 run。
- 训练阶段每个 epoch 只评估 train 和 validation；test 不参与早停或模型选择。
- 加载 validation 选出的最佳 checkpoint 后，test 只评估一次。
- train：逐分子预测 + 整体指标。
- validation：逐分子预测 + 整体指标。
- test：逐分子预测 + 整体指标 + 以 train 为参考库的 NN 诊断和 Activity Cliff 诊断。

## 2. 文件清单

```text
train_kpgt_antibioticbench.py       单个 run 的训练、预测、指标与诊断
kpgt_benchmark_utils.py             指标、NN/AC 诊断、KPGT 数据预处理工具
summarize_kpgt_outputs.py           192 个 run 的汇总脚本
compute_landscape_roughness.py       可选的模型无关分类/回归 roughness
dataset_config.csv                  12 个数据集、菌属、MIC boundary 阈值
kpgt_classification_array.slurm     分类 Slurm array 子任务
kpgt_regression_array.slurm         回归 Slurm array 子任务
submit_kpgt_classification_slurm.sh 分类提交脚本
submit_kpgt_regression_slurm.sh     回归提交脚本
postprocess_kpgt_classification.sh  分类汇总
postprocess_kpgt_regression.sh      回归汇总
```

## 3. 路径配置

从仓库根目录进入 `code/KPGT`。准备好按任务、划分方式和种子组织的 train、validation、test CSV；仓库中的完整数据表不含这些划分。按实际环境设置：

```bash
cd code/KPGT
export DATA_ROOT="relative/or/absolute/path/to/prepared-splits"
export KPGT_ROOT="relative/or/absolute/path/to/KPGT-source"
# 如权重不在 KPGT 默认位置，额外设置：
export KPGT_PRETRAINED_MODEL="relative/or/absolute/path/to/base.pth"
```

`KPGT_ROOT` 必须包含 `src/` 和 `scripts/preprocess_downstream_dataset.py`。脚本目录自动识别；`SCRIPT_DIR`、`BENCHMARK_ROOT`、`OUT_ROOT`、`DATASET_CONFIG` 均可覆盖。默认输出和缓存位于仓库的 `outputs/KPGT/` 与 `.cache/kpgt/`。Slurm 作业应通过本目录的提交脚本启动，以传递源码目录。

请使用可被计算节点访问的路径。代码按现状提供，不保证可执行或复现；运行前请检查依赖、数据和环境。

## 4. 先检查 192 个输入 run 是否齐全

```bash
cd code/KPGT

source ~/miniforge3/etc/profile.d/conda.sh
conda activate KPGT

python train_kpgt_antibioticbench.py \
  --task classification \
  --data-root "${DATA_ROOT}" \
  --dataset-config dataset_config.csv \
  --validate-inputs

python train_kpgt_antibioticbench.py \
  --task regression \
  --data-root "${DATA_ROOT}" \
  --dataset-config dataset_config.csv \
  --validate-inputs
```

两个命令都应显示：

```text
Validated all 192 ... runs
```

## 5. 先做短时测试

### 5.1 分类测试：仅运行 array index 0，训练 3 个 epoch

```bash
cd code/KPGT

RUN_MODE=test \
EPOCHS=3 \
PATIENCE=2 \
MAX_CONCURRENT=1 \
POSTPROCESS=0 \
RESUME=0 \
bash submit_kpgt_classification_slurm.sh
```

### 5.2 回归测试

```bash
RUN_MODE=test \
EPOCHS=3 \
PATIENCE=2 \
MAX_CONCURRENT=1 \
POSTPROCESS=0 \
RESUME=0 \
bash submit_kpgt_regression_slurm.sh
```

测试通过后，建议删除测试 run 的输出，或正式任务设置 `FORCE=1` 覆盖。否则在 `RESUME=1` 时，存在 `_SUCCESS` 的测试 run 会被跳过。

## 6. 提交正式分类任务

```bash
cd code/KPGT

RUN_MODE=batch \
MAX_CONCURRENT=4 \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
COMPUTE_LANDSCAPE=0 \
CONDA_ENV=KPGT \
bash submit_kpgt_classification_slurm.sh
```

默认提交：

```text
array = 0-191%4
```

即最多同时运行 4 个 GPU 子任务。可根据 GPU 数量修改 `MAX_CONCURRENT`。
如需排除 Slurm 节点，可在提交时设置 `EXCLUDE_NODES`；默认不强制排除任何节点。

## 7. 提交正式回归任务

```bash
cd code/KPGT

RUN_MODE=batch \
MAX_CONCURRENT=4 \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
COMPUTE_LANDSCAPE=0 \
CONDA_ENV=KPGT \
bash submit_kpgt_regression_slurm.sh
```

`POSTPROCESS=1` 会提交一个 `afterok` 汇总作业；只有全部 192 个 array 子任务成功后才汇总。若有任一子任务失败，先修复并重跑失败任务，再手工执行对应的 `postprocess_*.sh`。

## 8. 固定超参数

本脚本默认固定：

```text
KPGT config                  base
pretrained checkpoint        models/pretrained/base/base.pth
learning rate                3e-5
dropout                      0.1
weight decay                 1e-5
batch size                   32
evaluation batch size        64
maximum epochs               300
early stopping patience      20
predictor layers             2
predictor hidden dimension   256
warmup                       10% training steps
gradient clipping            5.0
advanced strategies          FLAG/LLRD/L2SP/ReInit 全部关闭
```

分类 checkpoint：默认按 `val_pr_auc` 最大选择。固定阈值指标统一使用 0.5；`val_best_f1_threshold` 只作为敏感性分析输出，不用于主 benchmark。

回归 checkpoint：默认按 `val_rmse` 最小选择；模型训练目标是经过 train 均值和标准差标准化后的 `log10_mic`，最终预测恢复为原始 `log10_mic`。

两个选择指标都可以在提交时覆盖，例如旧脚本一致的分类选择标准可设为 `SELECT_METRIC=val_roc_auc`。但主 benchmark 应在运行前固定选择标准，不能依据 test 结果更换。

### 为什么采用这一组

KPGT 官方 README 的分类和回归示例均使用 `lr=3e-5、batch_size=32、dropout=0、weight_decay=0`，并明确说明 learning rate、dropout 和 weight decay 是可调超参数。因此，官方并没有一组适用于所有数据集的“唯一最优超参数”。

本 benchmark 不再对每个数据集单独搜索超参数，是为了避免不同模型因搜索预算不同而造成比较偏差。这里保留官方示例的核心学习率和 batch size，在你之前的 8 组 KPGT 搜索范围内选择较温和的 `dropout=0.1、weight_decay=1e-5`，作为跨 12 个数据集统一使用的正则化折中。它不是“官方最优值”，而是固定 benchmark 配置。

若需要做严格的官方示例对照，可提交时覆盖：

```bash
DROPOUT=0 WEIGHT_DECAY=0 LR=3e-5 ...
```

建议主 benchmark 只保留一组固定配置；官方零正则化配置最多作为补充敏感性实验，不应根据 test 结果选择配置。

## 9. 输出目录结构

单个分类 run 示例：

```text
output_classification/
└── KPGT/
    └── 1_Staphylococcus_median_16_MolWt_50_1000_value_le_8192_multitask/
        └── classification/
            └── random/
                └── seed_0/
```

单个回归 run 同理，位于 `output_regression/KPGT/.../regression/...`。

每个 run 保存：

```text
metrics_full.csv
metrics_train.csv
metrics_validation.csv
metrics_test.csv
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
best_model.pth
train_kpgt_antibioticbench.used.py
kpgt_benchmark_utils.used.py
dataset_config.used.csv
_SUCCESS 或 _FAILED
```

`train_log.csv` 和 `training_curve.csv` 每个 epoch 一行，保存 train loss、validation loss、学习率以及分类或回归训练指标。Slurm 的逐行标准输出和错误日志保存在：

```text
output_classification/slurm_logs/
output_regression/slurm_logs/
```

## 10. NN 与 Activity Cliff 诊断定义

只在 test 上计算，train 是唯一 reference library：

```text
NN fingerprint              Morgan radius=2, nBits=2048
similarity                  Tanimoto
NN bins                     >=0.7、0.4–0.7、<0.4
Activity Cliff similarity   >=0.7
Activity Cliff MIC gap      |Δlog10MIC| >= log10(16)
```

若分类 CSV 内没有 MIC 或 logMIC 字段，分类 AC 会退化为“高相似度且 active 标签相反”的后备定义，并在 `run_info.json` 中记录 `activity_cliff_method`。回归任务始终使用真实 `log10_mic` 的 16 倍差异定义。

## 11. 汇总输出

任务全部完成后，可手工汇总：

```bash
bash postprocess_kpgt_classification.sh
bash postprocess_kpgt_regression.sh
```

主要文件：

```text
KPGT/all_run_metrics.csv
KPGT/aggregate_mean_sd.csv
KPGT/ood_change_random_vs_ood_splits.csv
KPGT/all_diagnostics_metrics.csv
KPGT/_combined_summary/classification_summary_like_ml/
KPGT/_combined_summary/regression_summary_like_ml/
```

中途检查不完整结果：

```bash
ALLOW_INCOMPLETE=1 bash postprocess_kpgt_classification.sh
ALLOW_INCOMPLETE=1 bash postprocess_kpgt_regression.sh
```

正式结果不要使用 `ALLOW_INCOMPLETE=1`。

模型无关 roughness 只需要计算一次。若此前的 ML/GCN benchmark 已经生成，可直接复用；若尚未生成，可运行：

```bash
COMPUTE_LANDSCAPE=1 bash postprocess_kpgt_classification.sh
COMPUTE_LANDSCAPE=1 bash postprocess_kpgt_regression.sh
```

输出为：

```text
KPGT/_landscape_roughness/landscape_roughness_by_split_seed.csv
KPGT/_landscape_roughness/landscape_roughness_mean_sd.csv
KPGT/_landscape_roughness/landscape_roughness_molecule_level.csv
KPGT/_regression_landscape_roughness/regression_landscape_roughness_by_split_seed.csv
KPGT/_regression_landscape_roughness/regression_landscape_roughness_mean_sd.csv
KPGT/_regression_landscape_roughness/regression_landscape_roughness_molecule_level.csv
```

## 12. 重要说明

1. 脚本不会在训练过程中查看 test，因此不存在用 test 选择 epoch 的信息泄漏。
2. `RESUME=1` 只跳过含 `_SUCCESS` 的 run；含 `_FAILED` 或没有完成标记的 run 会重新运行。`FORCE=1` 只覆盖 run 输出；只有共享 KPGT 预处理缓存损坏时才设置 `FORCE_PREPROCESS=1`。
3. KPGT 官方实现依赖 DGL、RDKit、Transformers 和预训练权重；本脚本复用官方 `MoleculeDataset`、`Collator_tune`、`LiGhTPredictor` 和 `preprocess_downstream_dataset.py`。
4. KPGT/DGL 的部分 GPU 运算可能仍存在非确定性；脚本固定 Python、NumPy、PyTorch 和 DataLoader seed，并把软件版本写入 `run_info.json`。
5. 模型无关 landscape roughness 与模型训练解耦；本包提供 `compute_landscape_roughness.py`，但默认不重复运行。回归 test 分子的局部训练集 MIC 方差仍会在每个 KPGT run 内计算。
