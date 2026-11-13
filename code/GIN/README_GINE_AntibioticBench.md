# GINE / edge-aware GIN AntibioticBench 脚本使用说明

本脚本包用于评估 12 个抗菌数据集在 **GINE / edge-aware GIN** 模型上的表现，分类任务和回归任务分开提交、分开输出。

## 1. 路径配置

以下命令从仓库根目录开始；将示例中的数据路径替换为实际的预划分数据目录。仓库提供的原始 CSV 不包含训练、验证和测试划分。

```bash
export REPO_ROOT="$(pwd)"
export DATA_ROOT="$(cd path/to/prepared-splits && pwd)"
cd "${REPO_ROOT}/code/GIN"
```

分类和回归结果默认写入 `outputs/GIN/` 下各自的目录；可通过 `OUT_ROOT` 覆盖。检查已有结果，避免覆盖同名输出。Slurm 提交脚本会传递代码和数据路径；集群参数和环境需要按本机调整。

无 Slurm 启动器按 `GPUS` 和 `MAX_JOBS_PER_GPU` 分配任务；任一任务失败时等待已启动任务结束、返回非零状态并跳过自动汇总。`gpu_slots.sh` 是本包启动脚本的必要组成部分。

## 2. 环境

推荐直接用你 GCN 已经能跑通的环境，例如：

```bash
conda activate chemprop-repro
```

最低依赖见 `environment_minimal.yml`：Python、RDKit、pandas、numpy、scipy、scikit-learn、torch。

## 3. 快速测试

### 3.1 分类单 run 测试

```bash
cd "${REPO_ROOT}/code/GIN"
conda activate chemprop-repro

python train_gine_classification_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${REPO_ROOT}/outputs/GIN/output_classification" \
  --model-name GINE \
  --dataset-config dataset_config.csv \
  --tasks classification \
  --split-types random \
  --seeds 0 \
  --single-run-index 0 \
  --epochs 5 \
  --patience 3 \
  --batch-size 32 \
  --eval-batch-size 128 \
  --select-metric val_pr_auc \
  --use-pos-weight \
  --threshold-mode fixed_0p5 \
  --log-every-batches 10
```

### 3.2 回归单 run 测试

```bash
python train_gine_regression_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${REPO_ROOT}/outputs/GIN/output_regression" \
  --model-name GINE \
  --dataset-config dataset_config.csv \
  --tasks regression \
  --split-types random \
  --seeds 0 \
  --single-run-index 0 \
  --epochs 5 \
  --patience 3 \
  --batch-size 32 \
  --eval-batch-size 128 \
  --select-metric val_rmse \
  --log-every-batches 10
```

## 4. Slurm 提交完整 benchmark

分类：

```bash
cd "${REPO_ROOT}/code/GIN"
bash submit_gine_classification_slurm.sh
```

回归：

```bash
bash submit_gine_regression_slurm.sh
```

默认每类任务：

```text
12 datasets × (3 split types × 5 seeds + strict_perimeter × seed_0) = 192 runs
```

分类和回归分开提交，总计 384 runs。注意：`strict_perimeter` 只有 `seed_0`；`random/scaffold/repeated_stratified_perimeter` 仍使用 `seed_0–seed_4`。

如需限制并发：

```bash
MAX_CONCURRENT=4 bash submit_gine_classification_slurm.sh
MAX_CONCURRENT=4 bash submit_gine_regression_slurm.sh
```

如需排除某个 Slurm 节点，例如 node01：

```bash
EXCLUDE_NODES=node01 bash submit_gine_classification_slurm.sh
EXCLUDE_NODES=node01 bash submit_gine_regression_slurm.sh
```

## 5. 无 Slurm 运行

分类：

```bash
tmux new -s gine_cls
cd "${REPO_ROOT}/code/GIN"
GPUS="0,1,2,3" MAX_JOBS_PER_GPU=1 bash run_gine_classification_no_slurm.sh
```

回归：

```bash
tmux new -s gine_reg
cd "${REPO_ROOT}/code/GIN"
GPUS="0,1,2,3" MAX_JOBS_PER_GPU=1 bash run_gine_regression_no_slurm.sh
```

如果只有一张 GPU，使用默认 `GPUS="0"` 即可。

## 6. 训练完成后汇总

分类：

```bash
bash postprocess_gine_classification_outputs.sh
```

回归：

```bash
bash postprocess_gine_regression_outputs.sh
```

两类都汇总：

```bash
bash postprocess_gine_all_outputs.sh
```

## 7. 输出文件

每个 run 输出到：

```text
outputs/GIN/output_classification/<dataset_name>/classification/<split_type>/seed_<seed>/
outputs/GIN/output_regression/<dataset_name>/regression/<split_type>/seed_<seed>/
```

每个 run 包含：

```text
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
best_model.pth
run.log
```


## 8. 按数据划分阶段控制预测与诊断

本版本采用明确的 split 职责：

```text
train:      逐分子预测 + 整体指标，不计算 NN 或 Activity Cliff
validation: 逐分子预测 + 整体指标，不计算 NN 或 Activity Cliff
test:       逐分子预测 + 整体指标 + NN 诊断 + Activity Cliff 诊断
```

因此，`predictions_train.csv` 和 `predictions_validation.csv` 只保留真实值、预测值、误差/分类标签以及计算整体指标所需字段；`nn_*` 和 `ac_*` 字段只存在于 `predictions_test.csv`。`diagnostics_metrics.csv` 与 `diagnostic_subgroup_metrics.csv` 也只汇总 test set 的 NN 与 Activity Cliff 结果。

NN 诊断始终以 train set 为 reference library，使用 Morgan radius=2、nBits=2048 和 Tanimoto similarity。训练完成后的严格 pairwise Activity Cliff 后处理仅计算 `test_test` 范围，不再计算 `all_internal`。

## 9. 当前 GINE 模型设置

本脚本实现的是 **GINE / edge-aware GIN**：

```text
SMILES
  → RDKit 分子图
  → 原子特征 node features
  → 键特征 edge features
  → GINE message passing
  → mean pooling
  → MLP head
```

### 9.1 原子特征

```text
atomic number, degree, formal charge, hybridization,
aromaticity, ring membership, total hydrogen count, scaled atomic mass
```

### 9.2 键特征

```text
bond type: single / double / triple / aromatic / unknown
conjugation flag
ring-bond flag
bond stereo category / unknown
```

### 9.3 GINE 层

每一层采用 edge-aware message passing：

```text
m_{u→v} = ReLU(h_u + edge_encoder(e_uv))
h_v' = MLP((1 + eps) × h_v + sum m_{u→v})
```

其中 `eps` 默认可学习。

## 10. 默认强基线超参数

本版本仅保留 GINE / edge-aware GIN，不再包含 vanilla GIN；也不再强制为了和 GCN 完全同量级而使用 `hidden_dim=256, num_layers=4, mlp_hidden_dim=256`。当前分类和回归提交脚本均锁定为以下 GINE benchmark preset：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| hidden_dim | 300 | GINE 的原子隐藏表示维度；比 256 稍大，增强表达能力 |
| num_layers | 5 | 传播到约 5-hop 原子邻域，适合更充分利用键特征 |
| dropout | 0.20 | 控制 GINE 较强表达能力带来的过拟合风险 |
| mlp_hidden_dim | 300 | 与 hidden_dim 保持一致，避免预测头成为过小瓶颈 |
| pooling | mean | 分子大小不同，用 mean pooling 降低分子大小偏差 |
| lr | 5e-4 | 比 1e-3 更保守，适配更深和带边特征的 GINE |
| weight_decay | 1e-5 | 轻度 L2 正则 |
| batch_size | 64 | A800/A100 类 GPU 通常稳定；显存不足可改 32 |
| epochs | 300 | 给深度模型足够收敛机会 |
| patience | 40 | 早停，validation 长期不提升即停止 |
| 分类 best checkpoint | val_pr_auc | 分类主排序指标，更适合不平衡 active 数据 |
| 回归 best checkpoint | val_rmse | 回归主误差指标，稳定选择最优 checkpoint |
| 分类阈值 | fixed_0p5 | 主 benchmark 固定阈值；val_best_f1 仅作为敏感性分析字段保存 |

这些固定值是“强基线合理参数”，不是用 test set 调出来的最终最优参数。当前脚本不会通过环境变量改变这些核心超参数，以避免不同作业意外使用不同配置。若未来正式开展 train/validation 调参，应生成新的 preset 和独立结果目录，不应依据 test set 修改当前配置。

## 11. 自动检查

`postprocess_gine_classification_outputs.sh` 和 `postprocess_gine_regression_outputs.sh` 最后会运行：

```bash
python validate_gine_outputs.py --out-root <OUT_ROOT> --task <classification|regression> --strict
```

检查内容包括：

- train/validation 预测文件是否含有整体指标所需字段；
- train/validation 是否错误包含 `nn_*` 或 `ac_*` 诊断字段；
- test 是否完整包含 NN 与 Activity Cliff 字段；
- `metrics_full.csv` 是否同时包含 train、validation、test 整体指标和 test 诊断汇总；
- 汇总文件以及 test-test Activity Cliff pair summary 是否存在。

旧版本产生的 landscape roughness 目录和 all-internal AC 文件不会被 postprocess 自动删除；本版本的 pairwise 后处理仅计算 test-test 范围。
