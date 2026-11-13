
## Classification class weighting

The classification workflow uses unweighted `BCEWithLogitsLoss()` for every run. It does not calculate or pass `pos_weight`; positive and negative samples therefore have the same per-sample loss weight. Class counts are still written to the run log for transparency.

# AttentiveFP AntibioticBench 脚本包

用于在 12 个抗菌数据集上分别运行 AttentiveFP 分类和回归任务。模型使用 PyG 内置 `torch_geometric.nn.models.AttentiveFP`，复用 GINE 参考脚本的 43 维原子特征、14 维键特征、统一指标及 test-only NN/Activity Cliff 诊断。

## 路径配置

运行脚本前设置 `DATA_ROOT` 指向预先划分的数据。可设置 `OUT_ROOT`、`PROJECT_DIR` 或 `CONDA_SH` 覆盖默认位置。以下输出路径相对于仓库根目录；`BENCHMARK_ROOT` 可作为数据根目录的兼容配置。

## 目录

```text
${DATA_ROOT}
code/AttentiveFP
outputs/AttentiveFP/classification
outputs/AttentiveFP/regression
```

从仓库根目录进入模型目录：

```bash
cd code/AttentiveFP
chmod +x *.sh *.slurm *.py
```

## 固定参数

`43/14` 输入特征，`hidden_channels=200`，`out_channels=1`，`num_layers=2`，`num_timesteps=2`，`dropout=0.20`；AdamW，lr=5e-4，weight_decay=1e-5，grad_clip=5；batch=128，eval_batch=512；epochs=300，early-stopping patience=40；ReduceLROnPlateau factor=0.5、patience=10、min_lr=1e-6。分类按最大 val PR-AUC、固定阈值0.5；回归按最小 val RMSE。

## 环境与数据检查

默认环境名是 `AttentiveFP`，环境名不同可通过 `CONDA_ENV=实际名称` 覆盖。

```bash
python check_attentivefp_environment.py
python check_benchmark_data_layout.py --data-root "${DATA_ROOT}" --dataset-config dataset_config.csv
```

数据检查预期 1,152 个 CSV；每类任务 192 个 run，strict_perimeter 只有 seed_0。

## Slurm 测试

```bash
RUN_RANGE=0 MAX_CONCURRENT=1 EPOCHS=3 PATIENCE=2 CONDA_ENV=AttentiveFP bash submit_attentivefp_classification_slurm.sh
RUN_RANGE=0 MAX_CONCURRENT=1 EPOCHS=3 PATIENCE=2 CONDA_ENV=AttentiveFP bash submit_attentivefp_regression_slurm.sh
```

## Slurm 正式提交

```bash
MAX_CONCURRENT=4 CONDA_ENV=AttentiveFP bash submit_attentivefp_classification_slurm.sh
MAX_CONCURRENT=4 CONDA_ENV=AttentiveFP bash submit_attentivefp_regression_slurm.sh
```

默认 `CANCEL_ARRAY_ON_FAILURE=1`，任一 array task 报错时取消整个数组；不需要此策略时设为0。

## 无 Slurm

```bash
GPUS="1" MAX_JOBS_PER_GPU=1 CONDA_ENV=AttentiveFP bash run_attentivefp_classification_no_slurm.sh
GPUS="2" MAX_JOBS_PER_GPU=1 CONDA_ENV=AttentiveFP bash run_attentivefp_regression_no_slurm.sh
```

`MAX_JOBS_PER_GPU` 限制每张卡同时运行的任务数；任一任务失败时启动器返回非零并跳过自动汇总。若通过 `RUN_RANGE` 仅运行部分任务，即使 `POSTPROCESS=1` 也不会自动生成完整运行汇总；可在检查现有结果和运行范围后手动执行后处理。完整任务成功完成后才会按 `POSTPROCESS` 设置自动汇总。

## 每个 run 输出

`metrics_full.csv`、三套 predictions、`diagnostics_metrics.csv`、`diagnostic_subgroup_metrics.csv`、三套训练曲线文件、三个 JSON、`best_model.pth`、`run.log`。train/validation 只输出逐分子预测和整体指标；test 额外计算 NN 与 Activity Cliff。训练曲线逐 epoch 实时写盘。

## 后处理

```bash
bash postprocess_attentivefp_classification_outputs.sh
bash postprocess_attentivefp_regression_outputs.sh
```

后处理生成汇总、mean±SD、OOD/perimeter、test-test Activity Cliff pair 分析和输出合规报告。
