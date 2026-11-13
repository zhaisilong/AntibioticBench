# KPGT AntibioticBench：无 Slurm 服务器运行说明

先按 [KPGT 路径配置](README_KPGT_AntibioticBench.md) 设置 `DATA_ROOT`、`KPGT_ROOT` 和所需权重，然后从仓库根目录进入：

```bash
cd code/KPGT
```

建议在 `tmux` 中运行。tmux 已经能在 SSH 断线后保持任务，无需再套 `nohup`。

## 分类测试

```bash
RUN_MODE=test \
GPUS="1" \
MAX_JOBS_PER_GPU=1 \
EPOCHS=3 \
PATIENCE=2 \
POSTPROCESS=0 \
RESUME=0 \
SHOW_RUN_OUTPUT=1 \
bash run_kpgt_classification_no_slurm.sh
```

## 回归测试

```bash
RUN_MODE=test \
GPUS="0" \
MAX_JOBS_PER_GPU=1 \
EPOCHS=3 \
PATIENCE=2 \
POSTPROCESS=0 \
RESUME=0 \
SHOW_RUN_OUTPUT=1 \
bash run_kpgt_regression_no_slurm.sh
```

测试产生的 `run index 0` 结果不要直接当作正式结果。正式运行前可以删除测试输出，或者设置 `FORCE=1` 重跑该 run。

## 正式分类

```bash
RUN_MODE=batch \
GPUS="1" \
MAX_JOBS_PER_GPU=1 \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
STOP_ON_ERROR=1 \
bash run_kpgt_classification_no_slurm.sh
```

## 正式回归

```bash
RUN_MODE=batch \
GPUS="0" \
MAX_JOBS_PER_GPU=1 \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
STOP_ON_ERROR=1 \
bash run_kpgt_regression_no_slurm.sh
```

有两张空闲 GPU 时，可设为 `GPUS="0 1"`。脚本会在两张 GPU 上分别顺序运行，整体最多同时训练两个 run。除非显存非常充足，不建议把 `MAX_JOBS_PER_GPU` 设置为 2 或更大。

## tmux 示例

```bash
tmux new -s kpgt_cls
cd code/KPGT
RUN_MODE=batch GPUS="1" MAX_JOBS_PER_GPU=1 POSTPROCESS=1 RESUME=1 \
  bash run_kpgt_classification_no_slurm.sh
```

按 `Ctrl+B`，松开后按 `D`，即可退出而不中断任务。重新进入：

```bash
tmux attach -t kpgt_cls
```

另开回归会话：

```bash
tmux new -s kpgt_reg
cd code/KPGT
RUN_MODE=batch GPUS="0" MAX_JOBS_PER_GPU=1 POSTPROCESS=1 RESUME=1 \
  bash run_kpgt_regression_no_slurm.sh
```

## 日志和恢复

分类日志：

```text
.../output_classification/no_slurm_logs/<RUN_TAG>/run_000_gpu1.log
```

回归日志：

```text
.../output_regression/no_slurm_logs/<RUN_TAG>/run_000_gpu0.log
```

每个 run 的训练目录仍会保存 `train_log.csv`、`training_curve.csv`、预测、指标及 checkpoint。`RESUME=1` 会跳过已有 `_SUCCESS` 的 run，因此中断后重新执行同一正式命令即可续跑。

查看当前 GPU：

```bash
watch -n 2 nvidia-smi
```

查看某个 run 的训练日志：

```bash
tail -f "${OUT_ROOT}/no_slurm_logs/<RUN_TAG>/run_000_gpu1.log"
```
