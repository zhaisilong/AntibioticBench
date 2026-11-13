# AttentiveFP 脚本包检查报告

以下结果是原始脚本包的历史记录。此后代码中的路径、隐私处理和启动器调度已经修改；原报告不能证明当前版本已通过端到端训练。当前修改仅做了语法与模拟启动检查，未重新执行 GPU 训练或真实 Slurm 作业。

原始脚本包记录了以下检查：

1. 所有 Python 文件通过 `py_compile` 语法检查。
2. 所有 Bash/Slurm 文件通过 `bash -n` 语法检查。
3. 分类与回归 run plan 均为 192 个 run。
4. 每类任务的 `strict_perimeter` 仅生成 12 个 `seed_0` run，不生成 seed 1–4。
5. 使用 PyTorch Geometric 2.7.0 实例化固定 AttentiveFP：可训练参数量为 897,201，前向传播输出形状正确。
6. 使用真实 PyG AttentiveFP 在小型合成分子数据上完成分类和回归端到端测试：训练、best checkpoint、三套逐分子预测、整体指标、test-only NN/Activity Cliff、汇总及合规检查均成功。
7. 分类和回归输出合规检查结果均为 `overall_compliant=true`。

服务器正式运行前仍应执行：

```bash
python check_attentivefp_environment.py
python check_benchmark_data_layout.py \
  --data-root "${DATA_ROOT}" \
  --dataset-config dataset_config.csv
```

服务器的 CUDA、PyTorch、PyG 和 Slurm 配置与本地检查环境可能不同，因此建议先用 `RUN_RANGE=0 EPOCHS=3 PATIENCE=2` 分别测试分类和回归。
