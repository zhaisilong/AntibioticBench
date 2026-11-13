# v9 修复说明

修复 `set -e` 与可选参数判断组合导致正式任务在 `FORCE=0` 时立即退出的问题。

旧代码：

```bash
[[ "${FORCE}" == "1" ]] && FORCE_ARG+=(--force)
```

当 `FORCE=0` 时，`[[ ... ]]` 返回状态 1。在 `set -e` 下，这会被当作错误，
导致工作流在 build_run_plan 阶段停止。

v9 改为显式 if：

```bash
if [[ "${FORCE}" == "1" ]]; then
    FORCE_ARG+=(--force)
fi
```

`RESUME` 判断也同步改为显式 if。分类和回归脚本均已修复。
训练参数、数据路径、早停指标和汇总逻辑没有改变。
