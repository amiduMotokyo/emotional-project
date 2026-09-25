# 第三轮结果来源

这些小型记录由 `C/scripts/report_q2_round3.py` 从已审计的 `C/outputs/q2_round3_accuracy_v1/` 复制。`source_manifest.json` 保存源文件 SHA-256。

- `manifest.json`：固定配置、训练源码、输入缓存与预训练权重哈希。
- `preparation.json`、`preflight.json`：实际增强比例、输入掩码和微调梯度预检查。
- `recipe.json`：仅使用 stop 选择训练配方的完整指标。
- `selection.json`：18种候选的内部分组验证与退化检查。
- `lock.json`：测试前确定的模型及偏置；本轮保留 baseline_mean。
- `test_summary.json`：64条件锁定后复评；selected和baseline实际为同一方法。
- `analysis.json`：种子波动、训练曲线摘要、耗时及配对区间。
- `audit.json`：完成产物与数值审计结果。
- `serialization_adapter.json`：不改变数值的布尔值序列化修复及源码哈希。

完整逐样本预测、21个模型权重、378个epoch历史仍保存在原 outputs 中。此目录没有复制训练数据或模型。test是历史已查看过的复评数据，不能称为独立新盲测。
