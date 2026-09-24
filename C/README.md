# C：缺失鲁棒训练、优化与评估

C维护 M5连续缺失训练策略、M7指标与实验记录，并负责最终提交包。目前第二题核心实验已跑通；参数搜索范围和跨模块归档仍需补齐。

- `src/missingness.py`：训练与验证使用的连续局部缺失掩码。
- `src/metrics.py`：Accuracy、Macro-F1、MAE和Pearson统一评估。
- `src/training.py`：基线与鲁棒训练流程。
- `configs/q2_robust.json`：种子、优化器、掩码比例和模型选择指标。
- `scripts/run_q2.py`：训练、验证、缺失率网格和附件3推理。
- `scripts/analyze_q2.py`：逐样本验证预测、错误分析和图表。
- [第二题鲁棒实验记录](../docs/C/experiments/q2_robust_experiments.md)。
- `outputs/submission/question2_submission.zip`：第二题独立提交包，不含全题论文或问题1/3结果。

运行流程和依赖见根目录下的实验文档与提交包 README。原始数据仍从项目约定的共享 `data/` 目录读取，不复制进代码仓库。
