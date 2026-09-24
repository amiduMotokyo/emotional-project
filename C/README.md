# C：缺失鲁棒训练、优化与评估

C维护 M5连续缺失训练策略、M7指标与实验记录，并负责最终提交包。问题二输入层缺失一致性修订已训练，在附件2 train/valid/test 上完成训练、选型和留出测试评估，并生成附件3预测；超参数范围仍限于文中列出的候选设置。

- `src/missingness.py`：训练与验证使用的连续局部缺失掩码。
- `src/metrics.py`：Accuracy、Macro-F1、MAE和Pearson统一评估。
- `src/training.py`：基线与鲁棒训练流程。
- `configs/q2_robust.json`：种子、优化器、掩码比例和模型选择指标。
- `scripts/run_q2.py`：训练、验证、缺失率网格和附件3推理。
- `scripts/analyze_q2.py`：逐样本验证预测、错误分析和图表。
- `scripts/run_q2_corrected.py`：先替换缺失词元再编码，训练三种候选模型并评估全缺失网格。
- `scripts/analyze_q2_corrected.py`：按最终交付解码输出验证图表与误差归因。
- `scripts/evaluate_q2_attachment2_test.py`：核对 `label.xlsx` 与对齐特征，评估附件2的727条test及63种受控缺失场景；test不参与训练或选型。
- `scripts/infer_q2_corrected.py`：使用选定模型生成附件3全量预测。
- [最终修订实验记录](../docs/C/experiments/q2_input_consistent_experiments.md)：含附件2独立test评估和缺失规律；[历史协议记录](../docs/C/experiments/q2_robust_experiments.md)。
- `outputs/submission/question2_corrected.zip`：问题二模型、复现代码、附件2测试/验证分析和附件3预测。

运行流程和依赖见实验文档与提交包 README。原始数据仍从项目约定的共享 `data/` 目录读取，不复制进代码仓库。
