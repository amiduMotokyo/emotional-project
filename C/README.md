# C：智能优化、参数优化、特征选择与评估

负责M5中面向局部缺失鲁棒性的智能优化算法、参数优化与特征选择，维护M7统一评估，并承担提交包整理。外层优化调用B的共享模型进行训练与验证；不以重新设计融合网络为主要任务。2026-09-24按用户指令调整，自适应缺失采样方向暂停。

队友的问题二输入层缺失一致性修订已在附件2 train/valid/test 上完成训练、选型与留出测试评估，并生成附件3预测。C另已完成第一轮DE调参与BPSO特征选择、六组重复训练及独立测试；两轮实验按各自配置记录，C本轮最终保留E0默认参数模型。

## 已接入实现与实验

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
- 队友记录中的`outputs/submission/question2_corrected.zip`包含问题二模型、复现代码、附件2测试/验证分析和附件3预测；它与历史`question2_submission.zip`不是同一版本，实际产物及服务器位置见实验记录。

## 智能优化方向与归档

- `src/`：智能优化搜索、特征选择、缺失评估场景与统一评估实现。
- `configs/`：缺失类型、位置、时长、随机种子和实验参数。
- `scripts/`：训练、消融、评估与提交包生成入口。
- `outputs/`：实验日志、权重、附件3预测和 `submission/` 提交包。
- [个人文档](../docs/C/README.md)：需求、创新调研、实验说明和论文。
- [当前实验设计](../docs/C/research/2026-09-24_智能优化参数与特征选择实验设计.md)：第一轮固定预算DE调参、BPSO声学/视觉通道预选、六组重复训练及独立测试已完成；按验证规则选择E0，未取得全面性能提升。见[论文草稿](../docs/C/paper/第一轮智能优化实验论文草稿.md)及[Word](../docs/C/paper/第一轮智能优化实验论文草稿.docx)。
- [对接与运行说明](../docs/C/experiments/q2_optimization_integration.md)：统一B修订协议；合成集成检查和真实小样本冒烟通过，正式运行先prepare、preflight、pilot，再按预算启动搜索。
- `scripts/run_q2_optimization.py`、`configs/q2_optimization.json`：分阶段运行与搜索配置；`scripts/check_q2_optimization.py`：工程验证。
- `scripts/report_q2_optimization.py`：完整评估后导出六组对照表、逐类F1、混淆矩阵、强度散点图、缺失曲线和搜索过程图；raw/coherent及validation/test分别归档，不能混用B历史分数。
- 本机正式运行使用用户指定的`myenv`（CUDA版PyTorch、RTX 4070 Laptop 8GB）；`scripts/run_local_q2_pipeline.py`串行执行各阶段。当前运行目录为`outputs/q2_optimization_local_myenv_v1/`，实时状态见其中`pipeline_status.json`，配置及验证记录见对接文档第8节。
- [文献阅读笔记](../docs/C/research/2026-09-24_智能优化文献阅读笔记.md)：六篇全文阅读结论、采用内容及复现局限。
- [当前方向文献](../docs/C/research/papers/intelligent_optimization/README.md)：六篇本地PDF与来源。

复用B的数据接口和融合基线，不复制另一套基础实现。评估口径供B共同使用。运行流程和依赖见实验文档与提交包 README。原始数据仍从项目约定的共享 `data/` 目录读取，不复制进代码仓库。
