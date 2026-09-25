# C：智能优化、参数优化、特征选择与评估

负责M5中面向局部缺失鲁棒性的智能优化算法、参数优化与特征选择，维护M7统一评估，并承担提交包整理。外层优化调用B的共享模型进行训练与验证；不以重新设计融合网络为主要任务。2026-09-24按用户指令调整，自适应缺失采样方向暂停。

队友的问题二输入层缺失一致性修订已在附件2 train/valid/test 上完成训练、选型与留出测试评估，并生成附件3预测。C另已完成第一轮DE调参与BPSO特征选择、六组重复训练及独立测试；两轮实验按各自配置记录，C本轮最终保留E0默认参数模型。

## 已接入实现与实验

- 第五轮独立专家：`src/q2_experts.py`、`configs/q2_round5_experts.json`、`scripts/run_q2_round5.py`；[固定方案](../docs/C/experiments/第五轮独立专家与决策融合实验方案.md)。新增音视频专家，按视频分组折外训练决策融合器。`scripts/check_q2_round5.py`执行机制检查，`--audit`审计完成结果；`scripts/report_q2_round5.py`生成结果及小型归档。输出在`outputs/q2_round5_experts_v1/`，运行入口支持按完成标记断点恢复。
- 第五轮[结果与限制](../docs/C/experiments/第五轮独立专家与决策融合实验结果.md)：21个新专家、378 epoch、3个门控及64条件复评完成，审计和恢复检查通过。获选学习门控的干净test准确率均值65.61%（基线65.25%），64条件平均63.42%（基线62.94%）；中性F1从0.3941降至0.3770。属于小幅准确率改善，非全面提升。未触发更换文本编码器分支，未改写提交模型。推理需同种子的第四轮`TextOnly`、第五轮`AudioVisualExpert`及`DecisionGate`：概率`[...,2,3]`、强度`[...,2]`、相对干净有效位置的可用率`[...,3]`→融合概率、强度、专家权重；专家顺序为文本、音视频。全部缺失返回均匀概率和强度0，门控权重不作因果贡献解释。

- 第四轮结构对照：`src/q2_structural.py`、`configs/q2_round4_structure.json`、`scripts/run_q2_round4.py`；方案见[第四轮实验](../docs/C/experiments/第四轮结构优化实验方案.md)。`scripts/check_q2_round4.py`检查缺失掩码、概率分解、梯度与容量匹配；`--audit`核对完成产物。`scripts/report_q2_round4.py`归档逐种子及逐类指标。
- 第四轮[结果](../docs/C/experiments/第四轮结构优化实验结果.md)：18模型和64条件复评完成，审计通过。容量对照Fusion(dim=91)获选，三种子独立模型的干净test均值65.25%，对同轮基线提升2.15个百分点；序列交互模型未胜出。候选权重在`outputs/q2_round4_structure_v1/models/wide_control_<seed>/best.pt`，使用`q2_structural.build(checkpoint['recipe']['architecture'])`后加载`state_dict`；也可直接用B的Fusion(dim=91)。未改写正式提交包。

- 第三轮临时论文已归档：[Markdown](../docs/C/paper/第三轮准确率优化实验论文草稿.md) · [Word](../docs/C/paper/第三轮准确率优化实验论文草稿.docx)。`scripts/plot_q2_round3_paper.py`从审计产物绘制五图；`scripts/export_round3_paper.py`导出Word；`scripts/check_round3_paper.py`核对分页、图注、表格与公式。

- 第三轮准确率优化：`scripts/run_q2_round3.py`、`configs/q2_round3_accuracy.json`，使用`scripts/resume_q2_round3.py`入口修复NumPy布尔值序列化后运行；见[第三轮方案](../docs/C/experiments/第三轮准确率优化实验.md)与[结果](../docs/C/experiments/第三轮准确率优化实验结果.md)。21模型、378 epoch已完成，未取得符合退化限制的新方法提升。`scripts/check_q2_round3.py --audit`审计完成产物，`scripts/report_q2_round3.py`生成种子波动、配对区间和结果归档。

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

- 第二轮完整集成实验：`scripts/run_q2_round2.py`串行执行准备、两库训练、预测缓存、R0—R7三折搜索、锁定和64场景复评；`scripts/check_q2_round2.py`做数值与预算检查，`scripts/report_q2_round2.py`归档结果。启动、恢复与数据边界见[第二轮指南](../docs/C/experiments/第二轮优化实验启动指南.md)。输出为`outputs/q2_round2_ensemble_v1/`，不覆盖第一轮产物。
- 第二轮已完成并通过`scripts/audit_q2_round2.py`实际产物审计，见[完整结果](../docs/C/experiments/第二轮完整优化实验结果.md)。内部验证锁定R6，test的clean ACC下降且64条件平均S仍低于等权集成；结果保留正负两面，未改写正式提交包。主流程约16.8分钟，其中训练约5.5分钟。
- `src/ensemble_search.py`：标签只供fit使用，predict只接收权重/偏置及`[M,S,N,3]`概率、`[M,S,N]`回归缓存；三搜索种子先各自预测再平均。`src/q2_group_metadata.py`只提取原始train/valid片段ID用于视频分组，不还原数值特征或标签数组。
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
