# 第二轮临时论文片段：图表与结果快照

对应[第二轮论文片段](../../第二轮集成优化实验论文草稿.md)。原运行结果保留在`C/outputs/q2_round2_ensemble_v1/`；本目录保存可携带的聚合数据和正式配图，不复制原始数据或模型。

| 临时图号 | 文件主名 | 内容 |
|---|---|---|
| C2-1 | fig1_oof_comparison | 两库折外S、相对R0的配对bootstrap区间 |
| C2-2 | fig2_test_tradeoff | 64条件平均S与clean准确率分别比较 |
| C2-3 | fig3_missing_rates | 七种缺失组合，三个位置平均的Macro-F1 |
| C2-4 | fig4_test_confusion | 相同727条clean test的计数及行百分比 |
| C2-5 | fig5_ensemble_weights | 锁定R6在全valid重拟合后的三搜索种子平均权重 |

每图同时提供300 dpi PNG和SVG。SVG文字转路径以避免跨机器缺少中文字体，数值和标签应修改作图脚本或数据后重新生成，不直接手改曲线。

- `plot_data.json`：完整作图聚合数据，包括混淆矩阵和置信区间；不含逐样本标签。
- `plot_values.csv`：折外S、test汇总指标、全部缺失曲线与最终权重，便于表格核对。
- `source_manifest.json`：原运行来源文件哈希、数据快照哈希、队友图表参考来源。
- `render_manifest.json`：作图脚本、依赖版本及图片哈希。

作图入口：`C/scripts/plot_q2_round2_paper.py`。默认只依赖本目录快照；`--snapshot`会先核验原运行结果，再重新提取同一结果的聚合数据。没有重新训练、重拟合、重新推理或按test改变选择。

参考A的`A/problem2_training_v2/plot_weighted_results.py`与实际对比图，B的`B/scripts/plot_q3.py`及`docs/B/paper/q3_explain_model.md`。中文字体沿用A的Microsoft YaHei/SimHei设置，白底、浅网格、蓝绿主色及橙色强调延续B图表；本轮数值只来自C自己的已锁定运行。
