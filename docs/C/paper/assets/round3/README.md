# 第三轮论文配图与数据

由 `C/scripts/plot_q2_round3_paper.py` 从已审计的第三轮保存结果生成；不重新训练、推理或选择模型。默认从本目录 `plot_data.json` 重绘，显式 `--snapshot` 可重新核对原始完成记录并更新快照。

| 文件前缀 | 内容 |
|---|---|
| fig1_stop_ablation | 五组消融的三种子均值与样本标准差 |
| fig2_validation_tradeoff | 内部验证准确率、干净及缺失宏F1与约束 |
| fig3_learning_curves | 18 epoch训练目标和stop准确率，均值±标准差 |
| fig4_missing_rates | 锁定基线7种模态组合的缺失比例曲线 |
| fig5_test_confusion | 锁定基线干净test的计数及行百分比 |

每图均提供300dpi PNG和文字转路径的SVG。图中局部坐标轴使用点图或曲线，不使用截断基线的柱图。学习曲线阴影为三个种子的样本标准差，不是置信区间；不同配方的训练损失系数不同，不能直接比较绝对损失大小。

`plot_data.json`保留完整作图数值；`plot_values.csv`便于查阅18种验证候选；`source_manifest.json`记录结果源文件、快照及A/B/C作图参考代码哈希；`render_manifest.json`记录生成脚本和图片哈希。`word_layout_check.json`记录Word排版检查。

正文编辑源为上两级的 `第三轮准确率优化实验论文草稿.md`，Word导出入口为 `C/scripts/export_round3_paper.py`。Word使用内嵌PNG，三条展示公式为原生可编辑OMML。
