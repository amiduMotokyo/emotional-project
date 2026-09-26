# 第六、七轮论文图表与审计

正文编辑源：[论文片段](../../第六七轮优化局限与失败经验论文片段.md)。Word由同一Markdown导出。

- fig1_development：两轮stop的三种子均值及样本SD。
- fig2_precision：第七轮valid的FP32、int8、最终校准准确率。
- fig3_seed_calibration：配对种子差异及校准折外/全拟合口径。
- fig4_confusion：固定交付种子的728条valid混淆矩阵。

各图提供300dpi PNG和SVG。`plot_data.json`保存读取的结果快照；`plot_values.csv`提供逐种子准确率；`source_manifest.json`记录原产物SHA256、快照SHA256和风格参考；`render_manifest.json`记录图片哈希。绘图前对已登记在训练state中的原产物校验哈希，不重新训练、推理或选择模型。

风格沿用B的`plot_q3.py`配色及混淆矩阵“计数＋行比例”标注，参考A现有问题1章节的组织方式，并衔接C第五轮点估计与误差线规范。旧A训练绘图脚本目前不存在，没有把它列为本次已读取参考。

生成：`python C/scripts/plot_round67_paper.py`；Word导出与分页检查：`python C/scripts/export_round67_paper.py --check`（后者使用myenv中python-docx、pywin32及本机Word）。`word_layout_check.json`记录实际分页、图注同页和表格不跨页检查。预览PDF及逐页图在`C/outputs/round67_paper_preview/`。
