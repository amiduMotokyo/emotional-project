# 第四轮论文配图与复现

对应[论文编辑源](../../第四轮结构优化实验论文草稿.md)，Word同名导出。五幅图均有300 dpi PNG及SVG版本，采用A/B现有论文图的中文字体、蓝绿配色和混淆矩阵表示。

- `plot_data.json`：来自冻结实验产物的绘图快照，包含三种子指标、训练曲线、64条件结果及累计混淆矩阵。
- `source_manifest.json`：原始输入、数据快照及绘图风格参考文件的SHA-256。
- `plot_values.csv`：六种结构的验证指标原值。
- `render_manifest.json`：绘图脚本及五幅图两种格式的SHA-256。
- `word_layout_check.json`：Word分页、图题同页、完整表格及公式检查结果。

在项目根目录、具备numpy、matplotlib、python-docx与pywin32的Python环境中执行：

```powershell
python C/scripts/plot_q2_round4_paper.py
python C/scripts/export_round4_paper.py
python C/scripts/check_round4_paper.py
```

上述命令已在本机myenv环境执行；最后一步需要桌面版Microsoft Word。默认使用快照，添加`--snapshot`将重新读取并校验冻结实验输入。命令不训练、不运行模型推理、不重新选型。临时PDF及预览位于`C/outputs/q2_round4_paper_preview/`。

口径：所有汇总指标为三个独立模型均值，非集成。图5累计三个模型对同一727条样本的预测，共2181次计数，不能视为独立样本数。历史test仅作本轮锁定后复评。
