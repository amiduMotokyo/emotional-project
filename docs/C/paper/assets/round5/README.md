# 第五轮论文配图与复现

对应[Markdown编辑源](../../第五轮独立专家与决策融合实验论文草稿.md)及同名Word导出。五幅图均提供300 dpi PNG和SVG，沿用A/B论文及第四轮的配图风格。

`plot_data.json`为经完成标记哈希验证的实验快照；`source_manifest.json`记录实验输入、快照及风格参考文件哈希；`plot_values.csv`保存候选指标；`render_manifest.json`记录绘图脚本和图片哈希；`word_layout_check.json`记录分页、图题同页、表格完整性与原生公式检查。

在项目根目录执行（本机使用myenv Python，Word检查需要桌面版Microsoft Word）：

```powershell
python C/scripts/plot_q2_round5_paper.py
python C/scripts/export_round5_paper.py
python C/scripts/check_round5_paper.py
```

绘图默认从快照读取；添加`--snapshot`重新读取并校验实验输入。上述流程不训练、不推理、不重新选型。临时PDF预览位于`C/outputs/q2_round5_paper_preview/`。

所有指标按三个种子分别计算再取均值，未跨种子集成。图5为三个种子对同一727条样本的累计计数，2181次预测不是2181个独立样本。图1诊断上限使用真实标签识别正确专家，不是可部署结果。
