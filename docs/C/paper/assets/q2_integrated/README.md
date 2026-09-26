# 问题二统合稿的复现与编辑说明

正文唯一编辑源：[问题二.md](../../../../论文稿/问题二.md)。Word同目录，原问题一未修改。项目文件来源、编辑流程及合稿注意事项保留在本文件，不进入学术正文。

## 最终模型与历史证据

- 最终预测器S₀来自B/q2_single_model_final/q2_single_model_final：冻结BERT、32维A4、seed43、第3轮、偏置[0.375,0.275,0]，用于第三问入口。
- F₀、B₀及七轮、66模型三层消融均是历史MiniLM系列证据，不转写为S₀的消融或成绩。B₀的63条件结果仅作历史鲁棒性对照。
- S₀分类和回归独立输出，不强制符号一致。valid有46条非中性符号相反、157条中性对应非零回归；专项有2条非中性符号相反和7条中性非零，正文明确区别。
- S₀的valid和历史test指标由提供的逐样本CSV复算，与final_metrics.json核对。没有重训，没有重新运行测试推理，也没有改写B目录模型或输入。

## 文件对应

- 表5-24：问题二_最终第三问入口模型_附件3预测.csv，逐字节复制S₀专项输出，30行、10字段；表5-27为字段说明。
- 表5-28：问题二_历史MiniLM基线63种缺失条件完整结果.csv。原含“最终基线”的文件名已改为历史身份，防止误认。
- 表5-30至5-32：问题二_三层消融66次训练结果.csv，固定10轮的MiniLM消融。
- 表5-33至5-34：问题二_最终S0缺失敏感性144行.csv。比例网格60行、位置网格84行，正文分别对三个扰动种子取均值；两网格有交叠，不将144行当作独立条件数。

## 构建和检查

运行python C/scripts/build_q2_integrated_paper.py依次生成历史图表、supplement_q2_paper.py的补充证据、report_q2_three_layer_ablation.py的66模型结果和integrate_q2_single_final.py的S₀方法与结果，最后输出Word。自动生成章节的措辞改动需要同步相应脚本；其他正文直接编辑Markdown。

公式通过latex2mathml及Office的MML2OMML.XSL转换为可编辑OMML。使用安装pywin32的环境运行构建脚本--check进行Word分页检查，预览PDF保存于C/outputs/q2_integrated_paper_review。所有表格复制问题一的三线表、单元格字体与表题样式。

主要证据清单：source_manifest.json、supplement_manifest.json、three_layer_manifest.json及single_final_source_manifest.json。S₀逐样本复算记录在single_final_content_check.json，最新Word布局与文件哈希见word_layout_check.json。当前正文24页，9图、25个表格对象（含续表）、21个公式。

## 合稿编号

正文为5.3，图5-13至5-21、表5-13至5-34（28与34含续表），公式5.3-1至5.3-21。原问题一末尾“5.3.1 文本时间对齐特征构建”仍属于问题一，合稿时建议改为5.2.7；本次保留原件。

未修改旧提交zip，也未提交或推送Git。模型文件较小不等于完整冻结BERT依赖已纳入提交容量，最终打包需单独核查。
