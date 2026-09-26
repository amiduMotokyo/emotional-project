# 第三问第六轮MiniLM候选交付接口

状态：已实现并完成20条附件4导出；候选不替换原总提交包。训练与结果见[实验记录](../experiments/第六轮端到端MiniLM训练与第三问交付结果.md)。模型及解释共享实现归B，视频证据定位归A。

## 输入与运行

独立包：`C/outputs/submission/q3_round6_minilm_v2.zip`。解压目录执行：

```sh
python C/scripts/infer_q3_minilm_round6.py --package . --data-root /path/to/data --output results_new --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe
```

`data-root`保留原赛题附件4目录结构；ID为文件名01—20，与pickle中id一致，视频同名。单样本`text_bert`为3×50，音频50×74，视觉50×35；音视频float32，词元int64，掩码bool。训练和推理均采用MiniLM-L6-v2词表，推理编码器为已微调int8 ONNX、CPU单样本batch1，不能改为批量编码后声称精度相同。

音视频标准化使用完整train拟合的mean/std，截断至[-5,5]；无效位置为零。文本有效位置排除padding/UNK/CLS/SEP；音视频原始全零行无效，不能当成人工缺失。输入干预保持原50位置，文本替换UNK后重新编码，音视频置零，并更新相应掩码。

部署入口`B.src.q3_minilm_explain.MiniLMExplanation(package)`兼容B的`prepare/predict/explain`接口。`Q3Predictor`新增可选`ort_threads`、读取Fusion的`model_kwargs`，旧包缺失该字段时仍用默认Fusion。附件4读取器兼容NumPy2保存的数组pickle在NumPy1环境读取；原始附件只读。媒体子进程明确按UTF-8解码并容忍不可解码日志，避免Windows中文路径的GBK线程异常。

## 输出与语义

- CSV：`sample_id`、`predicted_polarity`、`predicted_intensity`、三类校准概率、`raw_prob_*`、原始强度、各模态gate/删除效应/绝对占比/Shapley、主要模态及证据索引/片段/时段/帧号/映射状态。
- 分类顺序Negative/Neutral/Positive，对应0/1/2；原始及最终强度范围[-3,3]，服务值按最终类别设置符号，中性为0。偏置已在推理中应用，使用者不能重复加偏置。
- JSON保留全部50位置效应、1—3位置联合窗口、8个模态子集分数和Shapley加和误差。旧字段名`biased_logits`、`delta_logit`在本候选解释中实际表示校准log概率及其下降，固定原预测类别；原模型logit另存`model_biased_logits`。
- gate是模型内部权重，不是贡献。删除效应与Shapley分别定义；各自绝对占比的分母为同种效应三模态绝对值之和，分母为零时全为0。没有正删除支持时主要模态为空，CSV为undetermined。
- 文本证据为零基字符起点和右开终点；时间单位秒、原始帧号从0开始。音视频通过特征匹配和名义采样率映射，保留歧义/时长核查状态，不将其称为精确强制对齐。13号样本没有有效视觉位置，因此不制造视觉证据或预览。

附件4没有标签，不输出准确率。当前候选验证ACC62.77%，未超过队友归档63.74%；应作为可复现研究候选保存。组件小于50MB不代表全队合包自动满足50MB要求。
