# 第二题数据接口说明

状态：第二题路径已实现；附件4和未对齐版尚未覆盖。
对应模块：B/M1。
数据约束：只读取赛题附件，不将附件3或4用于训练、验证或阈值选择。

## 输入与字段

当前入口固定使用附件2 `aligned_50.pkl` 的 `train`、`valid` 两个划分，以及附件3“对齐版本”目录下的30个 `.pkl` 文件。附件2每条样本以 `id` 追踪；训练字段为 `text_bert`、`audio`、`vision`、`classification_labels` 和 `regression_labels`。附件3每个文件取顶层 `test` 中的 `text_bert`、`audio` 和 `vision`。数据接口不改变原始文件。

附件2训练/验证张量按批次组织为 `text_bert: (N,3,50)`、`audio: (N,50,74)`、`vision: (N,50,35)`；文本的三路值为词元编号、注意力掩码和分段编号。附件3单文件通常有大小为1的批次维。文本词元由冻结的 `sentence-transformers/all-MiniLM-L6-v2` 编码，得到 `(N,50,384)` 的序列表示；训练和推理使用相同词元输入接口。最终压缩包包含 int8 ONNX 推理编码器。

`assemble_sample` 输出 `text`、`audio`、`vision` 以及 `tmask`、`amask`、`vmask`。极性编号为0负向、1中性、2正向；强度保持赛题给出的 `[-3,3]` 标度。样本编号沿用附件3文件名。

## 掩码和标准化

- `tmask` 要求注意力掩码有效，并排除 `[CLS]`、`[SEP]` 和 `[UNK]`。
- `amask`、`vmask` 要求对应位置同时有效于文本注意力掩码，且该位置模态向量非全零。
- 音频、视觉均值和标准差仅在训练集有效位置估计；标准差下限为 `1e-4`，标准化值裁剪至 `[-5,5]`。无效位置在标准化后仍置零。
- 全零特征只表示该位置不可用；接口不据此推断它是原始填充还是人为缺失。人为连续缺失由 C 的掩码生成器独立模拟。

## 当前覆盖范围与交接

`B/src/data.py` 被 C 的第二题训练入口调用，是共享数据实现的唯一来源。已核对训练/验证和附件3对齐版的字段与形状；附件4文件读取、未对齐版长度处理以及证据时间坐标尚未实现。因而当前 M1 为部分完成，不能用于宣称问题3已可推理。

最小预处理示例：

```bash
python B/scripts/prepare_q2_cache.py \
  --data-root /path/to/E题数据 \
  --text-model /path/to/all-MiniLM-L6-v2 \
  --cache /path/to/cache \
  --device cuda
```

预期得到 `train.npz`、`valid.npz` 和30个 `q2_*.npz`。环境依赖见 `C/outputs/submission/question2_submission.zip` 中的 `requirements.txt`。缓存属于可再生成的大文件，不提交到 Git。

## 2026-09-24 输入一致性修订

复核发现动态量化ONNX编码器对同一词元输入采用不同批次大小时，输出存在可测差异。附件3推理按单条样本调用，因此问题二修订流程对训练、验证、缺失视图和专项推理统一按单条样本编码。`B/scripts/reencode_q2_cache.py` 用随包ONNX模型重建缓存，避免复用与部署编码器或批次口径不同的文本向量。

文本连续缺失在编码前将新增缺失位置的词元替换为 `[UNK]`，保留原始attention mask，并重新编码；音频和视觉使用独立观测掩码。附件2中原始有效序列位置从掩码确定，连续区间保留原位置索引，不以有效点数量压缩重编号。该修订由 `C/src/q2_protocol.py` 和 `C/src/missingness.py` 实现，正式实验结果见 [输入层缺失一致性实验](../../C/experiments/q2_input_consistent_experiments.md)。
