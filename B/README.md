# B：数据接口、融合与解释

B维护 M1 数据接口、M4融合基线和M6解释实现。当前已接入第二题对齐版的数据路径和融合模型；附件4读取及M6解释仍待完成。

- `src/data.py`：附件2 train/valid/test与附件3对齐版的文本编码、有效掩码、音视频标准化和批次接口。
- `src/fusion.py`：掩码注意力池化、动态门控、极性与强度输出。
- `configs/q2_fusion.json`：维度、标签映射和模型参数。
- `scripts/prepare_q2_cache.py`：建立附件2 train/valid/test 与附件3缓存。
- `scripts/reencode_q2_cache.py`：用随包ONNX编码器逐条重建文本缓存，与附件3推理保持一致。
- `scripts/infer_attachment3.py`：用压缩推理包生成附件3预测。
- `src/literature.py`、`scripts/run_literature_models.py`：LMF、MISA、MulT及MAG-MiniLM适配版的统一比较实现。
- [数据接口说明](../docs/B/requirements/q2_data_interface.md)、[基线实验记录](../docs/B/experiments/q2_fusion_baseline.md)、[文献方法对比](../docs/B/experiments/q2_literature_comparison.md)、[论文方法稿](../docs/B/paper/q2_fusion_model.md)。

预处理命令示例：

```bash
python B/scripts/prepare_q2_cache.py \
  --data-root /path/to/E题数据 \
  --text-model /path/to/all-MiniLM-L6-v2 \
  --cache /path/to/cache --device cuda
```

本次已实现并验证附件2对齐版 train/valid/test 与附件3对齐版。附件4统一读取、证据时间定位和局部解释属于后续M1/M6工作；当前门控权重只描述模态级融合，不作为附件4局部因果解释。

修订后的问题二流程发现int8 ONNX编码输出随批次大小变化，因此附件2 train/valid/test 缓存和附件3逐条推理统一采用单样本调用。缺失文本在送入编码器前替换为 `[UNK]`；实现和指标见[问题二输入一致性实验](../docs/C/experiments/q2_input_consistent_experiments.md)。
