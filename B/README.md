# B：数据接口、融合与解释

B维护 M1 数据接口、M4融合基线和M6解释实现。当前已接入第二题对齐版的数据路径和融合模型；附件4全量读取、M6解释与原视频证据定位已完成。

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

本次已实现并验证附件2对齐版 train/valid/test、附件3及附件4。问题三用输入遮挡重新推理估计模态和局部作用；门控权重仅作为诊断，遮挡结果也不能解释为因果效应。

修订后的问题二流程发现int8 ONNX编码输出随批次大小变化，因此附件2 train/valid/test 缓存和附件3逐条推理统一采用单样本调用。缺失文本在送入编码器前替换为 `[UNK]`；实现和指标见[问题二输入一致性实验](../docs/C/experiments/q2_input_consistent_experiments.md)。

## 问题三交付（2026-09-25）

复用问题二当前最佳 MiniLM + Fusion 校准包（seed 20260925、第3轮），冻结权重，在附件2验证集复现 ACC 0.63736、Macro-F1 0.60008、MAE 0.59605、Pearson 0.64246；附件4完成20/20条预测与解释。附件4无标签，不能计算测试ACC。验证集用于模型和偏置选择，这组数值不是独立留出泛化估计。

- GitHub可查看：[全量预测与三模态解释CSV](../docs/B/experiments/results/q3_attachment4_predictions_explanations.csv)、[完整局部分布JSON](../docs/B/experiments/results/q3_attachment4_explanations.json)、[验证指标](../docs/B/experiments/results/q3_validation_summary.json)、[逐条验证结果](../docs/B/experiments/results/q3_validation_predictions.csv)、[20张解释卡](../docs/B/paper/cards/)。原始运行输出在本地忽略目录 `B/outputs/q3_explanations_20260925/`，服务器同名目录位于 `/mnt/disk/data/inainai/mosei_e_20260924/q3_explanations_20260925/`。`B/outputs/` 依仓库约定忽略，不纳入Git。
- [接口与边界](../docs/B/requirements/q3_explanation_interface.md)、[实验记录](../docs/B/experiments/q3_explanation_20260925.md)、[问题三论文稿](../docs/B/paper/q3_explain_model.md)、[参数配置](configs/q3_explain_20260925.json)。
- `src/attachment4.py` 配对附件4样本，并调用 `A/src/evidence_locator.py` 将对齐位置匹配至未对齐特征及原视频；`src/q3_explain.py` 实现固定原预测类别的模态和局部遮挡；`scripts/audit_attachment4.py` 审计附件4；`scripts/run_q3.py` 输出全量结果；`scripts/plot_q3.py` 绘图。

服务器运行示例（先准备上述已选模型包和附件数据）：

```bash
/home/inainai/anaconda3/envs/myenv/bin/python B/scripts/audit_attachment4.py \
  --data-root /mnt/disk/data/inainai/mosei_e_20260924/data \
  --tokenizer /mnt/disk/data/inainai/models/all-MiniLM-L6-v2 \
  --output /mnt/disk/data/inainai/mosei_e_20260924/q3_explanations_20260925/attachment4_audit.json \
  --ffprobe /home/inainai/anaconda3/envs/retfound/bin/ffprobe
/home/inainai/anaconda3/envs/myenv/bin/python B/scripts/run_q3.py \
  --data-root /mnt/disk/data/inainai/mosei_e_20260924/data \
  --cache /mnt/disk/data/inainai/mosei_e_20260924/cache_qint8 \
  --package /mnt/disk/data/inainai/mosei_e_20260924/q2_minilm_deploy_candidate_calibrated_20260925 \
  --tokenizer /mnt/disk/data/inainai/models/all-MiniLM-L6-v2 \
  --output /mnt/disk/data/inainai/mosei_e_20260924/q3_explanations_20260925 \
  --device cpu --onnxruntime-path /mnt/disk/data/inainai/mosei_e_20260924/vendor \
  --ffprobe /home/inainai/anaconda3/envs/retfound/bin/ffprobe \
  --ffmpeg /home/inainai/anaconda3/envs/retfound/bin/ffmpeg
/home/inainai/anaconda3/envs/retfound/bin/python B/scripts/plot_q3.py \
  --results /mnt/disk/data/inainai/mosei_e_20260924/q3_explanations_20260925 \
  --output /mnt/disk/data/inainai/mosei_e_20260924/q3_explanations_20260925/plots
```
