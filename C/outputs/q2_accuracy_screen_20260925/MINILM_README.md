# MiniLM + Fusion 校准候选（2026-09-25）

本目录归档旧Fusion的63.74%验证ACC实验，不包含后续MRRF、LSRRF或2026-09-26轻量续训实验。原项目提交包未替换。

## 结果入口

- `minilm_finetune/seed20260924.json`、`seed20260925.json`、`seed20260926.json`：三种子训练日志。
- `minilm_finetune/class_bias_nested_cv.json`：五折共享偏置校准。
- `minilm_finetune/validation_grid_full_calibrated_seed20260925.json`：1个原始条件+63个连续缺失条件。
- `deployment_candidate_seed20260925/attachment3_predictions.csv`：附件3全量30条，ID唯一。
- `deployment_candidate_seed20260925/`：Fusion权重、int8 MiniLM、scaler、bias、metadata，合计23,188,911字节（含预测CSV）。实际以manifest为准。
- `minilm_manifest.json`：各结果文件与部署文件的SHA256及字节数。

## 指标口径

MiniLM-L6-v2顶部两层微调；每种子10轮，最佳epoch2/3/5；选定seed20260925、epoch3。int8未校准ACC63.19%，共享bias[0.2,0,0]后464/728=63.74%。三个种子平均由62.55%变为63.42%。63种缺失条件平均61.47%。以上为反复用于选择的附件2验证集，不能当独立测试成绩；本轮没有用附件2test或外部情感标注数据。模型与偏置选择复用验证集，五折校准也不消除上游选模偏差。

## 推理

在仓库根目录，安装项目所需PyTorch、NumPy、scikit-learn及ONNX Runtime，然后运行：

```sh
python C/scripts/infer_q2_corrected.py --data-root /path/to/data --package C/outputs/q2_accuracy_screen_20260925/deployment_candidate_seed20260925 --output /path/to/attachment3_predictions.csv --device cpu
```

`data-root`下需有`附件3-模态缺失特征样本/对齐版本`。采用逐样本CPU ONNX路径。推理兼容旧包无bias时使用零偏置，并保留远端已有model_kwargs支持。

上传前已使用本次代码及服务器原包重跑30条附件3预测，与历史表逐字节一致；Python编译检查与git diff检查通过。没有重新训练或改写历史指标。

## 训练与保存范围

训练入口`C/scripts/finetune_q2_minilm.py`；评估`evaluate_q2_minilm_onnx.py`；校准`calibrate_q2_minilm_bias.py`；导出`export_q2_minilm_onnx.py`；打包`package_q2_minilm_candidate.py`。默认Fusion配置支持本轮流程；脚本还保留可选结构兼容代码，这些分支不代表本候选使用了它们。

87MB原始浮点训练checkpoint、原始附件、通用预训练模型及特征缓存未提交。原始最佳训练目录为服务器`/mnt/disk/data/inainai/mosei_e_20260924/q2_minilm_finetune_10ep_20260925/`，权重`checkpoints/minilm_top2_20260925.pt`含encoder_state_dict与fusion_state_dict。部署包源目录为同base下`q2_minilm_deploy_candidate_calibrated_20260925/`。只需复现附件3推理可使用仓库部署文件和赛题数据，无须训练checkpoint。

研究说明见`docs/B/research/q2_accuracy_redesign.md`（路径相对仓库根目录）。本次归档将该说明中后续结构实验段落省略，保留此轮相关内容；更早筛查数据不一定在本目录中。
