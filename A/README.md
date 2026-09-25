# A: 特征、对齐与视频证据

本目录整理题 E 问题 1 的代码、配置、100 条样本特征、汇总表、时间映射与证据产物。

## 目录结构

```text
A/
├── configs/
│   └── config.yaml
├── src/
│   ├── utils.py
│   ├── extract_audio.py
│   ├── extract_visual.py
│   ├── extract_text.py
│   └── merge_and_export.py
├── scripts/
│   ├── run_all.py
│   └── verify_outputs.py
├── validation/
│   ├── coverage/
│   ├── alignment/
│   ├── padding/
│   ├── stream_sync/
│   ├── downstream/
│   ├── autodl/
│   └── text_time/
│       ├── time_aligned/
│       └── words/
├── outputs/
│   ├── features/
│   │   ├── audio/
│   │   ├── visual/
│   │   ├── text/
│   │   └── final_features.pkl
│   ├── alignment/
│   ├── summary/
│   ├── logs/
│   └── evidence/
└── docs/
    └── A/
        ├── requirements/
        ├── innovation/
        ├── experiment/
        └── paper/
```

## 环境

- Windows 64 位
- Python 3.10.17
- numpy 1.25.0、pandas 2.2.3
- openSMILE 2.6.0、librosa 0.11.0
- PyTorch 2.7.0、transformers 4.52.4
- OpenFace 2.2.0
- ffmpeg 4.4
- BERT: `bert-base-uncased`

安装核心依赖：

```bash
pip install numpy==1.25.0 pandas==2.2.3 opensmile==2.6.0 librosa==0.11.0 torch==2.7.0 transformers==4.52.4 scikit-learn==1.6.1 scipy==1.11.4 pyyaml
```

## 路径配置

所有输入路径集中在 `configs/config.yaml`。其中：

- `video_root`: 附件 1 的 100 条原始视频目录；
- `label_file`: `label-100.xlsx`；
- `bert_path`: 本地 `bert-base-uncased` 模型目录；
- `openface_exe`: OpenFace `FeatureExtraction.exe`；
- `ffmpeg_exe`: ffmpeg 可执行文件或 PATH 中的 `ffmpeg`；
- `output_root`: 默认写入本目录的 `outputs/`。

## 运行

在 `A/` 目录运行：

```bash
python scripts/run_all.py
```

强制覆盖已经存在的特征：

```bash
python scripts/run_all.py --overwrite
```

检查现有产物：

```bash
python scripts/verify_outputs.py
```

总入口依次执行：

1. `extract_audio.py`: ffmpeg 分离音频，openSMILE 提取并对齐；
2. `extract_visual.py`: OpenFace 提取 AU/姿态并对齐；
3. `extract_text.py`: BERT 编码英文转写；
4. `merge_and_export.py`: 合并特征并导出全量汇总和对齐映射。

## 输出

- `outputs/features/audio/*.npy`: 100 条 `(50,25)` 音频特征；
- `outputs/features/visual/*.npy`: 100 条 `(50,20)` 视觉特征；
- `outputs/features/text/*.npy`: 100 条 `(50,768)` 文本特征；
- `outputs/features/final_features.pkl`: 三模态合并特征；
- `outputs/summary/full_result_summary.csv`: 300 行全量结果表；
- `outputs/summary/file_manifest.csv`: 特征文件对应关系、形状和校验值；
- `outputs/alignment/alignment_mapping_time.csv`: 音频/视觉时间片映射；
- `outputs/alignment/alignment_mapping_token.csv`: 文本词元映射；
- `outputs/logs/processing_log.md`: 处理过程与复现说明；
- `outputs/evidence/`: 证据定位接口、典型样本图和特征质量验证图。

## 验证产物

### 覆盖完整性

- `validation/coverage/coverage_audit.csv`: 100 条样本逐条覆盖审计；
- `validation/coverage/coverage_summary.csv`: 原始样本、模态文件和输出特征对应关系统计。

### 时序对齐与填充

- `validation/alignment/alignment_metrics.csv`: 严格长度一致率、正负一个时间片容差一致率和有效掩码 IoU；
- `validation/alignment/alignment_summary.csv`: 音频与视觉时间片一致性汇总；
- `validation/padding/padding_audit.csv`: 音频、视觉和文本的填充规则审计；
- `validation/padding/padding_summary.csv`: 零填充、非法值和有效位置连续率汇总；
- `validation/stream_sync/stream_sync_audit.csv`: 原始流起止时间与重采样边界诊断；
- `validation/stream_sync/stream_sync_summary.csv`: ffmpeg 提取误差和时长边界汇总。

### 文本时间对齐

- `validation/autodl/autodl_whisper_words.py`: 使用 Whisper 提取词级时间戳；
- `validation/autodl/map_text_time.py`: 将原始词映射到 BERT 子词，并按 0.5 秒时间片加权聚合；
- `validation/text_time/text_time_alignment.csv`: 词、BERT 子词和时间片的逐项映射；
- `validation/text_time/text_time_coverage.csv`: 文本时间片覆盖率审计；
- `validation/text_time/time_aligned/text_time_aligned.npy`: `(100,50,768)` 时间对齐文本特征；
- `validation/text_time/time_aligned/text_time_mask.npy`: `(100,50)` 有效时间片掩码；
- `validation/text_time/time_aligned/final_features_time_aligned.pkl`: 保留原始 float32 精度的时间对齐三模态特征；
- `validation/text_time/words/`: Whisper 词级时间戳原始结果和汇总清单。

### 下游探针

- `validation/downstream/downstream_baseline_folds.csv`: 基线模型逐折结果；
- `validation/downstream/downstream_baseline_summary.csv`: 基线模型汇总指标；
- `validation/downstream/downstream_probe_folds.csv`: 特征可用性探针逐折结果；
- `validation/downstream/downstream_probe_summary.csv`: 特征可用性探针汇总指标；
- `validation/README.md`: 验证流程、指标含义和运行方式。

## 证据定位接口

`outputs/evidence/evidence_interface.md` 明确说明时间单位、精度和样本对应关系。B 侧读取时间片时，应使用 `outputs/alignment/alignment_mapping_time.csv`，并通过 `sample_id = video_id + "_" + clip_id` 与原始视频对应。

## 实验验证

问题一的覆盖完整性、时序对齐、padding 规则、文本时间对齐和 GroupKFold
下游探针结果均以逐样本 CSV、汇总 CSV 和可复现脚本的形式保存在 `validation/`
目录中。`validation/autodl/README.md` 和 `validation/text_time/README.md`
分别说明服务器端词级时间戳提取和本地映射步骤。

## 已知边界

- 音频和视觉均按 0.5 秒时间片对齐，最多 50 片，覆盖 0–25 秒。
- 超过 25 秒的 `-yRb-Jum7EQ_1` 样本尾部截断。
- 默认文本特征按 BERT token 位置组织；另保留基于 Whisper 词级时间戳的 0.5 秒时间片加权聚合分支。
- 时间片内不存在带时间戳的词时，时间对齐文本特征填 0，掩码填 0，不改变原始 token 位置特征。
- openSMILE 对无法估计的共振峰参数使用 `-201` 哨兵值，原始特征中保留。

问题三证据定位接口：`src/evidence_locator.py` 将B输出的对齐局部位置映射到原文字符区间或附件4未对齐音视频特征帧，并估算原视频时间；无法精确匹配则返回状态。B通过 `B/src/attachment4.py` 调用，详情见 [问题三接口](../docs/B/requirements/q3_explanation_interface.md)。
