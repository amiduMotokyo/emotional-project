# 问题 1 处理日志与复现说明

## 1. 输入

- 附件 1：37 个 `video_id` 文件夹，100 条 `clip_id.mp4`。
- 标签文件：`label-100.xlsx`，字段为 `video_id, clip_id, text, label, annotation`。
- 输入路径由 `configs/config.yaml` 统一定义。

## 2. 环境

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows 64 位 |
| Python | 3.10.17 |
| numpy | 1.25.0 |
| pandas | 2.2.3 |
| openSMILE | 2.6.0 |
| librosa | 0.11.0 |
| PyTorch | 2.7.0 |
| transformers | 4.52.4 |
| scikit-learn | 1.6.1 |
| scipy | 1.11.4 |
| OpenFace | 2.2.0 |
| ffmpeg | 4.4 |
| BERT | `bert-base-uncased` |

## 3. 处理流程

```text
clip_id.mp4
  -> ffmpeg -> 16 kHz mono wav
  -> openSMILE eGeMAPSv02 LLD -> (n_frames, 25)
  -> 0.5 s mean pooling -> (50, 25)

clip_id.mp4
  -> OpenFace -> 17 AU + 3 pose -> (n_frames, 20)
  -> 0.5 s mean pooling -> (50, 20)

label-100.xlsx text
  -> bert-base-uncased
  -> last hidden state -> (50, 768)

three modalities -> sample_id -> final_features.pkl
```

## 4. 对齐规则

- 音频和视觉：每 0.5 秒一个时间片，共 50 片，覆盖 0–25 秒。
- 片内多帧取算术平均；不足 50 片补 0；超过 50 片截断。
- 文本：按 BERT token 位置对齐到 50 个位置。
- 文本位置是对齐位置，不是逐词时间戳；时间证据由
  `alignment_mapping_time.csv` 和典型样本图给出。

## 5. 输出

| 文件 | 形状或行数 |
|---|---|
| `features/audio/*.npy` | 100 × `(50, 25)` |
| `features/visual/*.npy` | 100 × `(50, 20)` |
| `features/text/*.npy` | 100 × `(50, 768)` |
| `features/final_features.pkl` | audio `(100,50,25)`、vision `(100,50,20)`、text `(100,50,768)` |
| `summary/full_result_summary.csv` | 300 行 |
| `alignment/alignment_mapping_time.csv` | 10000 行 |
| `alignment/alignment_mapping_token.csv` | 5000 行 |

## 6. 复现命令

在 `A/` 目录执行：

```bash
python scripts/run_all.py
python scripts/verify_outputs.py
```

已有完整输出时，只需执行第二个命令进行核查，不需要重复提取。

## 7. 已知说明

- 文本和音频/视觉的 `original_duration_or_length` 单位分别是字符和秒，
  `full_result_summary.csv` 已显式增加 `unit` 列。
- openSMILE eGeMAPSv02 对无法估计的共振峰参数使用 `-201` 哨兵值，
  原始特征文件保留该值。
- 样本 `-yRb-Jum7EQ_1` 原时长约 29.133 秒，按 50 片规则截断到 25 秒。
- OpenFace 在多人画面中跟随主检测目标，可能对视觉特征产生一定影响。

