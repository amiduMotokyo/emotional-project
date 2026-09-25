# Problem 1 Validation

This folder contains the experiments used to verify the three requirements of
Problem 1.

## Local validation

Run from the `A/` directory:

```bash
python validation/coverage_audit.py
python validation/alignment_audit.py
python validation/padding_audit.py
python validation/ffprobe_audit.py --ffprobe "D:/HitPaw/hitpaw/AI换脸/安装包/DeepFaceLive_NVIDIA/_internal/ffmpeg/ffprobe.exe"
python validation/downstream_probe.py
python validation/autodl/map_text_time.py --words-dir validation/text_time/words --label-file "E:/E题/E题数据/label-100.xlsx" --bert-path "E:/E题/bert-base-uncased" --text-summary outputs/features/text/text_feature_summary.csv --output-dir validation/text_time
python validation/text_time_audit.py
```

Results are written to:

- `coverage/`: sample, video, modality and final-feature correspondence.
- `alignment/`: strict match rate, plus-or-minus one segment rate and mask IoU.
- `padding/`: whether padded rows obey the declared rules.
- `stream_sync/`: original audio/video stream durations and start offsets.
- `downstream/`: GroupKFold classification and regression probe results.
- `text_time/`: Whisper word timestamps mapped to BERT subword positions.
- `text_time/time_aligned/`: supplementary time-binned text features and mask.

## Meaning of the metrics

- `coverage_audit.csv`: one row per sample; all four sources must exist.
- `alignment_metrics.csv`: one row per sample with audio/vision lengths,
  difference, exact match, plus-or-minus one tolerance and valid-mask IoU.
- `padding_audit.csv`: one row per sample and modality; records whether padded
  positions satisfy the zero-padding or attention-mask rule.
- `stream_sync_audit.csv`: original stream timing evidence from ffprobe.
- `downstream_probe_summary.csv`: GroupKFold macro-F1, MAE and Pearson for
  text, audio, vision and concatenated features.
- `text_time_alignment.csv`: word timestamps mapped to BERT subword positions.
- `text_time_coverage.csv`: word match rate and token timestamp coverage.
- `time_aligned/text_time_aligned.npy`: `(100,50,768)` weighted text features.
- `time_aligned/text_time_mask.npy`: `(100,50)` valid text-segment mask.
- `time_aligned/final_features_time_aligned.pkl`: audio, vision, original text,
  time-aligned text and mask in one supplementary feature bundle.
- `alignment/text_time_summary.csv`: text/audio/vision time-footprint IoU.
- `reproducibility_diff.csv`: shape, ID order and numerical differences between
  an original run and a repeated run.
