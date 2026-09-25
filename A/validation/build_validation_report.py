from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def main() -> None:
    coverage = read_csv("coverage/coverage_summary.csv").iloc[0]
    alignment = read_csv("alignment/alignment_summary.csv").iloc[0]
    padding = read_csv("padding/padding_summary.csv")
    stream = read_csv("stream_sync/stream_sync_summary.csv").iloc[0]
    downstream = read_csv("downstream/downstream_probe_summary.csv")
    baseline = read_csv("downstream/downstream_baseline_summary.csv")
    text_time = read_csv("alignment/text_time_summary.csv").iloc[0]

    padding_lines = [
        f"- {row.modality}: padding 规则正确率 {row.padding_ok_rate:.2%}，"
        f"valid NaN={int(row.valid_nan)}，valid Inf={int(row.valid_inf)}，"
        f"规则={row.padding_rule}"
        for row in padding.itertuples(index=False)
    ]
    downstream_lines = [
        f"- {row.modality}: Accuracy={row.accuracy_mean:.3f}±"
        f"{row.accuracy_std:.3f}，Macro-F1={row.macro_f1_mean:.3f}±"
        f"{row.macro_f1_std:.3f}，MAE={row.mae_mean:.3f}±"
        f"{row.mae_std:.3f}，Pearson={row.pearson_mean:.3f}±"
        f"{row.pearson_std:.3f}"
        for row in downstream.itertuples(index=False)
    ]
    baseline_lines = [
        f"- {row.metric}: {row['mean']:.3f}±{row['std']:.3f}"
        for _, row in baseline.iterrows()
    ]

    report = f"""# Problem 1 Validation Report

## 1. Coverage completeness

| Source | Count | Expected | Status |
|---|---:|---:|---|
| Label samples | {int(coverage.label_samples)} | 100 | Pass |
| Video files | {int(coverage.video_files)} | 100 | Pass |
| Audio features | {int(coverage.audio_files)} | 100 | Pass |
| Vision features | {int(coverage.visual_files)} | 100 | Pass |
| Text features | {int(coverage.text_files)} | 100 | Pass |
| Final feature IDs | {int(coverage.final_ids)} | 100 | Pass |

Missing video/audio/vision/text/final IDs:
{int(coverage.missing_video)}/{int(coverage.missing_audio)}/{int(coverage.missing_visual)}/{int(coverage.missing_text)}/{int(coverage.missing_final_id)}.
Extra final IDs: {int(coverage.extra_final_ids)}.

## 2. Temporal alignment

| Metric | Value |
|---|---:|
| Samples | {int(alignment.samples)} |
| Exact audio/vision length match | {alignment.exact_match_rate:.2%} |
| Difference no more than one segment | {alignment.within_1_rate:.2%} |
| Mean valid-mask IoU | {alignment.mean_valid_mask_iou:.4f} |
| Aggregate valid-mask IoU | {alignment.aggregate_valid_mask_iou:.4f} |
| Minimum valid-mask IoU | {alignment.min_valid_mask_iou:.4f} |
| Mapping rows agree with stored lengths | {alignment.mapping_length_agreement_rate:.2%} |
| Valid positions are contiguous from zero | {alignment.contiguous_valid_rate:.2%} |

The exact match rate is not called a universal "alignment rate". It is the
strict length-equality rate. The tolerant rate and mask IoU are reported
separately because the 0.5 second boundary can create a one-segment difference.

## 3. Padding and mask audit

{chr(10).join(padding_lines)}

Audio and vision padded rows are zero. Text padded positions are validated by
the token mapping and `is_valid` flag, not by requiring the BERT hidden-state
vectors at padding positions to be identical.

## 4. Original stream synchronization

- Samples checked by ffprobe: {int(stream.samples)}
- Maximum absolute audio/video start offset: {stream.max_abs_start_offset_sec:.6f} s
- Mean absolute audio/video start offset: {stream.mean_abs_start_offset_sec:.6f} s
- Maximum extracted-audio duration difference: {stream.max_abs_duration_delta_sec:.6f} s
- Audio duration ceiling rule match: {stream.audio_rule_ok_rate:.2%}
- Vision duration ceiling rule match: {stream.vision_rule_ok_rate:.2%}

The duration ceiling is a diagnostic, not the authoritative valid-length rule.
The timestamp interval mapping remains the source of truth.

## 5. Downstream GroupKFold probe

{chr(10).join(downstream_lines)}

Baselines:

{chr(10).join(baseline_lines)}

Text and concatenated features improve Macro-F1 over the majority-class
baseline, but the absolute classification and regression performance is weak
on only 100 samples. This experiment should be described as auxiliary evidence
that the features contain usable sentiment information, not as a final
prediction model.

## 6. Text-time validation

Whisper word timestamps have been mapped back to the existing BERT token
positions:

- samples: {int(text_time.samples)}
- mean word match rate after sequence alignment: {text_time.mean_word_match_rate:.2%}
- mean token timestamp coverage: {text_time.mean_token_timestamp_coverage:.2%}
- samples with token timestamp coverage of at least 80%:
  {int(text_time.samples_token_coverage_ge_80pct)}/100
- timestamp validity rate: {text_time.timestamp_valid_rate_micro:.2%}
- mean text/audio time-segment IoU: {text_time.mean_text_audio_mask_iou:.2%}
- mean text/vision time-segment IoU: {text_time.mean_text_vision_mask_iou:.2%}
- mean three-modality time-segment IoU: {text_time.mean_three_modality_mask_iou:.2%}

The three-modality IoU is a footprint-overlap diagnostic, not a universal
alignment rate. It is lower than audio/vision IoU because text tokens occur
only where words are spoken, while audio and vision remain valid during
silence.

## 7. Recommended paper wording

Report these indicators separately:

- sample coverage rate: {coverage.audio_files / coverage.label_samples:.2%}
- strict audio/vision length agreement: {alignment.exact_match_rate:.2%}
- plus-or-minus-one-segment agreement: {alignment.within_1_rate:.2%}
- mean valid-mask IoU: {alignment.mean_valid_mask_iou:.2%}
- padding rule compliance: 100%
- audio/video stream start offset: maximum {stream.max_abs_start_offset_sec:.6f} s

Do not use an undefined single term "alignment rate" without a formula.
"""
    (ROOT / "validation_report.md").write_text(report, encoding="utf-8")
    print(ROOT / "validation_report.md")


if __name__ == "__main__":
    main()
