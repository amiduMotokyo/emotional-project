from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


A_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A_ROOT))

from src.utils import load_config, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_config_paths(cfg)
    output_root = paths["output_root"]
    validation_root = A_ROOT / "validation"
    text_root = validation_root / "text_time"
    alignment_root = validation_root / "alignment"
    alignment_root.mkdir(parents=True, exist_ok=True)

    alignment = pd.read_csv(text_root / "text_time_alignment.csv")
    coverage = pd.read_csv(text_root / "text_time_coverage.csv").set_index(
        "sample_id"
    )
    summary = pd.read_csv(output_root / "summary" / "full_result_summary.csv")
    lengths = summary.pivot(
        index="sample_id", columns="modality", values="valid_length"
    )
    durations = summary.pivot(
        index="sample_id",
        columns="modality",
        values="original_duration_or_length",
    )
    segment_len = float(cfg["alignment"]["segment_len"])
    max_segments = int(cfg["alignment"]["max_segments"])

    rows = []
    for sample_id, sample in alignment.groupby("sample_id", sort=True):
        timestamped = sample[sample["timestamp_available"]].copy()
        start = timestamped["start_sec"].to_numpy(dtype=float)
        end = timestamped["end_sec"].to_numpy(dtype=float)
        duration = max(
            float(durations.loc[sample_id, "audio"]),
            float(durations.loc[sample_id, "vision"]),
        )
        valid_interval = (
            (start >= 0)
            & (end >= start)
            & (end <= duration + 0.05)
            if len(timestamped)
            else np.array([], dtype=bool)
        )

        audio_mask = np.zeros(max_segments, dtype=bool)
        vision_mask = np.zeros(max_segments, dtype=bool)
        text_mask = np.zeros(max_segments, dtype=bool)
        audio_mask[: int(lengths.loc[sample_id, "audio"])] = True
        vision_mask[: int(lengths.loc[sample_id, "vision"])] = True
        for row in timestamped.itertuples(index=False):
            first = max(0, min(max_segments - 1, math.floor(row.start_sec / segment_len)))
            last = max(0, min(max_segments - 1, math.ceil(row.end_sec / segment_len) - 1))
            text_mask[first : last + 1] = True

        def iou(left: np.ndarray, right: np.ndarray) -> float:
            union = int(np.logical_or(left, right).sum())
            return float(np.logical_and(left, right).sum()) / union if union else 1.0

        rows.append(
            {
                "sample_id": sample_id,
                "word_match_rate": float(coverage.loc[sample_id, "word_match_rate"]),
                "token_timestamp_coverage": float(
                    coverage.loc[sample_id, "token_timestamp_coverage"]
                ),
                "timestamped_tokens": int(len(timestamped)),
                "valid_timestamped_tokens": int(valid_interval.sum()),
                "timestamp_valid_rate": (
                    float(valid_interval.mean()) if len(valid_interval) else 0.0
                ),
                "text_segments": int(text_mask.sum()),
                "text_audio_mask_iou": iou(text_mask, audio_mask),
                "text_vision_mask_iou": iou(text_mask, vision_mask),
                "three_modality_mask_iou": (
                    float(
                        np.logical_and.reduce(
                            [audio_mask, vision_mask, text_mask]
                        ).sum()
                    )
                    / max(
                        int(
                            np.logical_or.reduce(
                                [audio_mask, vision_mask, text_mask]
                            ).sum()
                        ),
                        1,
                    )
                ),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(
        alignment_root / "text_time_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_row = {
        "samples": len(audit),
        "mean_word_match_rate": audit["word_match_rate"].mean(),
        "mean_token_timestamp_coverage": audit[
            "token_timestamp_coverage"
        ].mean(),
        "samples_token_coverage_ge_80pct": int(
            (audit["token_timestamp_coverage"] >= 0.8).sum()
        ),
        "timestamp_valid_rate_micro": (
            audit["valid_timestamped_tokens"].sum()
            / max(audit["timestamped_tokens"].sum(), 1)
        ),
        "mean_text_audio_mask_iou": audit["text_audio_mask_iou"].mean(),
        "mean_text_vision_mask_iou": audit["text_vision_mask_iou"].mean(),
        "mean_three_modality_mask_iou": audit[
            "three_modality_mask_iou"
        ].mean(),
    }
    pd.DataFrame([summary_row]).to_csv(
        alignment_root / "text_time_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame([summary_row]).to_string(index=False))


if __name__ == "__main__":
    main()

