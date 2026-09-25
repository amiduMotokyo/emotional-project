from __future__ import annotations

import argparse
import math
import pickle
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
    validation_root = A_ROOT / "validation" / "text_time"
    output_dir = validation_root / "time_aligned"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_root / "features" / "final_features.pkl").open("rb") as handle:
        features = pickle.load(handle)
    mapping = pd.read_csv(validation_root / "text_time_alignment.csv")
    sample_ids = list(features["id"])
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    text_features = np.asarray(features["text"], dtype=np.float32)
    segment_len = float(cfg["alignment"]["segment_len"])
    max_segments = int(cfg["alignment"]["max_segments"])

    aligned = np.zeros_like(text_features, dtype=np.float32)
    masks = np.zeros((len(sample_ids), max_segments), dtype=np.uint8)
    summary_rows = []

    for sample_id, sample_tokens in mapping.groupby("sample_id", sort=True):
        if sample_id not in sample_index:
            continue
        index = sample_index[sample_id]
        weight_sum = np.zeros(max_segments, dtype=np.float64)
        accumulator = np.zeros((max_segments, text_features.shape[-1]), dtype=np.float64)
        timestamped = sample_tokens[
            sample_tokens["timestamp_available"]
            & sample_tokens["token_valid"]
            & (sample_tokens["source_word_index"] >= 0)
        ]

        for token in timestamped.itertuples(index=False):
            token_position = int(token.token_position)
            if token_position < 0 or token_position >= text_features.shape[1]:
                continue
            start = max(0.0, float(token.start_sec))
            end = min(max_segments * segment_len, float(token.end_sec))
            if end <= start:
                end = min(max_segments * segment_len, start + segment_len)
            first_segment = max(0, math.floor(start / segment_len))
            last_segment = min(
                max_segments - 1,
                max(first_segment, math.ceil(end / segment_len) - 1),
            )
            for segment in range(first_segment, last_segment + 1):
                segment_start = segment * segment_len
                segment_end = (segment + 1) * segment_len
                overlap = max(0.0, min(end, segment_end) - max(start, segment_start))
                if overlap <= 0:
                    continue
                accumulator[segment] += overlap * text_features[index, token_position]
                weight_sum[segment] += overlap

        valid = weight_sum > 0
        if valid.any():
            accumulator[valid] /= weight_sum[valid, None]
            aligned[index, valid] = accumulator[valid].astype(np.float32)
        masks[index] = valid.astype(np.uint8)
        summary_rows.append(
            {
                "sample_id": sample_id,
                "timestamped_tokens": len(timestamped),
                "text_time_segments": int(valid.sum()),
                "empty_text_segments": int((~valid).sum()),
                "text_time_coverage": float(valid.mean()),
            }
        )

    np.save(output_dir / "text_time_aligned.npy", aligned)
    np.save(output_dir / "text_time_mask.npy", masks)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "text_time_aligned_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_time_aligned = {
        "audio": features["audio"],
        "vision": features["vision"],
        "text": features["text"],
        "text_time_aligned": aligned,
        "text_time_mask": masks,
        "id": sample_ids,
        "text_len": features["text_len"],
        "audio_len": features["audio_len"],
        "vision_len": features["vision_len"],
    }
    with (output_dir / "final_features_time_aligned.pkl").open("wb") as handle:
        pickle.dump(final_time_aligned, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"text_time_aligned={aligned.shape}, mask={masks.shape}, "
        f"mean_coverage={summary['text_time_coverage'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
