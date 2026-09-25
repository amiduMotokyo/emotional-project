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
    output_dir = A_ROOT / "validation" / "alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(output_root / "summary" / "full_result_summary.csv")
    mapping = pd.read_csv(
        output_root / "alignment" / "alignment_mapping_time.csv"
    )
    segment_len = float(cfg["alignment"]["segment_len"])
    max_segments = int(cfg["alignment"]["max_segments"])

    pivot = summary.pivot(
        index="sample_id",
        columns="modality",
        values=["original_duration_or_length", "valid_length"],
    )
    rows = []
    for sample_id in sorted(summary["sample_id"].unique()):
        audio_len = int(pivot.loc[sample_id, ("valid_length", "audio")])
        vision_len = int(pivot.loc[sample_id, ("valid_length", "vision")])
        audio_duration = float(
            pivot.loc[sample_id, ("original_duration_or_length", "audio")]
        )
        vision_duration = float(
            pivot.loc[sample_id, ("original_duration_or_length", "vision")]
        )
        expected_audio = min(max_segments, math.ceil(audio_duration / segment_len))
        expected_vision = min(
            max_segments, math.ceil(vision_duration / segment_len)
        )
        inter = min(audio_len, vision_len)
        union = max(audio_len, vision_len)
        iou = inter / union if union else 1.0
        sample_map = mapping[mapping["sample_id"] == sample_id]
        mapping_counts = sample_map.groupby("modality").size().to_dict()
        valid_counts = (
            sample_map[sample_map["is_valid"]]
            .groupby("modality")
            .size()
            .to_dict()
        )
        contiguous = {}
        for modality in ("audio", "vision"):
            positions = sorted(
                sample_map.loc[
                    (sample_map["modality"] == modality) & sample_map["is_valid"],
                    "position",
                ].tolist()
            )
            contiguous[modality] = positions == list(range(len(positions)))
        rows.append(
            {
                "sample_id": sample_id,
                "audio_length": audio_len,
                "vision_length": vision_len,
                "length_diff": audio_len - vision_len,
                "exact_match": audio_len == vision_len,
                "within_1": abs(audio_len - vision_len) <= 1,
                "valid_mask_iou": iou,
                "audio_duration_sec": audio_duration,
                "vision_duration_sec": vision_duration,
                "expected_audio_length": expected_audio,
                "expected_vision_length": expected_vision,
                "audio_length_rule_ok": audio_len == expected_audio,
                "vision_length_rule_ok": vision_len == expected_vision,
                "audio_mapping_rows": mapping_counts.get("audio", 0),
                "vision_mapping_rows": mapping_counts.get("vision", 0),
                "audio_valid_mapping": valid_counts.get("audio", 0),
                "vision_valid_mapping": valid_counts.get("vision", 0),
                "audio_mapping_matches": valid_counts.get("audio", 0) == audio_len,
                "vision_mapping_matches": valid_counts.get("vision", 0)
                == vision_len,
                "audio_valid_contiguous": contiguous["audio"],
                "vision_valid_contiguous": contiguous["vision"],
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        output_dir / "alignment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_row = {
        "samples": len(metrics),
        "exact_match_rate": metrics["exact_match"].mean(),
        "within_1_rate": metrics["within_1"].mean(),
        "mean_valid_mask_iou": metrics["valid_mask_iou"].mean(),
        "aggregate_valid_mask_iou": metrics["audio_length"].sum()
        / max(metrics["vision_length"].sum(), 1),
        "min_valid_mask_iou": metrics["valid_mask_iou"].min(),
        "audio_rule_ok_rate": metrics["audio_length_rule_ok"].mean(),
        "vision_rule_ok_rate": metrics["vision_length_rule_ok"].mean(),
        "mapping_length_agreement_rate": np.mean(
            metrics["audio_mapping_matches"] & metrics["vision_mapping_matches"]
        ),
        "contiguous_valid_rate": np.mean(
            metrics["audio_valid_contiguous"]
            & metrics["vision_valid_contiguous"]
        ),
    }
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "alignment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics[metrics["length_diff"] != 0].to_csv(
        output_dir / "alignment_mismatch_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame([summary_row]).to_string(index=False))


if __name__ == "__main__":
    main()

