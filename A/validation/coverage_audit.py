from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


A_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A_ROOT))

from src.utils import load_config, load_labels, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_config_paths(cfg)
    labels = load_labels(cfg)
    output_root = paths["output_root"]
    output_dir = A_ROOT / "validation" / "coverage"
    output_dir.mkdir(parents=True, exist_ok=True)

    import pickle

    final_path = output_root / "features" / "final_features.pkl"
    with final_path.open("rb") as handle:
        final = pickle.load(handle)

    final_ids = set(final["id"])
    expected_ids = set(labels["sample_id"])
    rows = []
    for row in labels.itertuples(index=False):
        sample_id = row.sample_id
        video = paths["video_root"] / row.video_id / f"{row.clip_id}.mp4"
        audio = output_root / "features" / "audio" / f"{sample_id}.npy"
        visual = output_root / "features" / "visual" / f"{sample_id}.npy"
        text = output_root / "features" / "text" / f"{sample_id}.npy"
        shapes = {}
        for modality, path in (("audio", audio), ("visual", visual), ("text", text)):
            shapes[modality] = (
                tuple(np.load(path, mmap_mode="r").shape) if path.exists() else None
            )
        expected_shapes = {
            "audio": (50, 25),
            "visual": (50, 20),
            "text": (50, 768),
        }
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": row.video_id,
                "clip_id": row.clip_id,
                "video_exists": video.exists(),
                "audio_exists": audio.exists(),
                "visual_exists": visual.exists(),
                "text_exists": text.exists(),
                "final_id_exists": sample_id in final_ids,
                "audio_shape": str(shapes["audio"]),
                "visual_shape": str(shapes["visual"]),
                "text_shape": str(shapes["text"]),
                "audio_shape_ok": shapes["audio"] == expected_shapes["audio"],
                "visual_shape_ok": shapes["visual"] == expected_shapes["visual"],
                "text_shape_ok": shapes["text"] == expected_shapes["text"],
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(
        output_dir / "coverage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(
        [
            {
                "label_samples": len(labels),
                "unique_label_ids": labels["sample_id"].nunique(),
                "video_files": int(audit["video_exists"].sum()),
                "audio_files": int(audit["audio_exists"].sum()),
                "visual_files": int(audit["visual_exists"].sum()),
                "text_files": int(audit["text_exists"].sum()),
                "final_ids": len(final_ids),
                "missing_video": int((~audit["video_exists"]).sum()),
                "missing_audio": int((~audit["audio_exists"]).sum()),
                "missing_visual": int((~audit["visual_exists"]).sum()),
                "missing_text": int((~audit["text_exists"]).sum()),
                "missing_final_id": int((~audit["final_id_exists"]).sum()),
                "bad_audio_shape": int((~audit["audio_shape_ok"]).sum()),
                "bad_visual_shape": int((~audit["visual_shape_ok"]).sum()),
                "bad_text_shape": int((~audit["text_shape_ok"]).sum()),
                "extra_final_ids": len(final_ids - expected_ids),
            }
        ]
    )
    summary.to_csv(
        output_dir / "coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

