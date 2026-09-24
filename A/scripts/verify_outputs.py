from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, load_labels, resolve_config_paths


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg = load_config()
    paths = resolve_config_paths(cfg)
    output_root = paths["output_root"]
    labels = load_labels(cfg)
    expected_ids = sorted(labels["sample_id"])

    final_path = output_root / "features" / "final_features.pkl"
    with final_path.open("rb") as handle:
        final = pickle.load(handle)

    if sorted(final["id"]) != expected_ids:
        raise ValueError("final_features.pkl does not cover all 100 label samples.")
    if final["audio"].shape != (100, 50, 25):
        raise ValueError(f"Unexpected audio shape: {final['audio'].shape}")
    if final["vision"].shape != (100, 50, 20):
        raise ValueError(f"Unexpected vision shape: {final['vision'].shape}")
    if final["text"].shape != (100, 50, 768):
        raise ValueError(f"Unexpected text shape: {final['text'].shape}")

    manifest_rows: list[dict] = []
    for sample_id in expected_ids:
        for modality, suffix, expected_shape in (
            ("audio", "audio", (50, 25)),
            ("vision", "visual", (50, 20)),
            ("text", "text", (50, 768)),
        ):
            feature_path = output_root / "features" / suffix / f"{sample_id}.npy"
            array = np.load(feature_path, mmap_mode="r")
            if array.shape != expected_shape:
                raise ValueError(
                    f"{feature_path} shape {array.shape} != {expected_shape}"
                )
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "modality": modality,
                    "feature_file": feature_path.relative_to(PROJECT_ROOT).as_posix(),
                    "shape": "x".join(map(str, array.shape)),
                    "dtype": str(array.dtype),
                    "size_bytes": feature_path.stat().st_size,
                    "sha256": _sha256(feature_path),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        output_root / "summary" / "file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.read_csv(output_root / "summary" / "full_result_summary.csv")
    time_mapping = pd.read_csv(
        output_root / "alignment" / "alignment_mapping_time.csv"
    )
    token_mapping = pd.read_csv(
        output_root / "alignment" / "alignment_mapping_token.csv"
    )
    checks = {
        "sample_count": len(expected_ids),
        "feature_manifest_rows": len(manifest),
        "summary_rows": len(summary),
        "time_mapping_rows": len(time_mapping),
        "token_mapping_rows": len(token_mapping),
        "audio_shape": list(final["audio"].shape),
        "vision_shape": list(final["vision"].shape),
        "text_shape": list(final["text"].shape),
        "audio_nan": int(np.isnan(final["audio"]).sum()),
        "vision_nan": int(np.isnan(final["vision"]).sum()),
        "text_nan": int(np.isnan(final["text"]).sum()),
    }
    report_lines = ["# Verification Report", ""]
    report_lines.extend(f"- {key}: {value}" for key, value in checks.items())
    report_lines.extend(
        [
            "",
            "All 100 label samples have audio, vision and text feature files.",
            "Feature manifest: outputs/summary/file_manifest.csv.",
            "Time mapping: outputs/alignment/alignment_mapping_time.csv.",
            "Token mapping: outputs/alignment/alignment_mapping_token.csv.",
        ]
    )
    (output_root / "logs" / "verification_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()

