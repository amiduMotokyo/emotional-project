from __future__ import annotations

import argparse
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
    output_dir = A_ROOT / "validation" / "padding"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_root / "features" / "final_features.pkl").open("rb") as handle:
        data = pickle.load(handle)
    token_mapping = pd.read_csv(
        output_root / "alignment" / "alignment_mapping_token.csv"
    )

    rows = []
    for index, sample_id in enumerate(data["id"]):
        for modality, array_key, length_key in (
            ("audio", "audio", "audio_len"),
            ("vision", "vision", "vision_len"),
            ("text", "text", "text_len"),
        ):
            array = np.asarray(data[array_key][index])
            length = int(data[length_key][index])
            padded = array[length:]
            valid = array[:length]
            if modality in {"audio", "vision"}:
                padded_max_abs = (
                    float(np.max(np.abs(padded))) if padded.size else 0.0
                )
                padding_ok = padded_max_abs == 0.0
                padding_rule = "zero_padding"
                padded_constant = None
            else:
                padded_max_abs = (
                    float(np.ptp(padded, axis=0).max()) if padded.size else 0.0
                )
                sample_tokens = token_mapping[
                    token_mapping["sample_id"] == sample_id
                ]
                mapped_valid = int(sample_tokens["is_valid"].sum())
                padding_ok = (
                    len(sample_tokens) == array.shape[0]
                    and mapped_valid == length
                )
                padding_rule = "attention_mask"
                padded_constant = padded_max_abs < 1e-6
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality": modality,
                    "valid_length": length,
                    "padded_length": array.shape[0] - length,
                    "padded_max_abs": padded_max_abs,
                    "padding_rule": padding_rule,
                    "padding_ok": padding_ok,
                    "padded_rows_constant": padded_constant,
                    "valid_nan": int(np.isnan(valid).sum()),
                    "valid_inf": int(np.isinf(valid).sum()),
                    "all_nan": int(np.isnan(array).sum()),
                    "all_inf": int(np.isinf(array).sum()),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(
        output_dir / "padding_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        audit.groupby("modality")
        .agg(
            records=("sample_id", "size"),
            padding_ok_rate=("padding_ok", "mean"),
            valid_nan=("valid_nan", "sum"),
            valid_inf=("valid_inf", "sum"),
            max_padded_abs=("padded_max_abs", "max"),
        )
        .reset_index()
    )
    summary["padding_rule"] = summary["modality"].map(
        {
            "audio": "zero_padding",
            "vision": "zero_padding",
            "text": "attention_mask",
        }
    )
    summary.to_csv(
        output_dir / "padding_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
