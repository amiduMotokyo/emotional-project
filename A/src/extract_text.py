from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    load_config,
    load_labels,
    prepare_output_dirs,
    resolve_config_paths,
)


def main(config_path: str | None = None, overwrite: bool = False) -> None:
    cfg = load_config(config_path)
    paths = resolve_config_paths(cfg)
    dirs = prepare_output_dirs(cfg)
    labels = load_labels(cfg)
    text_cfg = cfg["text"]

    try:
        import torch
        from transformers import BertModel, BertTokenizer
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for text extraction.") from exc

    if not paths["bert_path"].exists():
        raise FileNotFoundError(f"BERT directory not found: {paths['bert_path']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(paths["bert_path"]))
    model = BertModel.from_pretrained(str(paths["bert_path"])).to(device)
    model.eval()

    records: list[dict] = []
    failures: list[dict] = []
    for index, row in labels.iterrows():
        sample_id = row["sample_id"]
        output_path = dirs["text"] / f"{sample_id}.npy"
        if output_path.exists() and not overwrite:
            records.append(
                {
                    "sample_id": sample_id,
                    "video_id": row["video_id"],
                    "clip_id": row["clip_id"],
                    "text_length_chars": len(row["text"]),
                    "valid_tokens": None,
                    "feature_file": output_path.relative_to(paths["project_root"]).as_posix(),
                    "status": "existing",
                }
            )
            continue

        try:
            encoded = tokenizer(
                row["text"],
                return_tensors="pt",
                max_length=int(text_cfg["max_length"]),
                padding="max_length",
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state.squeeze(0)
            feature = hidden.detach().cpu().numpy().astype(np.float32)
            if feature.shape != (
                int(text_cfg["max_length"]),
                int(text_cfg["feature_dim"]),
            ):
                raise ValueError(f"Unexpected text feature shape: {feature.shape}")
            np.save(output_path, feature)
            records.append(
                {
                    "sample_id": sample_id,
                    "video_id": row["video_id"],
                    "clip_id": row["clip_id"],
                    "text_length_chars": len(row["text"]),
                    "valid_tokens": int(encoded["attention_mask"].sum().item()),
                    "feature_file": output_path.relative_to(paths["project_root"]).as_posix(),
                    "status": "ok",
                }
            )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
        if (index + 1) % 20 == 0:
            print(f"Text: processed {index + 1}/{len(labels)}")

    pd.DataFrame(records).to_csv(
        dirs["text"] / "text_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if failures:
        pd.DataFrame(failures).to_csv(
            dirs["logs"] / "text_failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"Text extraction complete: {len(records)} records, {len(failures)} failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config, args.overwrite)

