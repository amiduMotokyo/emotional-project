"""Rebuild Attachment-2 train/valid embeddings with the selected Q3 ONNX encoder."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import encode_text, load_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--onnxruntime-path", type=Path)
    args = parser.parse_args()
    if args.onnxruntime_path:
        sys.path.insert(0, str(args.onnxruntime_path))
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.package / "text_encoder_int8.onnx"),
                                   providers=["CPUExecutionProvider"])
    args.target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        row = load_npz(args.source / f"{split}.npz")
        ids = row["token_ids"].astype(np.int64)
        attention = row["attention"].astype(np.int64)
        bert = np.stack([ids, attention, np.zeros_like(ids)], axis=1)
        row["text"] = encode_text(session, bert, "cpu", batch_size=1)
        np.savez_compressed(args.target / f"{split}.npz", **row)
        print(f"reencoded {split} {row['text'].shape}", flush=True)


if __name__ == "__main__":
    main()
