"""Rebuild only cached text vectors with the exact ONNX encoder being deployed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import encode_text, load_npz


def rebuild(source_dir: Path, target_dir: Path, encoder_path: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    paths = [source_dir / "train.npz", source_dir / "valid.npz",
             source_dir / "test.npz",
             *sorted(source_dir.glob("q2_*.npz"))]
    if len(paths) != 33:
        raise ValueError(f"expected train, valid, Attachment-2 test, and 30 Attachment-3 caches; got {len(paths)}")
    for path in paths:
        data = load_npz(path)
        ids = data["token_ids"].astype(np.int64)
        attention = data["attention"].astype(np.int64)
        bert = np.stack([ids, attention, np.zeros_like(ids)], axis=1)
        data["text"] = encode_text(session, bert, "cpu", batch_size=1)
        np.savez_compressed(target_dir / path.name, **data)
        print("rebuilt", path.name, data["text"].shape, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.source, args.target, args.encoder)


if __name__ == "__main__":
    main()
