"""Prepare aligned train/valid and Attachment-3 caches through B's M1 interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import prepare_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    count = prepare_cache(args.data_root, args.text_model, args.cache,
                          args.device, args.onnx_model)
    print(f"prepared {count} Attachment-3 samples", flush=True)


if __name__ == "__main__":
    main()
