"""Evaluate a fine-tuned Q2 checkpoint on clean and controlled validation masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from C.scripts.finetune_q2_minilm import (FineTuneModel, RawTextDataset, evaluate,
                                         load_data)


def evaluate_checkpoint(cache: Path, encoder: Path, checkpoint_path: Path,
                        output: Path, device: str) -> dict:
    train, valid = load_data(cache)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "fusion")
    model = FineTuneModel(encoder, checkpoint["unfreeze_layers"], architecture).to(device)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.fusion.load_state_dict(checkpoint["fusion_state_dict"])
    loader = DataLoader(RawTextDataset(valid), batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=device.startswith("cuda"))
    cases = [{"condition": "clean", **evaluate(model, loader, device)}]
    for geometry in ("contiguous", "pointwise"):
        for mode in ("text", "audio", "vision"):
            cases.append({"condition": mode, "geometry": geometry,
                          **evaluate(model, loader, device, mode, geometry)})
    report = {
        "checkpoint": str(checkpoint_path),
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "architecture": architecture,
        "training_geometry": checkpoint.get("mask_geometry", "legacy_contiguous"),
        "validation_metrics": cases,
        "validation_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.cache, args.encoder, args.checkpoint,
                                 args.output, args.device)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
