"""Assemble a separate deployable MiniLM Q2 candidate package."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import fit_audio_vision_scale, load_npz
from B.src.fusion import Fusion
from B.src.temporal_fusion import TemporalFusion
from C.src.reliability_imputation import ReliabilityImputationFusion


def package(cache: Path, checkpoint_path: Path, encoder_path: Path,
            output: Path, class_bias: tuple[float, float, float]) -> dict:
    train = load_npz(cache / "train.npz")
    scale = fit_audio_vision_scale(train)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", "fusion")
    if architecture == "temporal":
        fusion = TemporalFusion()
    elif architecture == "impute":
        fusion = ReliabilityImputationFusion()
    else:
        fusion = Fusion()
    fusion.load_state_dict(checkpoint["fusion_state_dict"])

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(encoder_path, output / "text_encoder_int8.onnx")
    torch.save({"architecture": architecture,
                "state_dict": fusion.state_dict(),
                "source_checkpoint": str(checkpoint_path),
                "seed": checkpoint["seed"],
                "epoch": checkpoint["epoch"]}, output / "model.pt")
    np.savez_compressed(
        output / "audio_vision_normalization.npz",
        audio_mean=scale["audio"][0], audio_std=scale["audio"][1],
        vision_mean=scale["vision"][0], vision_std=scale["vision"][1],
    )
    bias_record = {
        "class_bias": list(class_bias),
        "classes": ["Negative", "Neutral", "Positive"],
        "basis": "5-fold stratified nested selection shared across three seeds",
        "source": "Attachment-2 validation only; applied to log class probabilities",
    }
    (output / "class_bias.json").write_text(
        json.dumps(bias_record, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "model": "all-MiniLM-L6-v2 fine-tuned on Attachment-2 train, top two layers",
        "fusion": {
            "temporal": "B.src.temporal_fusion.TemporalFusion",
            "impute": "C.src.reliability_imputation.ReliabilityImputationFusion",
            "fusion": "B.src.fusion.Fusion",
        }.get(architecture, architecture),
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "encoder_file_bytes": (output / "text_encoder_int8.onnx").stat().st_size,
        "payload_file_bytes_excluding_metadata": sum(
            path.stat().st_size for path in output.iterdir()
            if path.is_file() and path.name != "metadata.json"),
        "class_bias": list(class_bias),
        "external_sentiment_data_used": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["package_file_bytes_including_metadata"] = sum(
        path.stat().st_size for path in output.iterdir() if path.is_file())
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["package_file_bytes_including_metadata"] = sum(
        path.stat().st_size for path in output.iterdir() if path.is_file())
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-bias", type=float, nargs=3,
                        default=(0.0, 0.0, 0.0), metavar=("NEG", "NEU", "POS"))
    args = parser.parse_args()
    result = package(args.cache, args.checkpoint, args.encoder,
                     args.output, tuple(args.class_bias))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
