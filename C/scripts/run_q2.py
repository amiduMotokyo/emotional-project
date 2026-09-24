"""Train Q2 baseline/robust models, evaluate the missingness grid, and predict Attachment 3."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import (LABELS, MultimodalDataset, apply_audio_vision_scale,
                        fit_audio_vision_scale, load_npz, prepare_cache)
from B.src.fusion import Fusion, predict_one
from C.src.metrics import evaluate
from C.src.training import train_one

SEED = 20260924


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    train = load_npz(args.cache / "train.npz")
    valid = load_npz(args.cache / "valid.npz")
    scale = fit_audio_vision_scale(train)
    for split in (train, valid):
        apply_audio_vision_scale(split, scale)
    np.savez_compressed(args.output / "audio_vision_normalization.npz",
                        audio_mean=scale["audio"][0], audio_std=scale["audio"][1],
                        vision_mean=scale["vision"][0], vision_std=scale["vision"][1])
    train_loader = DataLoader(MultimodalDataset(train), batch_size=64, shuffle=True,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(MultimodalDataset(valid), batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=True)
    baseline = Fusion()
    baseline_training = train_one(baseline, train_loader, valid_loader, args.device,
                                  False, SEED, args.output / "baseline.pt")
    robust = Fusion()
    robust_training = train_one(robust, train_loader, valid_loader, args.device,
                                True, SEED + 1, args.output / "robust.pt")
    report = {
        "data_version": "aligned_50.pkl",
        "split_sizes": {"train": len(train["cls"]), "valid": len(valid["cls"])},
        "class_mapping": dict(enumerate(LABELS)),
        "text_encoder": str(args.onnx_model or args.text_model),
        "selection": "validation macro-F1 - 0.15 MAE + 0.05 Pearson; robust model averages clean and three 30% middle-missing cases",
        "baseline_training": baseline_training,
        "robust_training": robust_training,
        "clean": {"baseline": evaluate(baseline, valid_loader, args.device),
                  "robust": evaluate(robust, valid_loader, args.device)},
        "missing_grid": [],
        "baseline_30pct": {},
    }
    for modality in ("text", "audio", "vision"):
        for rate in (0.1, 0.3, 0.5):
            for position in ("start", "middle", "end"):
                result = evaluate(robust, valid_loader, args.device, modality, rate, position)
                report["missing_grid"].append({"modality": modality, "rate": rate,
                                                "position": position, **result})
        report["baseline_30pct"][modality] = evaluate(
            baseline, valid_loader, args.device, modality, 0.3, "middle")
    rows = []
    for path in sorted(args.cache.glob("q2_*.npz")):
        sample = load_npz(path)
        apply_audio_vision_scale(sample, scale)
        probabilities, score, gates = predict_one(robust, sample, args.device)
        rows.append({
            "sample_id": path.stem.removeprefix("q2_"),
            "polarity": LABELS[int(probabilities.argmax())],
            "intensity": round(score, 6),
            "prob_negative": round(float(probabilities[0]), 6),
            "prob_neutral": round(float(probabilities[1]), 6),
            "prob_positive": round(float(probabilities[2]), 6),
            "text_observed_ratio": round(float(sample["tmask"].mean()), 4),
            "audio_observed_ratio": round(float(sample["amask"].mean()), 4),
            "vision_observed_ratio": round(float(sample["vmask"].mean()), 4),
            "gate_text": round(float(gates[0]), 4),
            "gate_audio": round(float(gates[1]), 4),
            "gate_vision": round(float(gates[2]), 4),
        })
    if len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30:
        raise ValueError(f"expected 30 unique Attachment-3 samples, got {len(rows)}")
    with (args.output / "attachment3_predictions.csv").open(
            "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"clean": report["clean"], "test_predictions": len(rows)},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=Path("."))
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    seed_all(SEED)
    if args.prepare:
        count = prepare_cache(args.data_root, args.text_model, args.cache,
                              args.device, args.onnx_model)
        print(f"prepared {count} Attachment-3 samples", flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
