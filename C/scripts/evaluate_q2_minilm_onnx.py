"""Evaluate a quantized fine-tuned encoder with the final per-sample ONNX path."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import apply_audio_vision_scale, fit_audio_vision_scale, load_npz
from B.src.fusion import Fusion
from B.src.temporal_fusion import TemporalFusion
from C.src.reliability_imputation import ReliabilityImputationFusion
from C.src.q2_protocol import (MODES, POSITIONS, RATES, _encode_with_masks,
                               base_masks, evaluate_loader, ViewDataset)
from C.src.missingness import corrupt_masks


def case_data(data: dict, session, mode: str | None, rate: float = 0.3,
              position: str = "middle", text_cache: dict | None = None):
    masks = base_masks(data)
    if mode is not None:
        masks = corrupt_masks(masks, mode, rate=rate, position=position,
                              rng=random.Random(20260924))
    stacked = np.stack([mask.numpy() for mask in masks], axis=1)
    # Text representations depend only on whether/how text was masked. Reuse
    # the same ONNX results across audio/vision-only conditions and combinations.
    has_text_missing = mode is not None and "text" in mode.split("+")
    key = ("masked", rate, position) if has_text_missing else ("clean",)
    if text_cache is None:
        text_cache = {}
    if key not in text_cache:
        text_cache[key] = _encode_with_masks(session, data, stacked)
    text = text_cache[key]
    changed = dict(data)
    changed["text"] = text
    for index, key in enumerate(("tmask", "amask", "vmask")):
        changed[key] = stacked[:, index, :]
    return changed


def observed_missing_rate(data: dict, masks: np.ndarray) -> dict[str, float]:
    original = np.stack([data[name].astype(bool)
                         for name in ("tmask", "amask", "vmask")], axis=1)
    removed = original & ~masks
    names = ("text", "audio", "vision")
    result = {}
    for index, name in enumerate(names):
        denominator = int(original[:, index, :].sum())
        result[name] = (float(removed[:, index, :].sum() / denominator)
                        if denominator else 0.0)
    return result


def summarize_grid(cases: list[dict]) -> dict:
    grouped = {
        "mode": defaultdict(list),
        "rate": defaultdict(list),
        "position": defaultdict(list),
        "mode_rate": defaultdict(list),
        "mode_position": defaultdict(list),
        "rate_position": defaultdict(list),
    }
    for case in cases:
        key = case["condition"]
        acc = case["accuracy"]
        grouped["mode"][key["mode"]].append(acc)
        grouped["rate"][str(key["rate"])].append(acc)
        grouped["position"][key["position"]].append(acc)
        grouped["mode_rate"][f"{key['mode']}|{key['rate']}"].append(acc)
        grouped["mode_position"][f"{key['mode']}|{key['position']}"].append(acc)
        grouped["rate_position"][f"{key['rate']}|{key['position']}"].append(acc)
    return {
        name: {key: float(np.mean(values)) for key, values in groups.items()}
        for name, groups in grouped.items()
    }


def evaluate(cache: Path, encoder_path: Path, checkpoint_path: Path,
             output: Path, device: str, full_grid: bool = False,
             class_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    import onnxruntime as ort
    train = load_npz(cache / "train.npz")
    valid = load_npz(cache / "valid.npz")
    scale = fit_audio_vision_scale(train)
    apply_audio_vision_scale(train, scale)
    apply_audio_vision_scale(valid, scale)
    session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "fusion")
    if architecture == "temporal":
        model = TemporalFusion().to(device)
    elif architecture == "impute":
        model = ReliabilityImputationFusion().to(device)
    else:
        model = Fusion().to(device)
    model.load_state_dict(checkpoint["fusion_state_dict"])
    cases = []
    text_cache = {}
    conditions = [(None, None, None)]
    if full_grid:
        conditions.extend(product(MODES, RATES, POSITIONS))
    else:
        conditions.extend((mode, 0.3, "middle")
                          for mode in ("text", "audio", "vision"))
    base = base_masks(valid)
    clean_prediction_rows = None
    for condition_index, (mode, rate, position) in enumerate(conditions, start=1):
        changed = case_data(valid, session, mode,
                            rate=0.3 if rate is None else rate,
                            position="middle" if position is None else position,
                            text_cache=text_cache)
        loader = DataLoader(ViewDataset(changed), batch_size=64, shuffle=False)
        metrics = evaluate_loader(model, loader, device, coherent=True,
                                  include_rows=mode is None,
                                  class_bias=class_bias)
        rows = metrics.pop("rows", None)
        result = {"mode": "clean" if mode is None else mode, **metrics}
        if mode is None:
            true_classes = rows["actual_class"].astype(np.int64)
            predicted_classes = rows["predicted"].astype(np.int64)
            result["confusion_matrix_true_by_pred"] = confusion_matrix(
                true_classes, predicted_classes, labels=[0, 1, 2]).tolist()
            result["true_class_counts"] = np.bincount(true_classes, minlength=3).tolist()
            result["predicted_class_counts"] = np.bincount(
                predicted_classes, minlength=3).tolist()
            clean_prediction_rows = [{
                "row": index,
                "true_class": int(true_classes[index]),
                "predicted_class": int(predicted_classes[index]),
                "probabilities": rows["probabilities"][index].tolist(),
                "target_score": float(rows["actual"][index]),
                "raw_predicted_score": float(rows["raw_score"][index]),
                "served_score": float(rows["score"][index]),
            } for index in range(len(true_classes))]
            result["condition"] = {"mode": "clean", "rate": None,
                                   "position": None}
            result["observed_missing_rate_by_modality"] = {
                "text": 0.0, "audio": 0.0, "vision": 0.0}
        else:
            masks = np.stack([mask.numpy() for mask in corrupt_masks(
                base, mode, rate=rate, position=position,
                rng=random.Random(20260924))], axis=1)
            result["condition"] = {"mode": mode, "rate": rate,
                                   "position": position}
            result["observed_missing_rate_by_modality"] = observed_missing_rate(
                valid, masks)
        cases.append(result)
        if condition_index == 1 or condition_index % 7 == 0 or condition_index == len(conditions):
            print(f"evaluated validation cases {condition_index}/{len(conditions)}",
                  flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "encoder": str(encoder_path),
        "encoder_execution": "ONNX Runtime CPU; batch size 1 per sample",
        "metrics": cases,
        "package_encoder_bytes": encoder_path.stat().st_size,
        "text_encoder_forward_cache_entries": len(text_cache),
        "inference_class_bias": list(class_bias),
    }
    if full_grid:
        report["grid_summary"] = summarize_grid(cases[1:])
    prediction_path = output.with_name(f"{output.stem}_clean_predictions.json")
    prediction_path.write_text(json.dumps(clean_prediction_rows, ensure_ascii=False,
                                          indent=2), encoding="utf-8")
    report["clean_predictions_file"] = prediction_path.name
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-grid", action="store_true",
                        help="evaluate all 7 modality sets x 3 rates x 3 positions")
    parser.add_argument("--class-bias", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0), metavar=("NEG", "NEU", "POS"),
                        help="additive bias applied to log class probabilities")
    args = parser.parse_args()
    report = evaluate(args.cache, args.encoder, args.checkpoint, args.output,
                      args.device, args.full_grid, tuple(args.class_bias))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
