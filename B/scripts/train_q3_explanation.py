"""Fine-tune the selected Fusion head with a deletion-faithfulness constraint.

The text encoder and its single-sample ONNX cache stay frozen. Validation is
used only for epoch selection and diagnostics, never for gradients.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import apply_audio_vision_scale, load_npz
from B.src.fusion import Fusion
from C.src.q2_protocol import coherent_score

FEATURES = ("text", "audio", "vision")
MASKS = ("tmask", "amask", "vmask")


class CachedDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self) -> int:
        return len(self.data["cls"])

    def __getitem__(self, index: int):
        row = self.data
        return tuple(torch.from_numpy(row[key][index].copy())
                     for key in (*FEATURES, *MASKS)) + (
            torch.tensor(int(row["cls"][index]), dtype=torch.long),
            torch.tensor(float(row["score"][index]), dtype=torch.float32),
        )


def ablate_modality(inputs: tuple[torch.Tensor, ...], index: int) -> tuple[torch.Tensor, ...]:
    """Whole-modality deletion at the Fusion input, identical to Q3 inference."""
    values = list(inputs)
    values[index] = torch.zeros_like(values[index])
    values[index + 3] = torch.zeros_like(values[index + 3])
    return tuple(values)


def deletion_effects(model: Fusion, inputs: tuple[torch.Tensor, ...],
                     classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean gates and three true-class logit drops with dropout disabled."""
    training = model.training
    model.eval()
    logits, _, gates = model(*inputs)
    clean = logits.gather(1, classes[:, None])
    effects = []
    for index in range(3):
        removed, _, _ = model(*ablate_modality(inputs, index))
        effects.append((clean - removed.gather(1, classes[:, None])).squeeze(1))
    if training:
        model.train()
    return gates, torch.stack(effects, dim=1)


def explanation_loss(gates: torch.Tensor, effects: torch.Tensor,
                     masks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Align gates to actual positive, label-class deletion effects.

    The deletion response is a stop-gradient target; no extra labels are used.
    When all observed effects are nonpositive, the target is uniform over the
    observed modalities. Missing modalities receive neither target nor gate.
    """
    present = torch.stack([mask.any(dim=1) for mask in masks], dim=1)
    positive = effects.detach().relu() * present
    fallback = present.float() / present.float().sum(dim=1, keepdim=True).clamp_min(1)
    target = torch.where(positive.sum(dim=1, keepdim=True) > 1e-7,
                         positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-7),
                         fallback)
    # Gate output has exact zeros for unavailable modalities.
    return (target * (target.clamp_min(1e-8).log() -
                      gates.clamp_min(1e-8).log())).sum(dim=1).mean()


def validation(model: Fusion, loader: DataLoader, device: str,
               class_bias: torch.Tensor) -> dict:
    model.eval()
    predictions, truths, intensities, scores = [], [], [], []
    gate_matches, gate_l1, n_explained = 0, 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            *raw_inputs, labels, score = (item.to(device) for item in batch)
            inputs = tuple(raw_inputs)
            logits, intensity, gates = model(*inputs)
            predicted = (logits + class_bias).argmax(dim=1)
            predictions.extend(predicted.cpu().numpy().tolist())
            truths.extend(labels.cpu().numpy().tolist())
            intensities.extend(intensity.cpu().numpy().tolist())
            scores.extend(score.cpu().numpy().tolist())
            _, effects = deletion_effects(model, inputs, predicted)
            present = torch.stack([mask.any(dim=1) for mask in inputs[3:]], dim=1)
            positive = effects.relu() * present
            fallback = present.float() / present.float().sum(dim=1, keepdim=True).clamp_min(1)
            target = torch.where(positive.sum(dim=1, keepdim=True) > 1e-7,
                                 positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-7),
                                 fallback)
            gate_matches += int((gates.argmax(dim=1) == target.argmax(dim=1)).sum())
            gate_l1 += float((gates - target).abs().sum())
            n_explained += len(predicted)
    predictions = np.asarray(predictions)
    truths = np.asarray(truths)
    served = coherent_score(predictions, np.asarray(intensities))
    return {
        "n": len(truths),
        "accuracy": float(accuracy_score(truths, predictions)),
        "macro_f1": float(f1_score(truths, predictions, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(scores, served)),
        "gate_deletion_top1_match": gate_matches / n_explained,
        "gate_deletion_l1": gate_l1 / n_explained,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explanation-weight", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.explanation_weight < 0:
        raise ValueError("explanation weight must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    train = load_npz(args.cache / "train.npz")
    valid = load_npz(args.cache / "valid.npz")
    with np.load(args.package / "audio_vision_normalization.npz") as scale:
        normalizer = {feature: (scale[f"{feature}_mean"], scale[f"{feature}_std"])
                      for feature in ("audio", "vision")}
    apply_audio_vision_scale(train, normalizer)
    apply_audio_vision_scale(valid, normalizer)
    train_loader = DataLoader(CachedDataset(train), batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    valid_loader = DataLoader(CachedDataset(valid), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    saved = torch.load(args.package / "model.pt", map_location="cpu", weights_only=False)
    if saved["architecture"] != "fusion":
        raise ValueError("this explanation experiment requires Fusion")
    model = Fusion().to(args.device)
    model.load_state_dict(saved["state_dict"])
    bias = json.loads((args.package / "class_bias.json").read_text(encoding="utf-8"))
    class_bias = torch.tensor(bias["class_bias"], device=args.device)
    counts = np.bincount(train["cls"], minlength=3)
    class_weights = torch.tensor(np.sqrt(counts.sum() / (3 * counts)),
                                 dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    baseline = validation(model, valid_loader, args.device, class_bias)
    history = [{"epoch": 0, "validation": baseline}]
    best_key = (-1.0, -1.0)
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_losses, explanation_losses = [], []
        for batch in train_loader:
            *raw_inputs, cls, score = (item.to(args.device) for item in batch)
            inputs = tuple(raw_inputs)
            logits, intensity, _ = model(*inputs)
            supervised = (nn.functional.cross_entropy(logits, cls, weight=class_weights)
                          + 0.8 * nn.functional.smooth_l1_loss(intensity, score))
            if args.explanation_weight:
                gates, effects = deletion_effects(model, inputs, cls)
                explanatory = explanation_loss(gates, effects, inputs[3:])
            else:
                explanatory = supervised.new_zeros(())
            total = supervised + args.explanation_weight * explanatory
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_losses.append(float(total.detach().cpu()))
            explanation_losses.append(float(explanatory.detach().cpu()))
        metrics = validation(model, valid_loader, args.device, class_bias)
        record = {"epoch": epoch, "train_loss": float(np.mean(training_losses)),
                  "explanation_loss": float(np.mean(explanation_losses)),
                  "validation": metrics}
        history.append(record)
        key = (metrics["accuracy"], metrics["macro_f1"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save({"architecture": "fusion",
                        "state_dict": {name: value.detach().cpu().clone()
                                       for name, value in model.state_dict().items()},
                        "seed": args.seed, "epoch": epoch,
                        "explanation_weight": args.explanation_weight,
                        "base_checkpoint": str(args.package / "model.pt")},
                       args.output / "model.pt")
        print(json.dumps({"weight": args.explanation_weight, "epoch": epoch,
                          "accuracy": metrics["accuracy"],
                          "macro_f1": metrics["macro_f1"],
                          "gate_match": metrics["gate_deletion_top1_match"]}), flush=True)
        (args.output / "training_report.json").write_text(json.dumps({
            "explanation_weight": args.explanation_weight, "seed": args.seed,
            "epochs_completed": epoch, "best_epoch": best_epoch,
            "best_key_accuracy_macro_f1": list(best_key),
            "baseline_validation": baseline, "history": history,
            "data_boundary": "Attachment-2 train gradients; valid checkpoint selection only",
            "text_encoder": "frozen selected MiniLM int8 ONNX; cached single-sample embeddings",
            "loss": "weighted CE + 0.8 SmoothL1 + lambda KL(positive label-class deletion effects || gates)",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["elapsed_seconds"] = time.perf_counter() - started
    report["selected_new_checkpoint"] = best_epoch > 0
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_epoch > 0:
        for name in ("text_encoder_int8.onnx", "audio_vision_normalization.npz",
                     "class_bias.json"):
            shutil.copy2(args.package / name, args.output / name)
        (args.output / "metadata.json").write_text(json.dumps({
            "source": str(args.package), "seed": args.seed, "epoch": best_epoch,
            "explanation_weight": args.explanation_weight,
            "training": "Fusion-only continued training with deletion-gate alignment",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_accuracy": best_key[0],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
