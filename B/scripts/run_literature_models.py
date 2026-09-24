"""Train literature-inspired multimodal fusion models on the Q2 train/valid cache.

Attachment 3 is deliberately never loaded here. All models use the same frozen
MiniLM token states, AV normalization, missingness augmentation, loss, checkpoint
selection metric, and validation corruption grid.
"""
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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import (LABELS, MultimodalDataset, apply_audio_vision_scale,
                        fit_audio_vision_scale, load_npz)
from B.src.fusion import Fusion
from B.src.literature import (LowRankMultimodalFusion, MAGMiniLM,
                              MISAStyleFusion, MulTAligned)
from C.src.missingness import corrupt_masks, sample_training_masks
from C.src.training import criterion

MODEL_BUILDERS = {
    "gated_baseline": Fusion,
    "lmf": LowRankMultimodalFusion,
    "misa": MISAStyleFusion,
    "mult": MulTAligned,
    "mag_minilm": MAGMiniLM,
}
SELECTION_CASES = (("none", 0.0, "middle"),
                   ("text", 0.3, "middle"),
                   ("audio", 0.3, "middle"),
                   ("vision", 0.3, "middle"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selection_value(scores: list[dict]) -> float:
    f1 = np.mean([x["macro_f1"] for x in scores])
    mae = np.mean([x["mae"] for x in scores])
    corr = np.mean([x["pearson"] for x in scores])
    return float(f1 - 0.15 * mae + 0.05 * corr)


def evaluate_model(model, loader, device: str, mode: str = "none", rate: float = 0.3,
                   position: str = "middle", seed: int = 20260924) -> dict:
    """Evaluate with corruption on CPU to avoid per-row GPU synchronization."""
    model.eval()
    old_state = random.getstate()
    random.seed(seed)
    classes, actual, predicted, estimates = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            text, audio, vision, tm, am, vm, cls, score = batch
            masks = corrupt_masks([tm, am, vm], mode, rate, position)
            values = [x.to(device, non_blocking=True) for x in
                      (text, audio, vision, *masks, cls, score)]
            text, audio, vision, tm, am, vm, cls, score = values
            logits, output_score, _ = model(text, audio, vision, tm, am, vm)
            classes.extend(cls.cpu().numpy())
            actual.extend(score.cpu().numpy())
            predicted.extend(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())
            estimates.extend(output_score.cpu().numpy())
    random.setstate(old_state)
    cls, actual = np.asarray(classes), np.asarray(actual)
    predicted, estimates = np.asarray(predicted), np.asarray(estimates)
    pearson = float(np.corrcoef(actual, estimates)[0, 1]) if np.std(estimates) > 1e-8 else 0.0
    return {"accuracy": float(accuracy_score(cls, predicted)),
            "macro_f1": float(f1_score(cls, predicted, average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(actual, estimates)),
            "pearson": pearson,
            "per_class_f1": f1_score(cls, predicted, average=None, labels=[0, 1, 2],
                                      zero_division=0).tolist(),
            "n": int(len(cls))}


def evaluate_selection(model, loader, device: str, robust: bool) -> tuple[float, list[dict]]:
    if not robust:
        scores = [evaluate_model(model, loader, device)]
    else:
        scores = [evaluate_model(model, loader, device, mode, rate, position)
                  for mode, rate, position in SELECTION_CASES]
    return selection_value(scores), scores


def fit_model(name: str, seed: int, train_loader, valid_loader, counts: np.ndarray,
              output: Path, device: str, epochs: int, patience: int, lr: float,
              robust: bool) -> dict:
    seed_all(seed)
    model = MODEL_BUILDERS[name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    weights = torch.tensor(np.sqrt(counts.sum() / (3 * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)
    checkpoint = output / "checkpoints" / name / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = -1e9, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            text, audio, vision, tm, am, vm, cls, score = batch
            masks = (sample_training_masks([tm, am, vm], probability=0.8)
                     if robust else [tm, am, vm])
            values = [x.to(device, non_blocking=True) for x in
                      (text, audio, vision, *masks, cls, score)]
            text, audio, vision, tm, am, vm, cls, score = values
            optimizer.zero_grad(set_to_none=True)
            logits, estimate, _ = model(text, audio, vision, tm, am, vm)
            loss = criterion(logits, estimate, cls, score, weights)
            if isinstance(model, MISAStyleFusion):
                loss = loss + model.auxiliary_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        score, cases = evaluate_selection(model, valid_loader, device, robust)
        record = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)),
                  "selection_score": score,
                  "selection_macro_f1": float(np.mean([x["macro_f1"] for x in cases])),
                  "selection_mae": float(np.mean([x["mae"] for x in cases])),
                  "selection_pearson": float(np.mean([x["pearson"] for x in cases]))}
        history.append(record)
        print(f"[{name} seed={seed}] epoch={epoch} loss={record['train_loss']:.4f} "
              f"F1={record['selection_macro_f1']:.4f} MAE={record['selection_mae']:.4f} "
              f"select={score:.4f}", flush=True)
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "model": name, "epoch": epoch}, checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    metrics = {"clean": evaluate_model(model, valid_loader, device), "missing_grid": []}
    for modality in ("text", "audio", "vision"):
        for rate in (0.1, 0.3, 0.5):
            for position in ("start", "middle", "end"):
                result = evaluate_model(model, valid_loader, device, modality, rate, position)
                metrics["missing_grid"].append({"modality": modality, "rate": rate,
                                                 "position": position, **result})
    return {"method": name, "seed": seed, "robust_missingness": robust,
            "best_epoch": best_epoch,
            "best_selection_score": best_score, "epochs_run": len(history),
            "history": history, "metrics": metrics,
            "checkpoint": str(checkpoint)}


def aggregate(runs: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = {}
    for run in runs:
        by_method.setdefault(run["method"], []).append(run)
    summary = {}
    for method, method_runs in by_method.items():
        clean_metrics = ("accuracy", "macro_f1", "mae", "pearson")
        clean = {}
        for key in clean_metrics:
            values = [r["metrics"]["clean"][key] for r in method_runs]
            clean[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))
                          if len(values) > 1 else 0.0,
                          "per_seed": values}
        conditions = {}
        for index, item in enumerate(method_runs[0]["metrics"]["missing_grid"]):
            key = f"{item['modality']}_{int(item['rate'] * 100)}_{item['position']}"
            values = [r["metrics"]["missing_grid"][index]["macro_f1"] for r in method_runs]
            mae_values = [r["metrics"]["missing_grid"][index]["mae"] for r in method_runs]
            corr_values = [r["metrics"]["missing_grid"][index]["pearson"] for r in method_runs]
            conditions[key] = {
                "macro_f1_mean": float(np.mean(values)),
                "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "mae_mean": float(np.mean(mae_values)),
                "pearson_mean": float(np.mean(corr_values)),
            }
        summary[method] = {"n_seeds": len(method_runs), "clean": clean,
                           "missing_grid": conditions,
                           "best_epoch_mean": float(np.mean([r["best_epoch"] for r in method_runs]))}
    return summary


def run(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    train = load_npz(args.cache / "train.npz")
    valid = load_npz(args.cache / "valid.npz")
    scale = fit_audio_vision_scale(train)
    for split in (train, valid):
        apply_audio_vision_scale(split, scale)
    np.savez_compressed(args.output / "audio_vision_normalization.npz",
                        audio_mean=scale["audio"][0], audio_std=scale["audio"][1],
                        vision_mean=scale["vision"][0], vision_std=scale["vision"][1])
    train_data, valid_data = MultimodalDataset(train), MultimodalDataset(valid)
    counts = np.bincount(train["cls"], minlength=3)
    run_records = []
    total = len(MODEL_BUILDERS) * len(args.seeds)
    run_number = 0
    for seed in args.seeds:
        for model_index, name in enumerate(MODEL_BUILDERS):
            run_number += 1
            seed_all(seed + model_index)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=device_is_cuda(args.device))
            valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False,
                                      num_workers=0, pin_memory=device_is_cuda(args.device))
            print(f"START {run_number}/{total}: {name}, seed={seed}", flush=True)
            run_records.append(fit_model(name, seed, train_loader, valid_loader, counts,
                                         args.output, args.device, args.epochs,
                                         args.patience, args.lr, args.robust))
            payload = {"data_version": "aligned_50.pkl / cached frozen MiniLM states",
                       "split_sizes": {"train": len(train_data), "valid": len(valid_data)},
                       "class_mapping": dict(enumerate(LABELS)),
                       "seeds": args.seeds, "robust_missingness_augmentation":
                           ("p=0.8; contiguous rate sampled from 0.1, 0.2, 0.3, 0.5; "
                            "1-3 random modalities" if args.robust else "disabled"),
                       "checkpoint_selection":
                           ("mean(clean + text/audio/vision 30%-middle missing) macro-F1 "
                            "- 0.15*MAE + 0.05*Pearson" if args.robust else
                            "clean validation macro-F1 - 0.15*MAE + 0.05*Pearson"),
                       "training": {"robust_missingness": args.robust,
                                    "epochs_max": args.epochs, "patience": args.patience,
                                    "batch_size": args.batch_size, "learning_rate": args.lr,
                                    "weight_decay": 0.01, "optimizer": "AdamW",
                                    "task_loss": "class-weighted CE + 0.8 SmoothL1 intensity",
                                    "misa_auxiliary": "0.3 CMD + 0.1 difference + 1.0 reconstruction"},
                       "mag_note": "MAG-style MiniLM adapter over frozen token states; not end-to-end MAG-BERT.",
                       "runs": run_records, "summary": aggregate(run_records)}
            (args.output / "per_seed_results.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"FINISHED {run_number}/{total}: {name}, seed={seed}", flush=True)
    summary = aggregate(run_records)
    payload["runs"] = run_records
    payload["summary"] = summary
    (args.output / "literature_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seeds", "accuracy_mean", "accuracy_std",
                         "macro_f1_mean", "macro_f1_std", "mae_mean", "mae_std",
                         "pearson_mean", "pearson_std", "best_epoch_mean"])
        for method, record in summary.items():
            writer.writerow([method, record["n_seeds"],
                             record["clean"]["accuracy"]["mean"], record["clean"]["accuracy"]["std"],
                             record["clean"]["macro_f1"]["mean"], record["clean"]["macro_f1"]["std"],
                             record["clean"]["mae"]["mean"], record["clean"]["mae"]["std"],
                             record["clean"]["pearson"]["mean"], record["clean"]["pearson"]["std"],
                             record["best_epoch_mean"]])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def device_is_cuda(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True,
                        help="directory with train.npz and valid.npz only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260924, 20260925, 20260926])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--robust", action=argparse.BooleanOptionalAction, default=True,
                        help="use contiguous missingness augmentation; --no-robust selects clean-only training")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run(args)


if __name__ == "__main__":
    main()
