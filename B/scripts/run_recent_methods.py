"""Evaluate Q2 adaptations of recent methods on the existing frozen-feature cache."""
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

from B.src.data import LABELS, MultimodalDataset, apply_audio_vision_scale, fit_audio_vision_scale, load_npz
from B.src.recent_methods import DPDFAligned, EBMCAligned, CMADAligned, cmad_distillation
from B.scripts.run_literature_models import aggregate, evaluate_model, evaluate_selection, seed_all
from C.src.missingness import sample_training_masks
from C.src.training import criterion

BUILDERS = {"dpdf_aligned": DPDFAligned, "ebmc_aligned": EBMCAligned,
            "cmad_aligned": CMADAligned}


def fit(name, seed, train_loader, valid_loader, counts, args):
    seed_all(seed)
    model = BUILDERS[name]().to(args.device)
    teacher = None
    if name == "cmad_aligned":
        if not args.teacher_dir:
            raise ValueError("--teacher-dir is required for CMAD")
        teacher_path = args.teacher_dir / "gated_baseline" / f"seed_{seed}.pt"
        saved = torch.load(teacher_path, map_location=args.device, weights_only=False)
        teacher = CMADAligned().to(args.device)
        teacher.load_state_dict(saved["state_dict"])
        teacher.eval()
        teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    weights = torch.tensor(np.sqrt(counts.sum() / (3 * np.maximum(counts, 1))),
                           dtype=torch.float32, device=args.device)
    checkpoint = args.output / "checkpoints" / name / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = -1e9, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            text, audio, vision, tm, am, vm, cls, score = batch
            masks = sample_training_masks([tm, am, vm], probability=0.8) if args.robust else [tm, am, vm]
            vals = [x.to(args.device, non_blocking=True) for x in
                    (text, audio, vision, *masks, cls, score)]
            text, audio, vision, tm_corrupt, am_corrupt, vm_corrupt, cls, score = vals
            optimizer.zero_grad(set_to_none=True)
            logits, prediction, gates = model(text, audio, vision, tm_corrupt, am_corrupt, vm_corrupt)
            loss = criterion(logits, prediction, cls, score, weights)
            if name == "ebmc_aligned":
                loss = loss + model.auxiliary_loss(cls, score, gates)
            elif name == "cmad_aligned":
                clean_inputs = (text, audio, vision, tm.to(args.device), am.to(args.device), vm.to(args.device))
                loss = loss + cmad_distillation(model, teacher, clean_inputs,
                                                (text, audio, vision, tm_corrupt, am_corrupt, vm_corrupt))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        selected, cases = evaluate_selection(model, valid_loader, args.device, args.robust)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "selection_score": selected,
                        "selection_macro_f1": float(np.mean([x["macro_f1"] for x in cases]))})
        print(f"[{name} seed={seed}] epoch={epoch} loss={history[-1]['train_loss']:.4f} "
              f"F1={history[-1]['selection_macro_f1']:.4f} select={selected:.4f}", flush=True)
        if selected > best_score + 1e-4:
            best_score, best_epoch, stale = selected, epoch, 0
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "model": name, "epoch": epoch}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=args.device,
                                     weights_only=False)["state_dict"])
    metrics = {"clean": evaluate_model(model, valid_loader, args.device), "missing_grid": []}
    for modality in ("text", "audio", "vision"):
        for rate in (0.1, 0.3, 0.5):
            for position in ("start", "middle", "end"):
                result = evaluate_model(model, valid_loader, args.device, modality, rate, position)
                metrics["missing_grid"].append({"modality": modality, "rate": rate,
                                                 "position": position, **result})
    return {"method": name, "seed": seed, "robust_missingness": args.robust,
            "best_epoch": best_epoch, "best_selection_score": best_score,
            "epochs_run": len(history), "history": history, "metrics": metrics,
            "checkpoint": str(checkpoint),
            "n_parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--methods", nargs="+", choices=list(BUILDERS), default=list(BUILDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260924, 20260925, 20260926])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--robust", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
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
    runs = []
    for seed in args.seeds:
        for index, name in enumerate(args.methods):
            seed_all(seed + index)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=args.device.startswith("cuda"))
            valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False,
                                      num_workers=0, pin_memory=args.device.startswith("cuda"))
            print(f"START {name} seed={seed} robust={args.robust}", flush=True)
            runs.append(fit(name, seed, train_loader, valid_loader, counts, args))
            result = {"data_version": "aligned_50.pkl / frozen MiniLM cache",
                      "split_sizes": {"train": len(train_data), "valid": len(valid_data)},
                      "class_mapping": dict(enumerate(LABELS)),
                      "adaptation_warning": "Small mask-aware adaptations; not exact author implementations.",
                      "training": {"robust": args.robust, "epochs_max": args.epochs,
                                   "patience": args.patience, "batch_size": args.batch_size,
                                   "learning_rate": args.lr, "weight_decay": 0.01,
                                   "loss": "weighted CE + 0.8 SmoothL1; method auxiliaries"},
                      "runs": runs, "summary": aggregate(runs)}
            (args.output / "recent_comparison.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"DONE {name} seed={seed}", flush=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "n_seeds", "accuracy", "macro_f1", "mae", "pearson"])
        for name, item in result["summary"].items():
            writer.writerow([name, item["n_seeds"],
                             *[item["clean"][key]["mean"] for key in
                               ("accuracy", "macro_f1", "mae", "pearson")]])
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
