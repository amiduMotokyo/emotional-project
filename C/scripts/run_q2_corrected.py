"""Prepare, train and evaluate the corrected aligned Q2 missingness protocol."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import (apply_audio_vision_scale, fit_audio_vision_scale,
                        load_npz)
from B.src.fusion import Fusion
from B.src.temporal_fusion import TemporalFusion
from C.src.q2_protocol import (MODES, POSITIONS, RATES, SELECT_CASES, ViewDataset,
                               evaluate_loader, forward_batch, prepare_case,
                               prepare_views, read_case)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cache: Path):
    train, valid = load_npz(cache / "train.npz"), load_npz(cache / "valid.npz")
    scale = fit_audio_vision_scale(train)
    for item in (train, valid):
        apply_audio_vision_scale(item, scale)
    return train, valid, scale


def loader_for_case(base: dict, case: Path | None, batch_size: int = 128):
    if case is None:
        return DataLoader(ViewDataset(base), batch_size=batch_size, shuffle=False)
    text, masks = read_case(case)
    changed = dict(base)
    changed["text"] = text
    for index, key in enumerate(("tmask", "amask", "vmask")):
        changed[key] = masks[:, index, :]
    return DataLoader(ViewDataset(changed), batch_size=batch_size, shuffle=False)


def model_for(name: str, **kwargs):
    return TemporalFusion(**kwargs) if name == "temporal" else Fusion(**kwargs)


def selection_result(model, clean_loader, case_loaders, device: str) -> dict:
    clean = evaluate_loader(model, clean_loader, device)
    damaged = [evaluate_loader(model, item, device) for item in case_loaders]
    cases = [clean, *damaged]
    selection = (np.mean([x["macro_f1"] for x in cases])
                 - 0.15 * np.mean([x["mae"] for x in cases])
                 + 0.05 * np.mean([x["pearson"] for x in cases]))
    return {"clean": clean, "damaged": damaged, "selection_score": float(selection)}


def train_one(name: str, seed: int, train: dict, clean_loader,
              case_loaders, views: Path, output: Path, device: str,
              epochs: int, patience: int, *, hyperparameters=None,
              model_kwargs=None, fixed_budget=False, resume: Path | None = None) -> dict:
    seed_all(seed)
    hyperparameters = dict(hyperparameters or {})
    model_kwargs = dict(model_kwargs or {})
    model = model_for(name, **model_kwargs).to(device)
    dataset = ViewDataset(train, None if name == "clean_gate" else views)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0,
                        pin_memory=device.startswith("cuda"))
    counts = np.bincount(train["cls"], minlength=3)
    weights = torch.tensor(np.sqrt(counts.sum() / (3 * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=hyperparameters.get('learning_rate', 8e-4),
                                 weight_decay=hyperparameters.get('weight_decay', 0.01))
    best = -1e9
    best_epoch = 0
    history = []
    checkpoint = output / "checkpoints" / f"{name}_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint.with_name(checkpoint.stem + '_last.pt')
    start_epoch = 0
    best_state = None
    if resume is not None:
        if not fixed_budget:
            raise ValueError('resume requires fixed_budget')
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        expected = (name, seed, hyperparameters, model_kwargs)
        actual = (saved['architecture'], saved['seed'], saved['hyperparameters'], saved['model_kwargs'])
        if actual != expected or saved['epoch'] >= epochs:
            raise ValueError('resume configuration or target epoch mismatch')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, best, best_epoch = saved['epoch'], saved['best'], saved['best_epoch']
        best_state, history = saved['best_state'], saved['history']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.perf_counter()
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        dataset.epoch = epoch - 1
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, predicted, _, cls, score = forward_batch(model, batch, device)
            loss = (nn.functional.cross_entropy(logits, cls, weight=weights) +
                    hyperparameters.get('lambda_reg', 0.8) *
                    nn.functional.smooth_l1_loss(predicted, score))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        selection = selection_result(model, clean_loader, case_loaders, device)
        if not np.isfinite(selection['selection_score']) or not np.isfinite(losses).all():
            raise FloatingPointError('non-finite candidate loss or validation score')
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "seconds": time.perf_counter() - started,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
                  if device.startswith('cuda') else None, **selection}
        history.append(record)
        print(name, seed, "epoch", epoch, "loss", round(record["train_loss"], 4),
              "clean_f1", round(selection["clean"]["macro_f1"], 4),
              "selection", round(selection["selection_score"], 4), flush=True)
        if selection["selection_score"] > best + 1e-4:
            best, best_epoch = selection["selection_score"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "architecture": name,
                        "seed": seed, "epoch": epoch,
                        "model_kwargs": model_kwargs, "hyperparameters": hyperparameters,
                        "protocol": "input_unk_before_int8_encoder_v1"}, checkpoint)
        if fixed_budget:
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'architecture': name, 'seed': seed, 'epoch': epoch,
                        'hyperparameters': hyperparameters, 'model_kwargs': model_kwargs,
                        'best': best, 'best_epoch': best_epoch, 'best_state': best_state,
                        'history': history, 'python_rng': random.getstate(),
                        'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if device.startswith('cuda') else []},
                       last_checkpoint)
        if not fixed_budget and epoch - best_epoch >= patience:
            break
    terminal = selection
    # Also supports resuming into a different output directory with no new best.
    torch.save({'state_dict': best_state, 'architecture': name, 'seed': seed,
                'epoch': best_epoch, 'model_kwargs': model_kwargs,
                'hyperparameters': hyperparameters,
                'protocol': 'input_unk_before_int8_encoder_v1'}, checkpoint)
    model.load_state_dict(best_state)
    final = selection_result(model, clean_loader, case_loaders, device)
    return {"architecture": name, "seed": seed, "checkpoint": str(checkpoint),
            "best_epoch": best_epoch, "best_selection": float(best),
            "final": final, "terminal": terminal, "epochs_completed": epoch,
            "last_checkpoint": str(last_checkpoint) if fixed_budget else None,
            "model_kwargs": model_kwargs, "hyperparameters": hyperparameters,
            "history": history}


def prepare(args, train: dict, valid: dict) -> None:
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.encoder), providers=["CPUExecutionProvider"])
    views = args.output / "views"
    if not (views / "train_text.npy").exists():
        prepare_views(train, session, views, args.views, args.seed)
    cases = args.output / "valid_cases"
    for mode, rate, position in SELECT_CASES:
        path = prepare_case(valid, session, cases, mode, rate, position)
        print("prepared", path.name, flush=True)


def train(args, train_data: dict, valid_data: dict) -> list[dict]:
    cases = args.output / "valid_cases"
    clean_loader = loader_for_case(valid_data, None)
    case_loaders = [loader_for_case(valid_data,
                    cases / f"{mode}_{round(rate * 100):02d}_{position}.npz")
                    for mode, rate, position in SELECT_CASES]
    records = []
    for name in ("clean_gate", "robust_gate", "temporal"):
        for seed in (args.seed, args.seed + 1, args.seed + 2):
            records.append(train_one(name, seed, train_data, clean_loader,
                                     case_loaders, args.output / "views",
                                     args.output, args.device, args.epochs,
                                     args.patience))
            (args.output / "training_report.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def load_checkpoint(path: Path, device: str):
    state = torch.load(path, map_location=device, weights_only=False)
    model = model_for(state["architecture"], **state.get('model_kwargs', {})).to(device)
    model.load_state_dict(state["state_dict"])
    return model


def evaluate(args, valid: dict, records: list[dict]):
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.encoder), providers=["CPUExecutionProvider"])
    directory = args.output / "valid_cases"
    # All 7 nonempty modality subsets, 3 rates, and 3 positions.
    case_paths = [(mode, rate, position,
                   prepare_case(valid, session, directory, mode, rate, position))
                  for mode in MODES for rate in RATES for position in POSITIONS]
    by_name = {}
    for record in records:
        by_name.setdefault(record["architecture"], []).append(record)
    best_arch = max(("robust_gate", "temporal"),
                    key=lambda name: np.mean([r["best_selection"] for r in by_name[name]]))
    selected = max(by_name[best_arch], key=lambda item: item["best_selection"])
    (args.output / "selected_model.json").write_text(json.dumps({
        "architecture": best_arch, "seed": selected["seed"],
        "checkpoint": selected["checkpoint"],
        "architecture_selection": "mean validation selection score across three seeds",
        "seed_selection": "highest validation selection score within selected architecture",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    # Report all three seeds for each robust candidate; clean baseline at the same conditions.
    for record in records:
        model = load_checkpoint(Path(record["checkpoint"]), args.device)
        for mode, rate, position, path in case_paths:
            result = evaluate_loader(model, loader_for_case(valid, path), args.device)
            rows.append({"architecture": record["architecture"], "seed": record["seed"],
                         "mode": mode, "rate": rate, "position": position, **result})
        print("evaluated grid", record["architecture"], record["seed"], flush=True)
    (args.output / "missing_grid.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "missing_grid.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("architecture", "seed", "mode", "rate",
                         "position", "accuracy", "macro_f1", "mae", "pearson", "n"))
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in rows])
    model = load_checkpoint(Path(selected["checkpoint"]), args.device)
    clean = loader_for_case(valid, None)
    raw = evaluate_loader(model, clean, args.device)
    coherent = evaluate_loader(model, clean, args.device, coherent=True)
    (args.output / "selected_validation.json").write_text(json.dumps({
        "raw": raw, "coherent": coherent, "selected": selected["checkpoint"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("selected", best_arch, selected["seed"], "raw", raw,
          "coherent", coherent, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_data, valid_data, scale = load_data(args.cache)
    np.savez_compressed(args.output / "audio_vision_normalization.npz",
                        audio_mean=scale["audio"][0], audio_std=scale["audio"][1],
                        vision_mean=scale["vision"][0], vision_std=scale["vision"][1])
    if args.phase == "prepare":
        prepare(args, train_data, valid_data)
    elif args.phase == "train":
        train(args, train_data, valid_data)
    else:
        records = json.loads((args.output / "training_report.json").read_text(encoding="utf-8"))
        evaluate(args, valid_data, records)


if __name__ == "__main__":
    main()
