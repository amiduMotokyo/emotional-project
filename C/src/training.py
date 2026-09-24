"""Baseline and corruption-robust fusion training used for Q2 experiments."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import evaluate
from .missingness import sample_training_masks


def criterion(logits, prediction, classes, score, class_weights):
    return (nn.functional.cross_entropy(logits, classes, weight=class_weights)
            + 0.8 * nn.functional.smooth_l1_loss(prediction, score))


def train_one(model, train_loader, valid_loader, device: str, robust: bool,
              seed: int, checkpoint: Path, epochs: int = 35, lr: float = 8e-4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    counts = np.bincount(train_loader.dataset.data["cls"], minlength=3)
    weights = torch.tensor(np.sqrt(counts.sum() / (3 * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)
    best_score, best_epoch, history = -1e9, 0, []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            values = [item.to(device, non_blocking=True) for item in batch]
            text, audio, vision, text_mask, audio_mask, vision_mask, cls, score = values
            masks = [text_mask, audio_mask, vision_mask]
            if robust:
                masks = sample_training_masks(masks)
            optimizer.zero_grad(set_to_none=True)
            logits, prediction, _ = model(text, audio, vision, *masks)
            loss = criterion(logits, prediction, cls, score, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        clean = evaluate(model, valid_loader, device)
        if robust:
            damaged = [evaluate(model, valid_loader, device, modality, 0.3, "middle")
                       for modality in ("text", "audio", "vision")]
            f1 = (clean["macro_f1"] + sum(item["macro_f1"] for item in damaged)) / 4
            mae = (clean["mae"] + sum(item["mae"] for item in damaged)) / 4
            corr = (clean["pearson"] + sum(item["pearson"] for item in damaged)) / 4
        else:
            f1, mae, corr = clean["macro_f1"], clean["mae"], clean["pearson"]
        selection = f1 - 0.15 * mae + 0.05 * corr
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "clean": clean, "selection_score": float(selection)}
        history.append(record)
        print("robust" if robust else "baseline", seed, epoch,
              "loss", round(record["train_loss"], 3),
              "f1", round(clean["macro_f1"], 3),
              "mae", round(clean["mae"], 3),
              "select", round(selection, 3), flush=True)
        if selection > best_score + 1e-4:
            best_score, best_epoch = selection, epoch
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "robust": robust, "epoch": epoch}, checkpoint)
        if epoch - best_epoch >= 7:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    return {"best_epoch": best_epoch, "best_score": best_score, "history": history}
