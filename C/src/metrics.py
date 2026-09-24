"""Shared classification and regression evaluation for M7."""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from .missingness import corrupt_masks

SEED = 20260924


def evaluate(model, loader, device: str, mode: str = "none", rate: float = 0.3,
             position: str = "middle", seed: int = SEED) -> dict:
    model.eval()
    old_state = random.getstate()
    random.seed(seed)
    classes, actual, predicted, estimates = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            values = [item.to(device, non_blocking=True) for item in batch]
            text, audio, vision, text_mask, audio_mask, vision_mask, cls, score = values
            masks = corrupt_masks([text_mask, audio_mask, vision_mask], mode, rate, position)
            logits, output_score, _ = model(text, audio, vision, *masks)
            classes.extend(cls.cpu().numpy())
            actual.extend(score.cpu().numpy())
            predicted.extend(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())
            estimates.extend(output_score.cpu().numpy())
    random.setstate(old_state)
    cls = np.asarray(classes)
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    estimates = np.asarray(estimates)
    pearson = float(np.corrcoef(actual, estimates)[0, 1]) if np.std(estimates) > 1e-8 else 0.0
    return {
        "accuracy": float(accuracy_score(cls, predicted)),
        "macro_f1": float(f1_score(cls, predicted, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(actual, estimates)),
        "pearson": pearson,
        "per_class_f1": f1_score(cls, predicted, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        "n": int(len(cls)),
    }
