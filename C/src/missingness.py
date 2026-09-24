"""Contiguous local modality-missing augmentation and evaluation masks (M5)."""
from __future__ import annotations

import random

import torch


MODALITY_INDEX = {"text": 0, "audio": 1, "vision": 2}


def corrupt_masks(masks: list[torch.Tensor], mode: str, rate: float = 0.3,
                  position: str = "random") -> list[torch.Tensor]:
    """Mask a contiguous interval of selected modalities without changing base padding masks."""
    result = [mask.clone() for mask in masks]
    batch_size, length = masks[0].shape
    if mode == "none":
        return result
    for row in range(batch_size):
        valid = int((masks[0][row] | masks[1][row] | masks[2][row]).sum().item())
        if valid < 2:
            continue
        width = max(1, round(rate * valid))
        if position == "start":
            start = 1
        elif position == "middle":
            start = max(1, (valid - width) // 2)
        elif position == "end":
            start = max(1, valid - width)
        else:
            start = random.randint(1, max(1, valid - width))
        start = min(start, length - 1)
        end = min(length, start + width)
        if mode == "random":
            count = random.choice((1, 1, 1, 2, 2, 3))
            selected = random.sample(range(3), count)
        else:
            selected = [MODALITY_INDEX[mode]]
        for index in selected:
            result[index][row, start:end] = False
    return result


def sample_training_masks(masks: list[torch.Tensor], probability: float = 0.8):
    """Apply 1-3 random modality masks to a batch with the configured probability."""
    if random.random() >= probability:
        return masks
    rate = random.choice((0.1, 0.2, 0.3, 0.5))
    return corrupt_masks(masks, "random", rate, "random")
