"""Reproducible contiguous corruption in original aligned sequence coordinates."""
from __future__ import annotations

import random

import torch


MODALITY_INDEX = {"text": 0, "audio": 1, "vision": 2}
MODALITY_SETS = (
    (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2),
)


def _selected(mode: str, rng: random.Random | None = None) -> tuple[int, ...]:
    if mode == "random":
        return (rng or random).choice(MODALITY_SETS)
    names = mode.split("+")
    if not names or any(name not in MODALITY_INDEX for name in names):
        raise ValueError(f"unknown missing modality combination: {mode}")
    return tuple(sorted({MODALITY_INDEX[name] for name in names}))


def _corrupt_row(result: list[torch.Tensor], masks: list[torch.Tensor], row: int,
                 selected: tuple[int, ...], rate: float, position: str,
                 rng: random.Random | None = None) -> None:
    """Keep gaps at their original indices; the final valid index remains eligible."""
    support = masks[0][row] | masks[1][row] | masks[2][row]
    indices = torch.nonzero(support, as_tuple=False).flatten()
    if len(indices) < 2:
        return
    left, right = int(indices[0]), int(indices[-1]) + 1
    span = right - left
    width = min(span - 1, max(1, round(rate * span)))
    if position == "start":
        start = left
    elif position == "middle":
        start = left + (span - width) // 2
    elif position == "end":
        start = right - width
    elif position == "random":
        start = (rng or random).randint(left, right - width)
    else:
        raise ValueError(f"unknown missing position: {position}")
    for index in selected:
        result[index][row, start:start + width] = False


def _corrupt_row_pointwise(result: list[torch.Tensor], masks: list[torch.Tensor], row: int,
                           selected: tuple[int, ...], rate: float,
                           rng: random.Random) -> None:
    """Drop a fixed number of original support positions without moving indices."""
    support = masks[0][row] | masks[1][row] | masks[2][row]
    indices = torch.nonzero(support, as_tuple=False).flatten().tolist()
    if len(indices) < 2:
        return
    count = min(len(indices) - 1, max(1, round(rate * len(indices))))
    for position in rng.sample(indices, count):
        for index in selected:
            result[index][row, position] = False


def corrupt_masks(masks: list[torch.Tensor], mode: str, rate: float = 0.3,
                  position: str = "random", rng: random.Random | None = None,
                  geometry: str = "contiguous"
                  ) -> list[torch.Tensor]:
    """Mask aligned positions without shifting padding or internal holes."""
    result = [mask.clone() for mask in masks]
    if mode == "none":
        return result
    if not 0 < rate < 1:
        raise ValueError("rate must lie strictly between zero and one")
    if geometry not in {"contiguous", "pointwise"}:
        raise ValueError(f"unknown missingness geometry: {geometry}")
    source = rng or random
    for row in range(masks[0].shape[0]):
        selected = _selected(mode, source)
        if geometry == "pointwise":
            _corrupt_row_pointwise(result, masks, row, selected, rate, source)
        else:
            _corrupt_row(result, masks, row, selected, rate, position, source)
    return result


def sample_training_masks(masks: list[torch.Tensor], probability: float = 0.8,
                          rng: random.Random | None = None,
                          geometry: str = "contiguous"):
    """Sample clean/corrupted status, modality subset, rate, and gap geometry."""
    if geometry not in {"contiguous", "mixed"}:
        raise ValueError(f"unknown training gap geometry: {geometry}")
    source = rng or random
    result = [mask.clone() for mask in masks]
    for row in range(masks[0].shape[0]):
        if source.random() >= probability:
            continue
        selected = _selected("random", source)
        rate = source.choice((0.1, 0.2, 0.3, 0.5))
        if geometry == "mixed" and source.random() < 0.5:
            _corrupt_row_pointwise(result, masks, row, selected, rate, source)
        else:
            _corrupt_row(result, masks, row, selected, rate, "random", source)
    return result
