"""Input-level local missingness protocol shared by Q2 training and evaluation."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import Dataset

from B.src.data import encode_text
from .missingness import corrupt_masks, sample_training_masks


SELECT_CASES = (("text", 0.3, "middle"),
                ("audio", 0.3, "middle"),
                ("vision", 0.3, "middle"))
MODES = ("text", "audio", "vision", "text+audio", "text+vision",
         "audio+vision", "text+audio+vision")
RATES = (0.1, 0.3, 0.5)
POSITIONS = ("start", "middle", "end")


def support_mask(data: dict) -> np.ndarray:
    ids = data["token_ids"]
    return (data["attention"].astype(bool) & (ids != 0) &
            (ids != 101) & (ids != 102))


def base_masks(data: dict) -> list[torch.Tensor]:
    return [torch.from_numpy(data[name].astype(bool).copy())
            for name in ("tmask", "amask", "vmask")]


def _encode_with_masks(session, data: dict, masks: np.ndarray) -> np.ndarray:
    """Replace newly lost tokens by [UNK] before the contextual encoder."""
    ids = data["token_ids"].astype(np.int64).copy()
    newly_missing = data["tmask"] & ~masks[:, 0, :]
    ids[newly_missing] = 100
    bert = np.stack([ids, data["attention"].astype(np.int64),
                     np.zeros_like(ids)], axis=1)
    # The deployed dynamic-int8 ONNX model is batch-sensitive: use the same
    # one-sample call shape as Attachment-3 inference for every split/view.
    return encode_text(session, bert, "cpu", batch_size=1)


def prepare_views(data: dict, session, directory: Path, views: int, seed: int) -> None:
    """Cache several independently sampled views; each preserves physical indices."""
    directory.mkdir(parents=True, exist_ok=True)
    shape = (views, len(data["cls"]), 50, 384)
    text = np.lib.format.open_memmap(directory / "train_text.npy", "w+",
                                     dtype=np.float16, shape=shape)
    masks_out = np.lib.format.open_memmap(directory / "train_masks.npy", "w+",
                                          dtype=np.bool_, shape=(views, len(data["cls"]), 3, 50))
    original = base_masks(data)
    for view in range(views):
        rng = random.Random(seed + 104729 * view)
        masks = sample_training_masks(original, probability=0.8, rng=rng)
        stacked = np.stack([item.numpy() for item in masks], axis=1)
        masks_out[view] = stacked
        text[view] = _encode_with_masks(session, data, stacked)
        text.flush()
        masks_out.flush()
        changed = int(np.count_nonzero(data["tmask"] & ~stacked[:, 0, :]))
        print(f"prepared train view {view + 1}/{views}; masked text positions {changed}", flush=True)


def case_name(mode: str, rate: float, position: str) -> str:
    return f"{mode.replace('+', '-')}_{round(rate * 100):02d}_{position}.npz"


def prepare_case(data: dict, session, directory: Path, mode: str,
                 rate: float, position: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / case_name(mode, rate, position)
    if path.exists():
        return path
    masks = corrupt_masks(base_masks(data), mode, rate, position,
                          rng=random.Random(20260924))
    stacked = np.stack([item.numpy() for item in masks], axis=1)
    text = _encode_with_masks(session, data, stacked)
    np.savez_compressed(path, text=text, masks=stacked)
    return path


def read_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        return archive["text"], archive["masks"]


class ViewDataset(Dataset):
    def __init__(self, base: dict, view_dir: Path | None = None):
        self.base = base
        self.support = support_mask(base)
        self.epoch = 0
        self.text = None if view_dir is None else np.load(view_dir / "train_text.npy", mmap_mode="r")
        self.masks = None if view_dir is None else np.load(view_dir / "train_masks.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.base["cls"])

    def __getitem__(self, index: int):
        if self.text is None:
            text = self.base["text"][index]
            masks = np.stack([self.base[name][index]
                              for name in ("tmask", "amask", "vmask")])
        else:
            view = (index * 7 + self.epoch) % len(self.text)
            text, masks = self.text[view, index], self.masks[view, index]
        audio = self.base["audio"][index] * masks[1, :, None]
        vision = self.base["vision"][index] * masks[2, :, None]
        return (torch.from_numpy(np.asarray(text, dtype=np.float32).copy()),
                torch.from_numpy(np.asarray(audio, dtype=np.float32).copy()),
                torch.from_numpy(np.asarray(vision, dtype=np.float32).copy()),
                *[torch.from_numpy(masks[i].copy()) for i in range(3)],
                torch.from_numpy(self.support[index].copy()),
                torch.tensor(int(self.base["cls"][index])),
                torch.tensor(float(self.base["score"][index]), dtype=torch.float32))


def forward_batch(model, batch, device: str):
    values = [item.to(device, non_blocking=True) for item in batch]
    text, audio, vision, tmask, amask, vmask, support, cls, score = values
    if getattr(model, "uses_support", False):
        logits, intensity, gates = model(text, audio, vision, tmask, amask, vmask, support)
    else:
        logits, intensity, gates = model(text, audio, vision, tmask, amask, vmask)
    return logits, intensity, gates, cls, score


def coherent_score(classes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map a dual-head result to the contest's sign and exact-neutral convention."""
    return np.where(classes == 1, 0.0,
                    np.where(classes == 2, np.abs(scores), -np.abs(scores)))


def evaluate_loader(model, loader, device: str, coherent: bool = False,
                    include_rows: bool = False) -> dict:
    model.eval()
    probs, outputs, labels, actual = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits, intensity, _, cls, score = forward_batch(model, batch, device)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            outputs.append(intensity.cpu().numpy())
            labels.append(cls.cpu().numpy())
            actual.append(score.cpu().numpy())
    probs = np.concatenate(probs)
    raw = np.concatenate(outputs)
    labels = np.concatenate(labels)
    actual = np.concatenate(actual)
    predictions = probs.argmax(axis=1)
    estimates = coherent_score(predictions, raw) if coherent else raw
    pearson = (float(np.corrcoef(actual, estimates)[0, 1])
               if np.std(estimates) > 1e-8 else 0.0)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(actual, estimates)),
        "pearson": pearson,
        "per_class_f1": f1_score(labels, predictions, labels=[0, 1, 2],
                                 average=None, zero_division=0).tolist(),
        "n": len(labels),
    }
    if include_rows:
        result["rows"] = {"probabilities": probs, "raw_score": raw,
                          "score": estimates, "predicted": predictions,
                          "actual": actual, "actual_class": labels}
    return result
