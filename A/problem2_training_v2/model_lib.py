"""Train and evaluate a missing-aware multimodal model for problem 2.

The official competition split is respected:
  train -> model fitting
  valid -> model selection and early stopping
  test  -> final hold-out evaluation only

The input is Attachment 2 aligned_50.pkl. No external emotion dataset is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABELS = ("Negative", "Neutral", "Positive")
MODES = (
    "text",
    "audio",
    "vision",
    "text+audio",
    "text+vision",
    "audio+vision",
    "text+audio+vision",
)
RATES = (0.1, 0.3, 0.5)
POSITIONS = ("start", "middle", "end")


@dataclass
class Config:
    data_path: str
    output_dir: str
    seed: int = 42
    epochs: int = 35
    patience: int = 7
    batch_size: int = 32
    lr: float = 8e-4
    weight_decay: float = 0.01
    missing_probability: float = 0.8
    aux_loss_weight: float = 0.12
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    hidden_dim: int = 128
    num_workers: int = 0
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_aligned(path: Path) -> dict:
    import pickle

    # Some provided pickles were created with NumPy 2.x while the local
    # PyTorch environment may use NumPy 1.x. These aliases keep pickle loading
    # compatible without changing the data.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as handle:
        return pickle.load(handle)


def special_mask(text_bert: np.ndarray) -> np.ndarray:
    ids = np.rint(text_bert[:, 0]).astype(np.int64)
    attention = text_bert[:, 1] > 0
    return attention & (ids != 0) & (ids != 100) & (ids != 101) & (ids != 102)


def prepare_split(split: dict) -> dict:
    audio = np.nan_to_num(split["audio"].astype(np.float32))
    vision = np.nan_to_num(split["vision"].astype(np.float32))
    text = np.nan_to_num(split["text"].astype(np.float32))

    tmask = special_mask(split["text_bert"])
    amask = np.any(audio != 0, axis=-1)
    vmask = np.any(vision != 0, axis=-1)

    # Fall back to the text attention mask if a sequence would otherwise have
    # no valid token.
    fallback_text = split["text_bert"][:, 1] > 0
    empty_text = ~tmask.any(axis=1)
    tmask[empty_text] = fallback_text[empty_text]

    return {
        "text": text,
        "audio": audio,
        "vision": vision,
        "tmask": tmask,
        "amask": amask,
        "vmask": vmask,
        "cls": np.rint(split["classification_labels"]).astype(np.int64),
        "score": split["regression_labels"].astype(np.float32),
        "sample_id": np.asarray(split["id"], dtype=object),
        "raw_text": np.asarray(split["raw_text"], dtype=object),
    }


def fit_scale(train: dict) -> dict:
    scale = {}
    for name, mask_name in (("audio", "amask"), ("vision", "vmask")):
        values = train[name][train[mask_name]]
        mean = values.mean(axis=0).astype(np.float32)
        std = np.maximum(values.std(axis=0).astype(np.float32), 1e-4)
        scale[name] = (mean, std)
    return scale


def apply_scale(split: dict, scale: dict) -> None:
    for name, mask_name in (("audio", "amask"), ("vision", "vmask")):
        mean, std = scale[name]
        value = np.clip((split[name] - mean) / std, -5.0, 5.0)
        split[name] = np.where(
            split[mask_name][..., None], value, 0.0
        ).astype(np.float32)


class AlignmentDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self) -> int:
        return len(self.data["cls"])

    def __getitem__(self, index: int):
        data = self.data
        return (
            torch.from_numpy(data["text"][index].copy()),
            torch.from_numpy(data["audio"][index].copy()),
            torch.from_numpy(data["vision"][index].copy()),
            torch.from_numpy(data["tmask"][index].copy()),
            torch.from_numpy(data["amask"][index].copy()),
            torch.from_numpy(data["vmask"][index].copy()),
            torch.tensor(int(data["cls"][index]), dtype=torch.long),
            torch.tensor(float(data["score"][index]), dtype=torch.float32),
            index,
        )


def contiguous_missing_masks(
    masks: list[torch.Tensor],
    mode: str,
    rate: float,
    position: str,
    rng: random.Random | None = None,
) -> list[torch.Tensor]:
    """Mask one contiguous interval in the original 50-step coordinate."""
    result = [mask.clone() for mask in masks]
    if mode == "none":
        return result

    selected_names = mode.split("+")
    selected = [{"text": 0, "audio": 1, "vision": 2}[name] for name in selected_names]
    support = masks[0] | masks[1] | masks[2]
    for row in range(masks[0].shape[0]):
        indices = torch.nonzero(support[row], as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        left = int(indices[0])
        right = int(indices[-1]) + 1
        span = right - left
        width = min(span - 1, max(1, round(rate * span)))
        if position == "start":
            start = left
        elif position == "middle":
            start = left + (span - width) // 2
        elif position == "end":
            start = right - width
        elif position == "random":
            generator = rng or random
            start = generator.randint(left, right - width)
        else:
            raise ValueError(f"unknown missing position: {position}")
        for modality_index in selected:
            result[modality_index][row, start:start + width] = False
    return result


def training_missing_masks(
    masks: list[torch.Tensor],
    probability: float,
    rng: random.Random,
) -> list[torch.Tensor]:
    result = [mask.clone() for mask in masks]
    for row in range(masks[0].shape[0]):
        if rng.random() >= probability:
            continue
        mode = rng.choice(MODES)
        rate = rng.choice((0.1, 0.2, 0.3, 0.5))
        position = rng.choice(("start", "middle", "end", "random"))
        row_masks = [mask[row:row + 1] for mask in result]
        row_masks = contiguous_missing_masks(row_masks, mode, rate, position, rng)
        for modality_index, mask in enumerate(row_masks):
            result[modality_index][row:row + 1] = mask
    return result


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bool_mask = mask.bool()
        logits = self.score(values).squeeze(-1)
        logits = logits.masked_fill(~bool_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * bool_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (values * weights.unsqueeze(-1)).sum(dim=1)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.pool = MaskedAttentionPool(hidden_dim * 2)
        self.out_dim = hidden_dim * 2

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bool_mask = mask.bool()
        state = self.proj(values) * bool_mask.unsqueeze(-1)
        local = self.conv(state.transpose(1, 2)).transpose(1, 2)
        state = self.conv_norm(state + local)
        state = state * bool_mask.unsqueeze(-1)
        state, _ = self.gru(state)
        state = state * bool_mask.unsqueeze(-1)
        return self.pool(state, bool_mask)


class MissingAwareFusion(nn.Module):
    def __init__(self, hidden_dim: int = 96, dropout: float = 0.25):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                TemporalEncoder(768, hidden_dim, dropout),
                TemporalEncoder(74, hidden_dim, dropout),
                TemporalEncoder(35, hidden_dim, dropout),
            ]
        )
        dim = hidden_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(dim * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.cls_aux = nn.ModuleList([nn.Linear(dim, 3) for _ in range(3)])
        self.reg_aux = nn.ModuleList([nn.Linear(dim, 1) for _ in range(3)])
        self.head = nn.Sequential(
            nn.Linear(dim * 4 + 6, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    @staticmethod
    def _longest_gap(mask: torch.Tensor) -> torch.Tensor:
        missing = ~mask.bool()
        longest = torch.zeros(mask.shape[0], device=mask.device)
        current = torch.zeros_like(longest)
        for step in range(mask.shape[1]):
            current = torch.where(missing[:, step], current + 1, 0)
            longest = torch.maximum(longest, current)
        return longest / mask.shape[1]

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        tmask: torch.Tensor,
        amask: torch.Tensor,
        vmask: torch.Tensor,
    ):
        masks = [tmask.bool(), amask.bool(), vmask.bool()]
        values = [
            text * masks[0].unsqueeze(-1),
            audio * masks[1].unsqueeze(-1),
            vision * masks[2].unsqueeze(-1),
        ]
        pooled = [
            encoder(value, mask)
            for encoder, value, mask in zip(self.encoders, values, masks)
        ]
        rates = [
            mask.float().mean(dim=1, keepdim=True) for mask in masks
        ]
        present = torch.stack([mask.any(dim=1) for mask in masks], dim=1)
        gaps = [
            self._longest_gap(mask).unsqueeze(1) for mask in masks
        ]
        reliability = torch.cat(
            [*rates, *gaps],
            dim=1,
        )
        gate_logits = self.gate(torch.cat([*pooled, reliability], dim=1))
        gate_logits = gate_logits.masked_fill(~present, -1e4)
        gates = torch.softmax(gate_logits, dim=1) * present.float()
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = sum(
            gates[:, index:index + 1] * pooled[index] for index in range(3)
        )
        head_input = torch.cat(
            [*pooled, fused, reliability],
            dim=1,
        )
        output = self.head(head_input)
        logits = output[:, :3]
        intensity = 3.0 * torch.tanh(output[:, 3])
        aux_logits = [
            linear(pooled[index]) for index, linear in enumerate(self.cls_aux)
        ]
        aux_intensity = [
            linear(pooled[index]).squeeze(-1)
            for index, linear in enumerate(self.reg_aux)
        ]
        return logits, intensity, gates, aux_logits, aux_intensity


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3)
    weights = np.sqrt(len(labels) / (3.0 * np.maximum(counts, 1)))
    return torch.tensor(weights, dtype=torch.float32)


def coherent_scores(predicted_classes: np.ndarray, raw_scores: np.ndarray) -> np.ndarray:
    return np.where(
        predicted_classes == 1,
        0.0,
        np.where(predicted_classes == 2, np.abs(raw_scores), -np.abs(raw_scores)),
    )


def metric_dict(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pred_class: np.ndarray,
    raw_score: np.ndarray,
) -> dict:
    final_score = coherent_scores(pred_class, raw_score)
    pearson = (
        float(np.corrcoef(y_score, final_score)[0, 1])
        if np.std(final_score) > 1e-8
        else 0.0
    )
    raw_pearson = (
        float(np.corrcoef(y_score, raw_score)[0, 1])
        if np.std(raw_score) > 1e-8
        else 0.0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred_class)),
        "macro_f1": float(
            f1_score(y_true, pred_class, average="macro", zero_division=0)
        ),
        "mae": float(mean_absolute_error(y_score, final_score)),
        "pearson": pearson,
        "raw_mae": float(mean_absolute_error(y_score, raw_score)),
        "raw_pearson": raw_pearson,
        "per_class_f1": f1_score(
            y_true, pred_class, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, pred_class, labels=[0, 1, 2]
        ).tolist(),
        "n": int(len(y_true)),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    missing_mode: str = "none",
    rate: float = 0.3,
    position: str = "middle",
    fixed_seed: int = 2026,
    return_rows: bool = False,
) -> tuple[dict, list[dict]]:
    model.eval()
    all_true_cls, all_true_score = [], []
    all_logits, all_raw = [], []
    score_rows: list[dict] = []
    rng = random.Random(fixed_seed)
    data = loader.dataset.data
    for batch in loader:
        text, audio, vision, tmask, amask, vmask, cls, score, index = batch
        masks = [tmask.to(device), amask.to(device), vmask.to(device)]
        if missing_mode != "none":
            masks = contiguous_missing_masks(
                masks, missing_mode, rate, position, rng
            )
        logits, raw, gates, _, _ = model(
            text.to(device),
            audio.to(device),
            vision.to(device),
            masks[0],
            masks[1],
            masks[2],
        )
        all_true_cls.append(cls.numpy())
        all_true_score.append(score.numpy())
        all_logits.append(logits.cpu().numpy())
        all_raw.append(raw.cpu().numpy())
        if return_rows:
            gate_values = gates.cpu().numpy()
            for local, global_index in enumerate(index.numpy()):
                predicted_class = int(logits.argmax(1)[local].item())
                raw_intensity = float(raw[local].item())
                if predicted_class == 1:
                    predicted_intensity = 0.0
                elif predicted_class == 2:
                    predicted_intensity = abs(raw_intensity)
                else:
                    predicted_intensity = -abs(raw_intensity)
                score_rows.append(
                    {
                        "sample_id": str(data["sample_id"][global_index]),
                        "raw_text": str(data["raw_text"][global_index]),
                        "true_polarity": LABELS[int(cls[local].item())],
                        "predicted_polarity": LABELS[predicted_class],
                        "true_intensity": float(score[local].item()),
                        "raw_intensity": raw_intensity,
                        "predicted_intensity": predicted_intensity,
                        "prob_negative": float(
                            torch.softmax(logits, dim=1)[local, 0].cpu().item()
                        ),
                        "prob_neutral": float(
                            torch.softmax(logits, dim=1)[local, 1].cpu().item()
                        ),
                        "prob_positive": float(
                            torch.softmax(logits, dim=1)[local, 2].cpu().item()
                        ),
                        "gate_text": float(gate_values[local, 0]),
                        "gate_audio": float(gate_values[local, 1]),
                        "gate_vision": float(gate_values[local, 2]),
                    }
                )
    y_true = np.concatenate(all_true_cls).astype(int)
    y_score = np.concatenate(all_true_score)
    probabilities = torch.softmax(
        torch.from_numpy(np.concatenate(all_logits)), dim=1
    ).numpy()
    pred_class = probabilities.argmax(axis=1)
    raw_score = np.concatenate(all_raw)
    metrics = metric_dict(y_true, y_score, pred_class, raw_score)
    return metrics, score_rows


def selection_score(metrics: dict) -> float:
    return (
        metrics["macro_f1"]
        - 0.15 * metrics["mae"]
        + 0.05 * metrics["pearson"]
    )


def evaluate_selection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, dict[str, dict]]:
    clean, _ = evaluate(model, loader, device)
    cases = {}
    for mode in MODES:
        case, _ = evaluate(
            model,
            loader,
            device,
            missing_mode=mode,
            rate=0.3,
            position="middle",
            fixed_seed=2027,
        )
        cases[mode] = case
    mean_score = np.mean(
        [
            selection_score(clean),
            *[selection_score(value) for value in cases.values()],
        ]
    )
    return {"clean": clean, "cases": cases, "selection_score": float(mean_score)}, cases


def loss_fn(
    logits: torch.Tensor,
    raw_score: torch.Tensor,
    aux_logits: list[torch.Tensor],
    aux_raw: list[torch.Tensor],
    cls: torch.Tensor,
    score: torch.Tensor,
    weights: torch.Tensor,
    masks: list[torch.Tensor],
    label_smoothing: float,
    aux_weight: float,
) -> tuple[torch.Tensor, dict]:
    main_cls = nn.functional.cross_entropy(
        logits,
        cls,
        weight=weights,
        label_smoothing=label_smoothing,
    )
    main_reg = nn.functional.smooth_l1_loss(raw_score, score)
    aux_terms = []
    for index in range(3):
        present = masks[index].any(dim=1)
        if present.any():
            aux_cls = nn.functional.cross_entropy(
                aux_logits[index][present],
                cls[present],
                weight=weights,
                label_smoothing=label_smoothing,
            )
            aux_reg = nn.functional.smooth_l1_loss(
                aux_raw[index][present], score[present]
            )
            aux_terms.append(aux_cls + 0.5 * aux_reg)
    aux_loss = torch.stack(aux_terms).mean() if aux_terms else torch.zeros_like(main_cls)
    total = main_cls + 0.8 * main_reg + aux_weight * aux_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "cls_loss": float(main_cls.detach().cpu()),
        "reg_loss": float(main_reg.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
    }


def train_one(
    config: Config,
    train_data: dict,
    valid_data: dict,
    device: torch.device,
) -> dict:
    set_seed(config.seed)
    train_loader = DataLoader(
        AlignmentDataset(train_data),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        AlignmentDataset(valid_data),
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = MissingAwareFusion(config.hidden_dim).to(device)
    weights = class_weights(train_data["cls"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.1
    )
    best_score = -1e9
    best_epoch = -1
    best_state = None
    history = []
    rng = random.Random(config.seed)
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            text, audio, vision, tmask, amask, vmask, cls, score, _ = batch
            masks = training_missing_masks(
                [tmask.to(device), amask.to(device), vmask.to(device)],
                config.missing_probability,
                rng,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, raw, _, aux_logits, aux_raw = model(
                text.to(device),
                audio.to(device),
                vision.to(device),
                masks[0],
                masks[1],
                masks[2],
            )
            loss, loss_parts = loss_fn(
                logits,
                raw,
                aux_logits,
                aux_raw,
                cls.to(device),
                score.to(device),
                weights,
                masks,
                config.label_smoothing,
                config.aux_loss_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_losses.append(loss_parts)
        scheduler.step()

        selection, cases = evaluate_selection(model, valid_loader, device)
        if not np.isfinite(selection["selection_score"]):
            raise FloatingPointError("non-finite validation selection score")
        record = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "train_loss": float(np.mean([value["loss"] for value in epoch_losses])),
            "train_cls_loss": float(
                np.mean([value["cls_loss"] for value in epoch_losses])
            ),
            "train_reg_loss": float(
                np.mean([value["reg_loss"] for value in epoch_losses])
            ),
            "train_aux_loss": float(
                np.mean([value["aux_loss"] for value in epoch_losses])
            ),
            "selection_score": selection["selection_score"],
            "clean": selection["clean"],
            "missing": cases,
        }
        history.append(record)
        print(
            "epoch",
            epoch,
            "loss",
            round(record["train_loss"], 4),
            "clean_f1",
            round(record["clean"]["macro_f1"], 4),
            "clean_mae",
            round(record["clean"]["mae"], 4),
            "selection",
            round(record["selection_score"], 4),
            flush=True,
        )
        if selection["selection_score"] > best_score + 1e-4:
            best_score = selection["selection_score"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch - best_epoch >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("training failed without a valid checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
    }


def write_history(path: Path, history: list[dict]) -> None:
    fields = [
        "epoch",
        "seconds",
        "train_loss",
        "train_cls_loss",
        "train_reg_loss",
        "train_aux_loss",
        "selection_score",
        "clean_accuracy",
        "clean_macro_f1",
        "clean_mae",
        "clean_pearson",
        "missing30_mean_macro_f1",
        "missing30_mean_mae",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            missing_values = list(record["missing"].values())
            writer.writerow(
                {
                    "epoch": record["epoch"],
                    "seconds": record["seconds"],
                    "train_loss": record["train_loss"],
                    "train_cls_loss": record["train_cls_loss"],
                    "train_reg_loss": record["train_reg_loss"],
                    "train_aux_loss": record["train_aux_loss"],
                    "selection_score": record["selection_score"],
                    "clean_accuracy": record["clean"]["accuracy"],
                    "clean_macro_f1": record["clean"]["macro_f1"],
                    "clean_mae": record["clean"]["mae"],
                    "clean_pearson": record["clean"]["pearson"],
                    "missing30_mean_macro_f1": float(
                        np.mean([value["macro_f1"] for value in missing_values])
                    ),
                    "missing30_mean_mae": float(
                        np.mean([value["mae"] for value in missing_values])
                    ),
                }
            )


def write_predictions(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path, create) -> None:
    plt.figure(figsize=(8, 5.5))
    create()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_figures(
    output_dir: Path,
    history: list[dict],
    valid_predictions: list[dict],
    test_predictions: list[dict],
    missing_valid: list[dict],
    missing_test: list[dict],
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def training_curve():
        epochs = [row["epoch"] for row in history]
        plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
        plt.plot(
            epochs,
            [row["clean"]["macro_f1"] for row in history],
            label="valid clean macro-F1",
        )
        plt.plot(
            epochs,
            [row["selection_score"] for row in history],
            label="selection score",
        )
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        plt.title("Training and validation trajectory")

    save_figure(figure_dir / "fig_training_curves.png", training_curve)

    def confusion(rows: list[dict], title: str, filename: str):
        y_true = np.array([LABELS.index(row["true_polarity"]) for row in rows])
        y_pred = np.array([LABELS.index(row["predicted_polarity"]) for row in rows])
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig, axis = plt.subplots(figsize=(5.5, 4.5))
        image = axis.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=axis)
        axis.set_xticks([0, 1, 2], LABELS)
        axis.set_yticks([0, 1, 2], LABELS)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_title(title)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                )
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)

    confusion(valid_predictions, "Validation confusion matrix", "fig_valid_confusion.png")
    confusion(test_predictions, "Test confusion matrix", "fig_test_confusion.png")

    def regression(rows: list[dict], title: str, filename: str):
        true_values = np.array([float(row["true_intensity"]) for row in rows])
        pred_values = np.array([float(row["predicted_intensity"]) for row in rows])
        plt.scatter(true_values, pred_values, alpha=0.45, s=18)
        lower = min(true_values.min(), pred_values.min())
        upper = max(true_values.max(), pred_values.max())
        plt.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        plt.xlabel("true intensity")
        plt.ylabel("predicted intensity")
        plt.title(title)

    save_figure(
        figure_dir / "fig_valid_regression.png",
        lambda: regression(valid_predictions, "Validation intensity", "fig_valid_regression.png"),
    )
    save_figure(
        figure_dir / "fig_test_regression.png",
        lambda: regression(test_predictions, "Test intensity", "fig_test_regression.png"),
    )

    rate_rows = [row for row in missing_valid if row["position"] == "middle"]
    save_figure(
        figure_dir / "fig_missing_rate_macro_f1.png",
        lambda: (
            [
                plt.plot(
                    [row["rate"] for row in rate_rows if row["mode"] == mode],
                    [row["macro_f1"] for row in rate_rows if row["mode"] == mode],
                    marker="o",
                    label=mode,
                )
                for mode in MODES
            ],
            plt.xlabel("missing rate"),
            plt.ylabel("Macro-F1"),
            plt.legend(fontsize=7),
            plt.title("Validation performance under missingness"),
        ),
    )

    def heatmap(rows: list[dict], mode: str, filename: str):
        rates = list(RATES)
        positions = list(POSITIONS)
        matrix = np.zeros((len(positions), len(rates)))
        for row in rows:
            if row["mode"] == mode:
                matrix[positions.index(row["position"]), rates.index(row["rate"])] = row[
                    "macro_f1"
                ]
        fig, axis = plt.subplots(figsize=(5.8, 3.8))
        image = axis.imshow(matrix, cmap="viridis")
        fig.colorbar(image, ax=axis, label="Macro-F1")
        axis.set_xticks(range(3), [f"{rate:.0%}" for rate in rates])
        axis.set_yticks(range(3), positions)
        axis.set_xlabel("missing rate")
        axis.set_ylabel("missing position")
        axis.set_title(f"Validation grid: {mode}")
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)

    heatmap(missing_valid, "text", "fig_missing_heatmap_text.png")
    heatmap(missing_valid, "audio", "fig_missing_heatmap_audio.png")
    heatmap(missing_valid, "vision", "fig_missing_heatmap_vision.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"E:\E题\E题数据\附件2-数据集特征文件\aligned_50.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--missing-probability", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)
    (args.output / "reports").mkdir(exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device", device, flush=True)

    config = Config(
        data_path=str(args.data),
        output_dir=str(args.output),
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        missing_probability=args.missing_probability,
    )
    if args.quick:
        config.epochs = 3
        config.patience = 2
    with (args.output / "reports" / "config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)

    print("loading", args.data, flush=True)
    raw = load_aligned(args.data)
    train = prepare_split(raw["train"])
    valid = prepare_split(raw["valid"])
    test = prepare_split(raw["test"])
    scale = fit_scale(train)
    apply_scale(train, scale)
    apply_scale(valid, scale)
    apply_scale(test, scale)
    np.savez_compressed(
        args.output / "reports" / "normalization.npz",
        audio_mean=scale["audio"][0],
        audio_std=scale["audio"][1],
        vision_mean=scale["vision"][0],
        vision_std=scale["vision"][1],
    )

    result = train_one(config, train, valid, device)
    model = result["model"]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "labels": LABELS,
        },
        args.output / "models" / "best_model.pt",
    )
    write_history(args.output / "reports" / "history.csv", result["history"])

    valid_loader = DataLoader(
        AlignmentDataset(valid), batch_size=128, shuffle=False
    )
    test_loader = DataLoader(
        AlignmentDataset(test), batch_size=128, shuffle=False
    )
    valid_metrics, valid_rows = evaluate(
        model, valid_loader, device, return_rows=True
    )
    test_metrics, test_rows = evaluate(
        model, test_loader, device, return_rows=True
    )
    write_predictions(args.output / "reports" / "validation_predictions.csv", valid_rows)
    write_predictions(args.output / "reports" / "test_predictions.csv", test_rows)

    missing_valid = []
    missing_test = []
    for mode in MODES:
        for rate in RATES:
            for position in POSITIONS:
                case_valid, _ = evaluate(
                    model,
                    valid_loader,
                    device,
                    missing_mode=mode,
                    rate=rate,
                    position=position,
                    fixed_seed=3000,
                )
                case_test, _ = evaluate(
                    model,
                    test_loader,
                    device,
                    missing_mode=mode,
                    rate=rate,
                    position=position,
                    fixed_seed=3000,
                )
                missing_valid.append(
                    {
                        "mode": mode,
                        "rate": rate,
                        "position": position,
                        **case_valid,
                    }
                )
                missing_test.append(
                    {
                        "mode": mode,
                        "rate": rate,
                        "position": position,
                        **case_test,
                    }
                )
    fields = [
        "mode",
        "rate",
        "position",
        "accuracy",
        "macro_f1",
        "mae",
        "pearson",
        "raw_mae",
        "raw_pearson",
        "n",
    ]
    for name, rows in (
        ("missing_grid_valid.csv", missing_valid),
        ("missing_grid_test.csv", missing_test),
    ):
        with (args.output / "reports" / name).open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)

    metrics = {
        "data_path": str(args.data),
        "split_sizes": {
            "train": len(train["cls"]),
            "valid": len(valid["cls"]),
            "test": len(test["cls"]),
        },
        "protocol": "train fit; valid select/early-stop; test final hold-out",
        "device": str(device),
        "best_epoch": result["best_epoch"],
        "best_selection_score": result["best_score"],
        "validation": valid_metrics,
        "test": test_metrics,
        "missing_grid_valid": missing_valid,
        "missing_grid_test": missing_test,
    }
    with (args.output / "reports" / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    make_figures(
        args.output,
        result["history"],
        valid_rows,
        test_rows,
        missing_valid,
        missing_test,
    )
    print(
        json.dumps(
            {
                "best_epoch": result["best_epoch"],
                "valid": valid_metrics,
                "test": test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print("output", args.output, flush=True)


if __name__ == "__main__":
    main()
