"""Fine-tune the last MiniLM layers for Q2 using Attachment-2 train labels only."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import apply_audio_vision_scale, fit_audio_vision_scale, load_npz
from B.src.fusion import Fusion
from B.src.temporal_fusion import TemporalFusion
from C.src.missingness import corrupt_masks, sample_training_masks
from C.src.q2_protocol import coherent_score
from C.src.reliability_imputation import ReliabilityImputationFusion


class RawTextDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data["cls"])

    def __getitem__(self, index):
        d = self.data
        return (
            torch.from_numpy(d["token_ids"][index].astype(np.int64, copy=True)),
            torch.from_numpy(d["attention"][index].astype(bool, copy=True)),
            torch.from_numpy(d["tmask"][index].astype(bool, copy=True)),
            torch.from_numpy(d["audio"][index].astype(np.float32, copy=True)),
            torch.from_numpy(d["vision"][index].astype(np.float32, copy=True)),
            torch.from_numpy(d["amask"][index].astype(bool, copy=True)),
            torch.from_numpy(d["vmask"][index].astype(bool, copy=True)),
            torch.tensor(int(d["cls"][index]), dtype=torch.long),
            torch.tensor(float(d["score"][index]), dtype=torch.float32),
        )


class FineTuneModel(nn.Module):
    def __init__(self, encoder_path: Path, unfreeze_layers: int = 2,
                 architecture: str = "fusion"):
        super().__init__()
        if architecture not in {"fusion", "temporal", "impute"}:
            raise ValueError(f"unknown fusion architecture: {architecture}")
        self.architecture = architecture
        self.encoder = AutoModel.from_pretrained(str(encoder_path), local_files_only=True)
        layers = self.encoder.encoder.layer
        if not 1 <= unfreeze_layers <= len(layers):
            raise ValueError(f"unfreeze_layers must be in [1, {len(layers)}]")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in layers[-unfreeze_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        if architecture == "temporal":
            self.fusion = TemporalFusion()
        elif architecture == "impute":
            self.fusion = ReliabilityImputationFusion()
        else:
            self.fusion = Fusion()

    def encode_text(self, input_ids, attention):
        return self.encoder(input_ids=input_ids,
                            attention_mask=attention.long(),
                            token_type_ids=torch.zeros_like(input_ids)).last_hidden_state

    def forward_detailed(self, input_ids, attention, audio, vision,
                         text_mask, audio_mask, vision_mask):
        hidden = self.encode_text(input_ids, attention)
        if self.architecture == "temporal":
            support = (attention.bool() & input_ids.ne(0) & input_ids.ne(101) &
                       input_ids.ne(102))
            return (*self.fusion(hidden, audio, vision, text_mask, audio_mask,
                                 vision_mask, support), None)
        if self.architecture == "impute":
            return self.fusion(hidden, audio, vision, text_mask, audio_mask, vision_mask)
        return (*self.fusion(hidden, audio, vision, text_mask, audio_mask, vision_mask), None)

    def forward(self, input_ids, attention, audio, vision,
                text_mask, audio_mask, vision_mask):
        return self.forward_detailed(input_ids, attention, audio, vision,
                                     text_mask, audio_mask, vision_mask)[:3]


def load_data(cache: Path):
    train = load_npz(cache / "train.npz")
    valid = load_npz(cache / "valid.npz")
    scale = fit_audio_vision_scale(train)
    apply_audio_vision_scale(train, scale)
    apply_audio_vision_scale(valid, scale)
    return train, valid


def forward_corrupted(model, batch, device: str, mode: str | None = None,
                      rng: random.Random | None = None,
                      train_geometry: str = "contiguous",
                      geometry: str = "contiguous",
                      return_aux_loss: bool = False):
    values = [value.to(device, non_blocking=True) for value in batch]
    ids, attention, tmask, audio, vision, amask, vmask, cls, score = values
    originals = [tmask.detach().cpu(), amask.detach().cpu(), vmask.detach().cpu()]
    original_ids = ids
    original_audio = audio
    original_vision = vision
    if mode is None:
        masks = originals
    elif mode == "train_random":
        masks = sample_training_masks(originals, probability=0.3, rng=rng,
                                      geometry=train_geometry)
    else:
        masks = corrupt_masks(originals, mode, rate=0.3,
                              position="middle", rng=rng, geometry=geometry)
    masks = [mask.to(device) for mask in masks]
    newly_missing = [original.to(device) & ~current
                     for original, current in zip(originals, masks)]
    ids = ids.clone()
    ids[newly_missing[0]] = 100
    audio = audio * masks[1].unsqueeze(-1)
    vision = vision * masks[2].unsqueeze(-1)
    if return_aux_loss and model.architecture == "impute" and mode == "train_random":
        # A clean, stop-gradient teacher provides targets only within this train batch.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            clean_hidden = model.encode_text(original_ids, attention)
            teacher_logits, teacher_intensity, _, teacher_aux = model.fusion(
                clean_hidden, original_audio, original_vision, *[m.to(device) for m in originals]
            )
        if was_training:
            model.train()
        logits, intensity, gates, aux = model.forward_detailed(
            ids, attention, audio, vision, *masks
        )
        targets = [clean_hidden, original_audio, original_vision]
        reconstruction_losses = []
        for reconstruction, logvar, target, missing in zip(
                aux["reconstruction"], aux["log_variance"], targets, newly_missing):
            missing = missing.to(device).float()
            if missing.any():
                error = (reconstruction - target.detach()).square().mean(dim=-1)
                nll = 0.5 * (torch.exp(-logvar) * error + logvar)
                reconstruction_losses.append((nll * missing).sum() / missing.sum())
        reconstruction_loss = (torch.stack(reconstruction_losses).mean()
                               if reconstruction_losses else logits.new_zeros(()))
        consistency = nn.functional.kl_div(
            nn.functional.log_softmax(logits, dim=1),
            nn.functional.softmax(teacher_logits, dim=1), reduction="batchmean")
        representation_consistency = nn.functional.mse_loss(
            nn.functional.normalize(aux["fusion_representation"], dim=1),
            nn.functional.normalize(teacher_aux["fusion_representation"], dim=1))
        intensity_consistency = nn.functional.smooth_l1_loss(intensity, teacher_intensity)
        aux_loss = (0.10 * reconstruction_loss + 0.10 * consistency +
                    0.03 * representation_consistency + 0.03 * intensity_consistency)
        return logits, intensity, gates, cls, score, aux_loss
    logits, intensity, gates = model(ids, attention, audio, vision, *masks)
    if return_aux_loss:
        return logits, intensity, gates, cls, score, logits.new_zeros(())
    return logits, intensity, gates, cls, score


def evaluate(model, loader, device: str, mode: str | None = None,
             geometry: str = "contiguous") -> dict:
    model.eval()
    predicted, actual, raw_score, target_score = [], [], [], []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            rng = random.Random(20260924 + batch_index)
            logits, intensity, _, cls, score = forward_corrupted(
                model, batch, device, mode, rng, geometry=geometry)
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            actual.append(cls.cpu().numpy())
            raw_score.append(intensity.cpu().numpy())
            target_score.append(score.cpu().numpy())
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)
    raw_score = np.concatenate(raw_score)
    target_score = np.concatenate(target_score)
    served_score = coherent_score(predicted, raw_score)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "per_class_f1": f1_score(actual, predicted, labels=[0, 1, 2],
                                  average=None, zero_division=0).tolist(),
        "mae": float(mean_absolute_error(target_score, served_score)),
        "n": int(len(actual)),
    }


def selection_key(clean: dict, damaged: list[dict]) -> tuple[float, ...]:
    return (clean["accuracy"], float(np.mean([row["accuracy"] for row in damaged])),
            clean["macro_f1"])


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    train_data, valid_data = load_data(args.cache)
    train_loader = DataLoader(RawTextDataset(train_data), batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed),
                              pin_memory=args.device.startswith("cuda"))
    valid_loader = DataLoader(RawTextDataset(valid_data), batch_size=args.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=args.device.startswith("cuda"))
    model = FineTuneModel(args.encoder, args.unfreeze_layers,
                          args.architecture).to(args.device)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    fusion_params = list(model.fusion.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": fusion_params, "lr": args.fusion_lr},
    ], weight_decay=0.01)
    counts = np.bincount(train_data["cls"], minlength=3)
    if args.classification_weighting == "sqrt":
        class_weights = torch.tensor(np.sqrt(counts.sum() / (3 * counts)),
                                     dtype=torch.float32, device=args.device)
    else:
        class_weights = None
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"minilm_top{args.unfreeze_layers}_{args.seed}.pt"
    best_key = None
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_index, batch in enumerate(train_loader):
            rng = random.Random(args.seed + epoch * 100003 + batch_index)
            logits, intensity, _, cls, score, auxiliary_loss = forward_corrupted(
                model, batch, args.device, "train_random", rng,
                train_geometry=args.mask_geometry, return_aux_loss=True)
            loss = (nn.functional.cross_entropy(logits, cls, weight=class_weights)
                    + 0.8 * nn.functional.smooth_l1_loss(intensity, score)
                    + auxiliary_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean = evaluate(model, valid_loader, args.device)
        damaged = [
            {"mode": mode, "geometry": geometry,
             **evaluate(model, valid_loader, args.device, mode, geometry)}
            for geometry in ("contiguous", "pointwise")
            for mode in ("text", "audio", "vision")
        ]
        key = selection_key(clean, damaged)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "clean": clean, "damaged": damaged,
                  "selection_key": list(key)}
        history.append(record)
        print(args.seed, "epoch", epoch, "loss", round(record["train_loss"], 4),
              "clean_acc", round(clean["accuracy"], 4),
              "clean_f1", round(clean["macro_f1"], 4), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            torch.save({"encoder_state_dict": model.encoder.state_dict(),
                        "fusion_state_dict": model.fusion.state_dict(),
                        "seed": args.seed, "epoch": epoch,
                        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                        "unfreeze_layers": args.unfreeze_layers,
                        "architecture": args.architecture,
                        "mask_geometry": args.mask_geometry,
                        "encoder_lr": args.encoder_lr,
                        "fusion_lr": args.fusion_lr,
                        "training_corruption_probability": 0.3,
                        "selection_key": list(key)}, checkpoint)
        (output / "training_report.json").write_text(json.dumps({
            "encoder": str(args.encoder),
            "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
            "unfreeze_layers": args.unfreeze_layers,
            "architecture": args.architecture,
            "mask_geometry": args.mask_geometry,
            "encoder_lr": args.encoder_lr, "fusion_lr": args.fusion_lr,
            "classification_weighting": args.classification_weighting,
            "batch_size": args.batch_size, "epochs": epoch,
            "training_corruption_probability": 0.3,
            "auxiliary_loss_weights": {"reconstruction_nll": 0.10,
                                        "class_kl": 0.10,
                                        "representation_consistency": 0.03,
                                        "intensity_consistency": 0.03},
            "validation_never_backpropagated": True,
            "seed": args.seed, "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint),
            "best_selection_key": list(best_key), "history": history,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    report["elapsed_seconds"] = elapsed
    report["last_epoch"] = history[-1]
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--architecture", choices=("fusion", "temporal", "impute"),
                        default="fusion")
    parser.add_argument("--mask-geometry", choices=("contiguous", "mixed"),
                        default="contiguous")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--fusion-lr", type=float, default=2e-4)
    parser.add_argument("--classification-weighting", choices=("sqrt", "none"),
                        default="sqrt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
