"""Export Q2 validation predictions, error analysis, and figures (M7)."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import LABELS, MultimodalDataset, apply_audio_vision_scale, load_npz
from B.src.fusion import Fusion


def analyze(data_root: Path, cache: Path, output: Path, device: str):
    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    valid = load_npz(cache / "valid.npz")
    with np.load(output / "audio_vision_normalization.npz") as arrays:
        scale = {"audio": (arrays["audio_mean"], arrays["audio_std"]),
                 "vision": (arrays["vision_mean"], arrays["vision_std"])}
    apply_audio_vision_scale(valid, scale)
    loader = DataLoader(MultimodalDataset(valid), batch_size=128, shuffle=False)
    model = Fusion().to(device)
    checkpoint = torch.load(output / "robust.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities, estimates, gate_values = [], [], []
    with torch.inference_mode():
        for batch in loader:
            values = [item.to(device) for item in batch]
            text, audio, vision, text_mask, audio_mask, vision_mask = values[:6]
            logits, score, gates = model(text, audio, vision, text_mask, audio_mask, vision_mask)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            estimates.append(score.cpu().numpy())
            gate_values.append(gates.cpu().numpy())
    probabilities = np.concatenate(probabilities)
    estimates = np.concatenate(estimates)
    gates = np.concatenate(gate_values)
    true_cls, true_score = valid["cls"], valid["score"]
    pred_cls = probabilities.argmax(axis=1)
    with (data_root / "附件2-数据集特征文件" / "aligned_50.pkl").open("rb") as handle:
        source = pickle.load(handle)["valid"]
    rows = []
    for index in range(len(true_cls)):
        rows.append({
            "sample_id": source["id"][index],
            "true_polarity": LABELS[int(true_cls[index])],
            "predicted_polarity": LABELS[int(pred_cls[index])],
            "true_intensity": round(float(true_score[index]), 6),
            "predicted_intensity": round(float(estimates[index]), 6),
            "absolute_error": round(float(abs(true_score[index] - estimates[index])), 6),
            "correct_polarity": int(true_cls[index] == pred_cls[index]),
            "valid_tokens": int(valid["attention"][index].sum()),
            "observed_text": int(valid["tmask"][index].sum()),
            "observed_audio": int(valid["amask"][index].sum()),
            "observed_vision": int(valid["vmask"][index].sum()),
            "gate_text": round(float(gates[index, 0]), 5),
            "gate_audio": round(float(gates[index, 1]), 5),
            "gate_vision": round(float(gates[index, 2]), 5),
            "raw_text": str(source["raw_text"][index]),
        })
    with (output / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix = confusion_matrix(true_cls, pred_cls, labels=[0, 1, 2])
    class_errors = {
        LABELS[label]: {
            "n": int((true_cls == label).sum()),
            "mae": float(np.abs(true_score[true_cls == label] - estimates[true_cls == label]).mean()),
            "accuracy": float(np.mean(pred_cls[true_cls == label] == label)),
        }
        for label in range(3)
    }
    lengths = valid["attention"].sum(axis=1)
    buckets = {}
    for name, mask in (("short_1_16", lengths <= 16),
                       ("medium_17_32", (lengths > 16) & (lengths <= 32)),
                       ("long_33_50", lengths > 32)):
        buckets[name] = {
            "n": int(mask.sum()),
            "mae": float(np.abs(true_score[mask] - estimates[mask]).mean()),
            "accuracy": float(np.mean(pred_cls[mask] == true_cls[mask])),
        }
    summary = {
        "confusion_matrix_rows_true_cols_predicted": matrix.tolist(),
        "class_errors": class_errors,
        "length_buckets": buckets,
        "mean_gate": dict(zip(("text", "audio", "vision"), gates.mean(axis=0).astype(float).tolist())),
        "median_absolute_error": float(np.median(np.abs(true_score - estimates))),
    }
    (output / "error_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 10})
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), LABELS, rotation=20)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation confusion matrix")
    for row in range(3):
        for col in range(3):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "fig_confusion.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(true_score, estimates, s=10, alpha=0.4, color="#3269a8")
    ax.plot([-3, 3], [-3, 3], color="#c44949", linewidth=1)
    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), xlabel="True intensity",
           ylabel="Predicted intensity", title="Validation intensity predictions")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "fig_regression.png")
    plt.close(fig)
    for metric, filename, ylabel in (("macro_f1", "fig_missing_f1.png", "Macro-F1"),
                                     ("mae", "fig_missing_mae.png", "MAE")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for modality, color in (("text", "#1c6eae"), ("audio", "#db7533"), ("vision", "#5c9952")):
            means = [np.mean([item[metric] for item in report["missing_grid"]
                              if item["modality"] == modality and item["rate"] == rate])
                     for rate in (0.1, 0.3, 0.5)]
            ax.plot([0.1, 0.3, 0.5], means, marker="o", linewidth=2, label=modality, color=color)
        ax.set(xlabel="Masked fraction of valid sequence", ylabel=ylabel,
               title=f"Validation {ylabel} under contiguous masking")
        ax.set_xticks([0.1, 0.3, 0.5])
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    analyze(args.data_root, args.cache, args.output, args.device)


if __name__ == "__main__":
    main()
