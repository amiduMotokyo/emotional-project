"""Produce Q2 validation diagnostics and figures from the selected deployable model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import LABELS
from C.scripts.run_q2_corrected import load_checkpoint, load_data, loader_for_case
from C.src.q2_protocol import (MODES, POSITIONS, RATES, case_name, evaluate_loader,
                               support_mask)


def analyze(cache: Path, output: Path, device: str):
    _, valid, _ = load_data(cache)
    selection = json.loads((output / "selected_model.json").read_text(encoding="utf-8"))
    model = load_checkpoint(Path(selection["checkpoint"]), device)
    served_grid = []
    case_dir = output / "valid_cases"
    for mode in MODES:
        for rate in RATES:
            for position in POSITIONS:
                case = case_dir / case_name(mode, rate, position)
                if not case.exists():
                    raise FileNotFoundError(case)
                result = evaluate_loader(model, loader_for_case(valid, case),
                                         device, coherent=True)
                served_grid.append({"mode": mode, "rate": rate,
                                    "position": position, **result})
    (output / "served_grid.json").write_text(
        json.dumps(served_grid, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "served_grid.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("mode", "rate", "position", "accuracy",
                         "macro_f1", "mae", "pearson", "n"))
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in served_grid])
    result = evaluate_loader(model, loader_for_case(valid, None), device,
                             coherent=True, include_rows=True)
    detail = result.pop("rows")
    predicted, actual_class = detail["predicted"], detail["actual_class"]
    estimates, actual = detail["score"], detail["actual"]
    support = support_mask(valid)
    rows = []
    for index in range(len(actual)):
        rows.append({
            "row_index": index,
            "actual_class": LABELS[int(actual_class[index])],
            "predicted_class": LABELS[int(predicted[index])],
            "actual_intensity": round(float(actual[index]), 6),
            "predicted_intensity": round(float(estimates[index]), 6),
            "absolute_error": round(float(abs(actual[index] - estimates[index])), 6),
            "support_length": int(support[index].sum()),
            "text_observed": int(valid["tmask"][index].sum()),
            "audio_observed": int(valid["amask"][index].sum()),
            "vision_observed": int(valid["vmask"][index].sum()),
        })
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix = confusion_matrix(actual_class, predicted, labels=[0, 1, 2])
    grouped = {}
    for cls, label in enumerate(LABELS):
        subset = actual_class == cls
        grouped[label] = {"n": int(subset.sum()),
                          "accuracy": float(np.mean(predicted[subset] == cls)),
                          "mae": float(np.abs(actual[subset] - estimates[subset]).mean())}
    lengths = support.sum(axis=1)
    by_length = {}
    for name, subset in (("short_1_16", lengths <= 16),
                         ("medium_17_32", (lengths > 16) & (lengths <= 32)),
                         ("long_33_50", lengths > 32)):
        by_length[name] = {"n": int(subset.sum()),
                           "accuracy": float(np.mean(predicted[subset] == actual_class[subset])),
                           "mae": float(np.abs(actual[subset] - estimates[subset]).mean())}
    summary = {"metrics": result, "confusion_matrix": matrix.tolist(),
               "by_class": grouped, "by_length": by_length,
               "neutral_nonzero_predictions": int(np.sum((predicted == 1) & (estimates != 0))),
               "median_absolute_error": float(np.median(np.abs(actual - estimates))),
               "served_missing_grid_mean": {
                   metric: float(np.mean([row[metric] for row in served_grid]))
                   for metric in ("accuracy", "macro_f1", "mae", "pearson")},
               "served_missing_grid_worst_macro_f1": float(
                   min(row["macro_f1"] for row in served_grid))}
    (output / "error_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3),
           xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "fig_confusion.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(actual, estimates, s=10, alpha=.4)
    ax.plot([-3, 3], [-3, 3], color="#b64141")
    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1),
           xlabel="True intensity", ylabel="Predicted intensity")
    fig.tight_layout()
    fig.savefig(output / "fig_regression.png")
    plt.close(fig)
    grid = served_grid
    for metric in ("macro_f1", "mae"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in ("text", "audio", "vision"):
            values = [np.mean([row[metric] for row in grid
                               if row["mode"] == mode and row["rate"] == rate])
                      for rate in (.1, .3, .5)]
            ax.plot([.1, .3, .5], values, marker="o", label=mode)
        ax.set(xlabel="Fraction of original valid span", ylabel=metric,
               title=f"Continuous local missingness: {metric}")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f"fig_missing_{metric}.png")
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    analyze(args.cache, args.output, args.device)


if __name__ == "__main__":
    main()
