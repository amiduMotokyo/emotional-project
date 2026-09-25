"""Create publication figures from measured Q3 validation and test explanations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODALITIES = ("text", "audio", "vision")
COLORS = {"text": "#355C7D", "audio": "#F67280", "vision": "#6C9A8B"}


def confusion_figure(summary: dict, output: Path) -> None:
    matrix = np.asarray(summary["confusion_matrix_true_by_pred"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=170)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    names = ["Negative", "Neutral", "Positive"]
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted polarity")
    ax.set_ylabel("True polarity")
    ax.set_title(f"Validation confusion matrix · ACC {summary['accuracy']:.3f}")
    for row in range(3):
        for column in range(3):
            count = matrix[row, column]
            rate = count / matrix[row].sum()
            ax.text(column, row, f"{count}\n{rate:.0%}", ha="center", va="center",
                    color="white" if count > matrix.max() * .55 else "#1C2A39",
                    fontsize=10)
    fig.colorbar(image, ax=ax, shrink=.72, label="Samples")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def modality_figure(summary: dict, records: list[dict], output: Path) -> None:
    valid_share = summary["mean_abs_effect_share"]
    test_share = {modality: float(np.mean([
        item["effects"][modality]["abs_share"] for item in records]))
        for modality in MODALITIES}
    gate = summary["mean_gate"]
    x = np.arange(3)
    width = .25
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=170)
    ax.bar(x - width, [valid_share[m] for m in MODALITIES], width,
           label="Validation |Δlogit| share", color="#355C7D")
    ax.bar(x, [test_share[m] for m in MODALITIES], width,
           label="Attachment 4 |Δlogit| share", color="#6C9A8B")
    ax.bar(x + width, [gate[m] for m in MODALITIES], width,
           label="Validation gate", color="#F5B971")
    ax.set_xticks(x, [name.title() for name in MODALITIES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean share / gate")
    ax.set_title("Measured modality effects and model gates")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def attachment4_figure(records: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=170)
    ids = [item["sample_id"] for item in records]
    x = np.arange(len(ids))
    bottom = np.zeros(len(ids))
    for modality in MODALITIES:
        values = np.asarray([item["effects"][modality]["abs_share"] for item in records])
        ax.bar(x, values, bottom=bottom, label=modality.title(),
               color=COLORS[modality], width=.78)
        bottom += values
    ax.set_xticks(x, ids)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Absolute deletion-effect share")
    ax.set_xlabel("Attachment-4 sample ID")
    ax.set_title("Three-modality effect decomposition for all 20 test samples")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0))
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def local_figure(record: dict, output: Path) -> None:
    modality = record["main_modality"]
    local = record["local"][modality]
    values = local["signed_logit_drop_by_position"]
    scores = np.asarray([np.nan if value is None else value for value in values])
    valid = np.flatnonzero(np.isfinite(scores))
    positive = np.maximum(np.nan_to_num(scores, nan=0), 0)
    negative = np.minimum(np.nan_to_num(scores, nan=0), 0)
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=170)
    ax.bar(np.arange(50), positive, color=COLORS[modality], width=.86,
           label="Supports predicted class")
    ax.bar(np.arange(50), negative, color="#C25553", width=.86,
           label="Opposes predicted class")
    window = local["top_window"]["aligned_positions_zero_based"]
    ax.axvspan(min(window) - .5, max(window) + .5, color="#F1CB65", alpha=.25,
               label="Selected evidence")
    ax.axhline(0, color="#22303C", linewidth=.7)
    ax.set_xlim(-.5, max(49.5, valid.max() + .5 if len(valid) else 49.5))
    ax.set_xlabel("Aligned position (zero based)")
    ax.set_ylabel("Fixed-class logit drop")
    ax.set_title(f"Sample {record['sample_id']} · {modality.title()} evidence · "
                 f"{record['prediction']['predicted_polarity']}")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="*", default=["02", "04", "10"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.results / "validation_summary.json").read_text(encoding="utf-8"))
    records = json.loads((args.results / "attachment4_explanations.json").read_text(encoding="utf-8"))
    confusion_figure(summary, args.output / "q3_validation_confusion.png")
    modality_figure(summary, records, args.output / "q3_modality_comparison.png")
    attachment4_figure(records, args.output / "q3_attachment4_modality_effects.png")
    indexed = {item["sample_id"]: item for item in records}
    for sample_id in args.samples:
        local_figure(indexed[sample_id], args.output / f"q3_local_{sample_id}.png")
    print(f"wrote {3 + len(args.samples)} figures to {args.output}", flush=True)


if __name__ == "__main__":
    main()
