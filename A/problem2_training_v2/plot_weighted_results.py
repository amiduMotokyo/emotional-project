"""Plot Chinese figures for the ten-model weighted ensemble."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


LABELS = ("Negative", "Neutral", "Positive")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def plot_confusion(rows: list[dict], output: Path, title: str) -> None:
    y_true = np.array([LABELS.index(row["true_polarity"]) for row in rows])
    y_pred = np.array([LABELS.index(row["predicted_polarity"]) for row in rows])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks([0, 1, 2], LABELS)
    axis.set_yticks([0, 1, 2], LABELS)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
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
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_regression(rows: list[dict], output: Path, title: str) -> None:
    y_true = np.array([float(row["true_intensity"]) for row in rows])
    y_pred = np.array([float(row["predicted_intensity"]) for row in rows])
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(y_true, y_pred, alpha=0.45, s=18)
    lower = min(float(y_true.min()), float(y_pred.min()))
    upper = max(float(y_true.max()), float(y_pred.max()))
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.set_xlabel("真实强度")
    axis.set_ylabel("预测强度")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    reports = root / "reports"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    valid = read_rows(reports / "加权集成验证集预测.csv")
    test = read_rows(reports / "加权集成测试集预测.csv")
    result = json.loads((reports / "加权集成结果.json").read_text(encoding="utf-8"))

    plot_confusion(valid, figures / "加权集成_验证集混淆矩阵.png", "十模型加权集成：验证集混淆矩阵")
    plot_confusion(test, figures / "加权集成_测试集混淆矩阵.png", "十模型加权集成：测试集混淆矩阵")
    plot_regression(valid, figures / "加权集成_验证集强度回归.png", "十模型加权集成：验证集情感强度")
    plot_regression(test, figures / "加权集成_测试集强度回归.png", "十模型加权集成：测试集情感强度")

    weights = result["ensemble_weights"]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar([str(seed) for seed in range(42, 52)], weights, color="#4472C4")
    axis.set_xlabel("模型随机种子")
    axis.set_ylabel("valid优化权重")
    axis.set_title("十模型加权集成权重")
    fig.tight_layout()
    fig.savefig(figures / "十模型加权集成权重.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4.5))
    values = [result["validation"]["accuracy"], result["test"]["accuracy"]]
    axis.bar(["验证集", "测试集"], values, color=["#4472C4", "#70AD47"])
    axis.set_ylim(0.6, 0.7)
    axis.set_ylabel("Accuracy")
    axis.set_title("十模型加权集成最终Accuracy")
    for index, value in enumerate(values):
        axis.text(index, value + 0.001, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(figures / "十模型加权集成准确率.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
