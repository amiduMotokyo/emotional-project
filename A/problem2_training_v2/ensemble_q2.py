"""Ensemble three problem-2 models and calibrate the validation decision bias."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_lib


def load_models(paths: list[Path], device: torch.device):
    models = []
    configs = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = model_lib.MissingAwareFusion(config["hidden_dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
        configs.append(config)
    return models, configs


def ensemble_predict(
    models: list[torch.nn.Module],
    data: dict,
    device: torch.device,
    missing_mode: str = "none",
    rate: float = 0.3,
    position: str = "middle",
    seed: int = 9000,
):
    loader = DataLoader(
        model_lib.AlignmentDataset(data),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    model_probabilities = []
    model_raw_scores = []
    labels = []
    scores = []
    for model_index, model in enumerate(models):
        probabilities = []
        raw_scores = []
        rng = random.Random(seed + model_index * 1009)
        for batch in loader:
            text, audio, vision, tmask, amask, vmask, cls, score, _ = batch
            masks = [tmask.to(device), amask.to(device), vmask.to(device)]
            if missing_mode != "none":
                masks = model_lib.contiguous_missing_masks(
                    masks, missing_mode, rate, position, rng
                )
            with torch.inference_mode():
                logits, raw, _, _, _ = model(
                    text.to(device),
                    audio.to(device),
                    vision.to(device),
                    masks[0],
                    masks[1],
                    masks[2],
                )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            raw_scores.append(raw.cpu().numpy())
            if model_index == 0:
                labels.append(cls.numpy())
                scores.append(score.numpy())
        model_probabilities.append(np.concatenate(probabilities))
        model_raw_scores.append(np.concatenate(raw_scores))
    return (
        np.mean(model_probabilities, axis=0),
        np.mean(model_raw_scores, axis=0),
        np.concatenate(labels),
        np.concatenate(scores),
        model_probabilities,
        model_raw_scores,
    )


def calibrated_predict(
    probabilities: np.ndarray,
    bias: np.ndarray,
    raw_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.log(np.clip(probabilities, 1e-9, 1.0)) + bias
    adjusted = adjusted - adjusted.max(axis=1, keepdims=True)
    adjusted = np.exp(adjusted)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    predicted = adjusted.argmax(axis=1)
    final_scores = np.where(
        predicted == 1,
        0.0,
        np.where(predicted == 2, np.abs(raw_scores), -np.abs(raw_scores)),
    )
    return predicted, final_scores


def metric_dict(
    y_true: np.ndarray,
    y_score: np.ndarray,
    predicted: np.ndarray,
    final_score: np.ndarray,
) -> dict:
    pearson = (
        float(np.corrcoef(y_score, final_score)[0, 1])
        if np.std(final_score) > 1e-8
        else 0.0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "mae": float(mean_absolute_error(y_score, final_score)),
        "pearson": pearson,
        "per_class_f1": f1_score(
            y_true, predicted, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=[0, 1, 2]
        ).tolist(),
        "n": int(len(y_true)),
    }


def search_bias(
    probabilities: np.ndarray,
    labels: np.ndarray,
    raw_scores: np.ndarray,
) -> tuple[np.ndarray, dict]:
    best_bias = np.zeros(3, dtype=np.float32)
    best_metric = {"accuracy": -1.0, "macro_f1": -1.0}
    grid = np.arange(-0.35, 0.351, 0.025, dtype=np.float32)
    best_accuracy = -1.0
    best_macro = -1.0
    for bias_negative in grid:
        for bias_neutral in grid:
            bias = np.array([bias_negative, bias_neutral, 0.0], dtype=np.float32)
            predicted, final_score = calibrated_predict(
                probabilities, bias, raw_scores
            )
            accuracy = float(accuracy_score(labels, predicted))
            macro = float(
                f1_score(labels, predicted, average="macro", zero_division=0)
            )
            if accuracy > best_accuracy + 1e-12 or (
                abs(accuracy - best_accuracy) <= 1e-12 and macro > best_macro
            ):
                best_accuracy = accuracy
                best_macro = macro
                best_bias = bias
                best_metric = {"accuracy": accuracy, "macro_f1": macro}
    return best_bias, best_metric


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion(
    path: Path,
    labels: np.ndarray,
    predicted: np.ndarray,
    title: str,
) -> None:
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks([0, 1, 2], model_lib.LABELS)
    axis.set_yticks([0, 1, 2], model_lib.LABELS)
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
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_regression(
    path: Path,
    y_true: np.ndarray,
    predicted: np.ndarray,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(y_true, predicted, alpha=0.45, s=18)
    lower = min(float(y_true.min()), float(predicted.min()))
    upper = max(float(y_true.max()), float(predicted.max()))
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.set_xlabel("真实强度")
    axis.set_ylabel("预测强度")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_missing_rate(
    path: Path,
    rows: list[dict],
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for mode in model_lib.MODES:
        selected = [row for row in rows if row["mode"] == mode and row["position"] == "middle"]
        selected.sort(key=lambda row: row["rate"])
        axis.plot(
            [row["rate"] for row in selected],
            [row["macro_f1"] for row in selected],
            marker="o",
            label=mode,
        )
    axis.set_xlabel("缺失比例")
    axis.set_ylabel("Macro-F1")
    axis.set_title("不同模态缺失比例下的验证集性能")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmaps(path: Path, rows: list[dict]) -> None:
    for mode in ("text", "audio", "vision"):
        rates = list(model_lib.RATES)
        positions = list(model_lib.POSITIONS)
        matrix = np.zeros((len(positions), len(rates)))
        for row in rows:
            if row["mode"] == mode:
                matrix[
                    positions.index(row["position"]), rates.index(row["rate"])
                ] = row["macro_f1"]
        fig, axis = plt.subplots(figsize=(6, 4.2))
        image = axis.imshow(matrix, cmap="viridis")
        fig.colorbar(image, ax=axis, label="Macro-F1")
        axis.set_xticks(range(3), [f"{rate:.0%}" for rate in rates])
        axis.set_yticks(range(3), positions)
        axis.set_xlabel("缺失比例")
        axis.set_ylabel("缺失位置")
        axis.set_title(f"{mode}模态缺失位置与比例")
        fig.tight_layout()
        fig.savefig(path / f"缺失热力图_{mode}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"E:\E题\E题数据\附件2-数据集特征文件\aligned_50.pkl"),
    )
    parser.add_argument(
        "--seed42",
        type=Path,
        default=Path(
            r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training\models\best_model.pt"
        ),
    )
    parser.add_argument(
        "--seed43",
        type=Path,
        default=Path(
            r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed43\models\best_model.pt"
        ),
    )
    parser.add_argument(
        "--seed44",
        type=Path,
        default=Path(
            r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed44\models\best_model.pt"
        ),
    )
    parser.add_argument(
        "--seed45",
        type=Path,
        default=Path(
            r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed45\models\best_model.pt"
        ),
    )
    parser.add_argument(
        "--seed46",
        type=Path,
        default=Path(
            r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed46\models\best_model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    reports = args.output / "reports"
    figures = args.output / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    raw = model_lib.load_aligned(args.data)
    train = model_lib.prepare_split(raw["train"])
    valid = model_lib.prepare_split(raw["valid"])
    test = model_lib.prepare_split(raw["test"])
    scale = model_lib.fit_scale(train)
    model_lib.apply_scale(valid, scale)
    model_lib.apply_scale(test, scale)

    model_paths = [args.seed42, args.seed43, args.seed44, args.seed45, args.seed46]
    seed_labels = [42, 43, 44, 45, 46]
    models, configs = load_models(model_paths, device)
    print("loaded models", len(models), "device", device, flush=True)

    valid_probs, valid_raw, valid_labels, valid_scores, seed_probs, seed_raw = ensemble_predict(
        models, valid, device, seed=9000
    )
    bias, bias_metrics = search_bias(valid_probs, valid_labels, valid_raw)
    valid_pred, valid_final = calibrated_predict(valid_probs, bias, valid_raw)
    valid_metrics = metric_dict(valid_labels, valid_scores, valid_pred, valid_final)

    test_probs, test_raw, test_labels, test_scores, _, _ = ensemble_predict(
        models, test, device, seed=9000
    )
    test_pred, test_final = calibrated_predict(test_probs, bias, test_raw)
    test_metrics = metric_dict(test_labels, test_scores, test_pred, test_final)

    valid_rows = []
    for index in range(len(valid_labels)):
        valid_rows.append(
            {
                "sample_id": valid["sample_id"][index],
                "true_polarity": model_lib.LABELS[int(valid_labels[index])],
                "predicted_polarity": model_lib.LABELS[int(valid_pred[index])],
                "true_intensity": float(valid_scores[index]),
                "predicted_intensity": float(valid_final[index]),
                "prob_negative": float(valid_probs[index, 0]),
                "prob_neutral": float(valid_probs[index, 1]),
                "prob_positive": float(valid_probs[index, 2]),
            }
        )
    test_rows = []
    for index in range(len(test_labels)):
        test_rows.append(
            {
                "sample_id": test["sample_id"][index],
                "true_polarity": model_lib.LABELS[int(test_labels[index])],
                "predicted_polarity": model_lib.LABELS[int(test_pred[index])],
                "true_intensity": float(test_scores[index]),
                "predicted_intensity": float(test_final[index]),
                "prob_negative": float(test_probs[index, 0]),
                "prob_neutral": float(test_probs[index, 1]),
                "prob_positive": float(test_probs[index, 2]),
            }
        )
    write_predictions(reports / "集成验证集预测.csv", valid_rows)
    write_predictions(reports / "集成测试集预测.csv", test_rows)

    missing_valid = []
    missing_test = []
    for mode in model_lib.MODES:
        for rate in model_lib.RATES:
            for position in model_lib.POSITIONS:
                probs, raw_scores, labels, scores, _, _ = ensemble_predict(
                    models,
                    valid,
                    device,
                    missing_mode=mode,
                    rate=rate,
                    position=position,
                    seed=9200,
                )
                predicted, final = calibrated_predict(probs, bias, raw_scores)
                missing_valid.append(
                    {
                        "mode": mode,
                        "rate": rate,
                        "position": position,
                        **metric_dict(labels, scores, predicted, final),
                    }
                )
                probs, raw_scores, labels, scores, _, _ = ensemble_predict(
                    models,
                    test,
                    device,
                    missing_mode=mode,
                    rate=rate,
                    position=position,
                    seed=9200,
                )
                predicted, final = calibrated_predict(probs, bias, raw_scores)
                missing_test.append(
                    {
                        "mode": mode,
                        "rate": rate,
                        "position": position,
                        **metric_dict(labels, scores, predicted, final),
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
        "n",
    ]
    for name, rows in (
        ("缺失网格_验证集.csv", missing_valid),
        ("缺失网格_测试集.csv", missing_test),
    ):
        with (reports / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    seed_rows = []
    for index, (seed, path) in enumerate(zip(seed_labels, model_paths)):
        predicted = seed_probs[index].argmax(axis=1)
        final = np.where(
            predicted == 1,
            0.0,
            np.where(predicted == 2, np.abs(seed_raw[index]), -np.abs(seed_raw[index])),
        )
        metrics = metric_dict(valid_labels, valid_scores, predicted, final)
        seed_rows.append({"seed": seed, "path": str(path), **metrics})
    with (reports / "单模型验证结果.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seed", "path", "accuracy", "macro_f1", "mae", "pearson", "n"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "seed": row["seed"],
                    "path": row["path"],
                    "accuracy": row["accuracy"],
                    "macro_f1": row["macro_f1"],
                    "mae": row["mae"],
                    "pearson": row["pearson"],
                    "n": row["n"],
                }
                for row in seed_rows
            ]
        )

    summary = {
        "protocol": "5-seed ensemble; bias calibrated on valid; test final hold-out",
        "split_sizes": {"train": len(train["cls"]), "valid": len(valid["cls"]), "test": len(test["cls"])},
        "calibrated_bias": bias.tolist(),
        "validation": valid_metrics,
        "test": test_metrics,
        "single_models": seed_rows,
        "missing_grid_valid": missing_valid,
        "missing_grid_test": missing_test,
    }
    (reports / "集成结果.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "类别偏置.json").write_text(
        json.dumps(
            {
                "bias_negative_neutral_positive": bias.tolist(),
                "validation_selection_metrics": bias_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_confusion(
        figures / "验证集混淆矩阵.png",
        valid_labels,
        valid_pred,
        "五模型集成：验证集混淆矩阵",
    )
    save_confusion(
        figures / "测试集混淆矩阵.png",
        test_labels,
        test_pred,
        "五模型集成：测试集混淆矩阵",
    )
    save_regression(
        figures / "验证集强度回归.png",
        valid_scores,
        valid_final,
        "五模型集成：验证集情感强度",
    )
    save_regression(
        figures / "测试集强度回归.png",
        test_scores,
        test_final,
        "五模型集成：测试集情感强度",
    )
    save_missing_rate(figures / "缺失比例性能曲线.png", missing_valid)
    save_heatmaps(figures, missing_valid)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [str(row["seed"]) for row in seed_rows] + ["五模型集成"],
        [row["accuracy"] for row in seed_rows] + [valid_metrics["accuracy"]],
        color=["#4472C4", "#70AD47", "#ED7D31", "#7B61FF"],
    )
    axis.axhline(0.65, color="red", linestyle="--", label="65%目标")
    axis.set_ylabel("验证集Accuracy")
    axis.set_title("单模型与五模型集成效果")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "单模型与集成对比.png", dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "bias": bias.tolist(),
                "validation": valid_metrics,
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
