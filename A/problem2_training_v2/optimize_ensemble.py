"""Optimize ensemble weights and class bias on validation only."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_lib


def load_models(paths, device):
    models = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = model_lib.MissingAwareFusion(checkpoint["config"]["hidden_dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    return models


def model_outputs(model, data, device, missing_mode="none", rate=0.3, position="middle", seed=7000):
    loader = DataLoader(model_lib.AlignmentDataset(data), batch_size=128, shuffle=False)
    rng = random.Random(seed)
    probabilities = []
    raw_scores = []
    for batch in loader:
        text, audio, vision, tmask, amask, vmask, cls, score, _ = batch
        masks = [tmask.to(device), amask.to(device), vmask.to(device)]
        if missing_mode != "none":
            masks = model_lib.contiguous_missing_masks(masks, missing_mode, rate, position, rng)
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
        if not raw_scores:
            pass
    return np.concatenate(probabilities), np.concatenate(raw_scores)


def all_model_outputs(models, data, device, missing_mode="none", rate=0.3, position="middle", seed=7000):
    probabilities = []
    raw_scores = []
    for index, model in enumerate(models):
        probs, raw = model_outputs(
            model,
            data,
            device,
            missing_mode=missing_mode,
            rate=rate,
            position=position,
            seed=seed + index * 101,
        )
        probabilities.append(probs)
        raw_scores.append(raw)
    return np.stack(probabilities), np.stack(raw_scores)


def decode(probs_all, raw_all, weights, bias):
    probabilities = np.tensordot(weights, probs_all, axes=(0, 0))
    raw = np.tensordot(weights, raw_all, axes=(0, 0))
    adjusted = np.log(np.clip(probabilities, 1e-9, 1.0)) + bias
    adjusted -= adjusted.max(axis=1, keepdims=True)
    adjusted = np.exp(adjusted)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    predicted = adjusted.argmax(axis=1)
    final = np.where(predicted == 1, 0.0,
                     np.where(predicted == 2, np.abs(raw), -np.abs(raw)))
    return adjusted, predicted, final


def metric_dict(y, score, predicted, final):
    pearson = float(np.corrcoef(score, final)[0, 1]) if np.std(final) > 1e-8 else 0.0
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(score, final)),
        "pearson": pearson,
        "per_class_f1": f1_score(y, predicted, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "n": len(y),
    }


def unpack(params, n_models):
    weights = np.exp(params[:n_models] - params[:n_models].max())
    weights /= weights.sum()
    bias = np.array([params[n_models], params[n_models + 1], 0.0])
    return weights, bias


def objective(params, probs, raw, labels, n_models):
    weights, bias = unpack(params, n_models)
    _, predicted, _ = decode(probs, raw, weights, bias)
    accuracy = accuracy_score(labels, predicted)
    macro = f1_score(labels, predicted, average="macro", zero_division=0)
    # Accuracy is the primary target; macro-F1 is only a small tie-breaker.
    return -float(accuracy) - 0.0005 * float(macro)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(r"E:\E题\E题数据\附件2-数据集特征文件\aligned_50.pkl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model_paths = [
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed43\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed44\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed45\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed46\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed47\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed48\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed49\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed50\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed51\models\best_model.pt"),
    ]
    n_models = len(model_paths)
    models = load_models(model_paths, device)
    raw = model_lib.load_aligned(args.data)
    train = model_lib.prepare_split(raw["train"])
    valid = model_lib.prepare_split(raw["valid"])
    test = model_lib.prepare_split(raw["test"])
    scale = model_lib.fit_scale(train)
    model_lib.apply_scale(valid, scale)
    model_lib.apply_scale(test, scale)

    print("computing validation probabilities", flush=True)
    valid_probs, valid_raw = all_model_outputs(models, valid, device)
    valid_labels = valid["cls"].astype(int)
    valid_scores = valid["score"]
    print("optimizing weights and bias", flush=True)
    bounds = [(-3, 3)] * n_models + [(-0.5, 0.5), (-0.5, 0.5)]
    result = differential_evolution(
        lambda p: objective(p, valid_probs, valid_raw, valid_labels, n_models),
        bounds=bounds,
        seed=2026,
        maxiter=45,
        popsize=12,
        tol=1e-6,
        polish=False,
        workers=1,
    )
    local = minimize(
        lambda p: objective(p, valid_probs, valid_raw, valid_labels, n_models),
        result.x,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9},
    )
    best = local.x if local.fun <= result.fun else result.x
    weights, bias = unpack(best, n_models)
    valid_calibrated, valid_pred, valid_final = decode(valid_probs, valid_raw, weights, bias)
    valid_metrics = metric_dict(valid_labels, valid_scores, valid_pred, valid_final)

    print("computing test probabilities", flush=True)
    test_probs, test_raw = all_model_outputs(models, test, device)
    test_labels = test["cls"].astype(int)
    test_scores = test["score"]
    test_calibrated, test_pred, test_final = decode(test_probs, test_raw, weights, bias)
    test_metrics = metric_dict(test_labels, test_scores, test_pred, test_final)

    reports = args.output / "reports"
    summary = {
        "protocol": "weighted five-model ensemble; weights and bias tuned on valid; test untouched",
        "model_paths": [str(path) for path in model_paths],
        "ensemble_weights": weights.tolist(),
        "class_bias": bias.tolist(),
        "validation": valid_metrics,
        "test": test_metrics,
    }
    (reports / "加权集成结果.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    valid_rows = []
    for index in range(len(valid_labels)):
        valid_rows.append({
            "sample_id": valid["sample_id"][index],
            "true_polarity": model_lib.LABELS[int(valid_labels[index])],
            "predicted_polarity": model_lib.LABELS[int(valid_pred[index])],
            "true_intensity": float(valid_scores[index]),
            "predicted_intensity": float(valid_final[index]),
            "prob_negative": float(valid_calibrated[index, 0]),
            "prob_neutral": float(valid_calibrated[index, 1]),
            "prob_positive": float(valid_calibrated[index, 2]),
        })
    test_rows = []
    for index in range(len(test_labels)):
        test_rows.append({
            "sample_id": test["sample_id"][index],
            "true_polarity": model_lib.LABELS[int(test_labels[index])],
            "predicted_polarity": model_lib.LABELS[int(test_pred[index])],
            "true_intensity": float(test_scores[index]),
            "predicted_intensity": float(test_final[index]),
            "prob_negative": float(test_calibrated[index, 0]),
            "prob_neutral": float(test_calibrated[index, 1]),
            "prob_positive": float(test_calibrated[index, 2]),
        })
    write_csv(reports / "加权集成验证集预测.csv", valid_rows)
    write_csv(reports / "加权集成测试集预测.csv", test_rows)
    print(json.dumps({"weights": weights.tolist(), "bias": bias.tolist(),
                      "validation": valid_metrics, "test": test_metrics},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
