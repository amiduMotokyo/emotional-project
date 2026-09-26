"""Select a small shared class-logit bias with stratified nested validation."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> tuple[list[int], list[list[float]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    labels = [int(row["true_class"]) for row in rows]
    probabilities = [[float(value) for value in row["probabilities"]]
                     for row in rows]
    if any(len(values) != 3 for values in probabilities):
        raise ValueError(f"expected three class probabilities in {path}")
    return labels, probabilities


def stratified_folds(labels: list[int], folds: int, seed: int) -> list[list[int]]:
    result = [[] for _ in range(folds)]
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        random.Random(seed + label).shuffle(indices)
        for position, index in enumerate(indices):
            result[position % folds].append(index)
    return result


def prediction_table(labels: list[int], probabilities: list[list[float]],
                     candidates: list[tuple[float, float, float]]
                     ) -> dict[tuple[float, float, float], list[int]]:
    log_probabilities = [
        [math.log(max(value, 1e-12)) for value in row]
        for row in probabilities
    ]
    return {
        bias: [max(range(3), key=lambda cls: log_probabilities[i][cls] + bias[cls])
               for i in range(len(labels))]
        for bias in candidates
    }


def select_bias(training_indices: list[int], labels_by_seed: list[list[int]],
                predictions_by_seed: list[dict], candidates: list[tuple[float, float, float]]
                ) -> tuple[float, float, float]:
    scores = {}
    for bias in candidates:
        scores[bias] = sum(
            predictions[bias][index] == labels[index]
            for labels, predictions in zip(labels_by_seed, predictions_by_seed)
            for index in training_indices
        )
    best = max(scores.values())
    tied = [bias for bias, score in scores.items() if score == best]
    return min(tied, key=lambda bias: (sum(abs(x) for x in bias), bias))


def calibrate(prediction_files: list[Path], output: Path, folds: int,
              seed: int, levels: list[float]) -> dict:
    labels_by_seed = []
    probabilities_by_seed = []
    for path in prediction_files:
        labels, probabilities = read_rows(path)
        labels_by_seed.append(labels)
        probabilities_by_seed.append(probabilities)
    if not labels_by_seed or any(labels != labels_by_seed[0]
                                 for labels in labels_by_seed[1:]):
        raise ValueError("prediction files must contain the same validation rows and labels")
    labels = labels_by_seed[0]
    candidates = [(negative, neutral, 0.0)
                  for negative, neutral in itertools.product(levels, repeat=2)]
    predictions_by_seed = [prediction_table(targets, probabilities, candidates)
                            for targets, probabilities in
                            zip(labels_by_seed, probabilities_by_seed)]
    baseline_bias = (0.0, 0.0, 0.0)
    if baseline_bias not in candidates:
        candidates.append(baseline_bias)
        predictions_by_seed = [prediction_table(targets, probabilities, candidates)
                               for targets, probabilities in
                               zip(labels_by_seed, probabilities_by_seed)]

    fold_indices = stratified_folds(labels, folds, seed)
    all_indices = list(range(len(labels)))
    nested_predictions = [[None] * len(labels) for _ in labels_by_seed]
    fold_biases = []
    for test_indices in fold_indices:
        test_set = set(test_indices)
        train_indices = [index for index in all_indices if index not in test_set]
        bias = select_bias(train_indices, labels_by_seed,
                           predictions_by_seed, candidates)
        fold_biases.append(bias)
        for seed_index, predictions in enumerate(predictions_by_seed):
            for index in test_indices:
                nested_predictions[seed_index][index] = predictions[bias][index]

    full_bias = select_bias(all_indices, labels_by_seed,
                            predictions_by_seed, candidates)
    per_seed = []
    for seed_index, (targets, predictions) in enumerate(
            zip(labels_by_seed, predictions_by_seed)):
        baseline_correct = sum(predictions[baseline_bias][i] == targets[i]
                               for i in all_indices)
        nested_correct = sum(nested_predictions[seed_index][i] == targets[i]
                             for i in all_indices)
        selected_correct = sum(predictions[full_bias][i] == targets[i]
                               for i in all_indices)
        per_seed.append({
            "prediction_file": prediction_files[seed_index].name,
            "baseline_correct": baseline_correct,
            "baseline_accuracy": baseline_correct / len(targets),
            "nested_correct": nested_correct,
            "nested_accuracy": nested_correct / len(targets),
            "selected_bias_correct": selected_correct,
            "selected_bias_accuracy": selected_correct / len(targets),
        })
    total = len(labels) * len(labels_by_seed)
    baseline_total = sum(row["baseline_correct"] for row in per_seed)
    nested_total = sum(row["nested_correct"] for row in per_seed)
    selected_total = sum(row["selected_bias_correct"] for row in per_seed)
    report = {
        "method": "5-fold stratified nested selection; one shared bias across all seeds",
        "selection_scope": "Attachment-2 validation labels only; no model gradient updates",
        "n_validation": len(labels),
        "n_seed_predictions": len(labels_by_seed),
        "folds": folds,
        "seed": seed,
        "candidate_levels": levels,
        "candidate_biases": len(candidates),
        "fold_selected_biases": [list(bias) for bias in fold_biases],
        "fold_bias_counts": {str(list(bias)): count
                              for bias, count in Counter(fold_biases).items()},
        "nested_cv_pooled_accuracy": nested_total / total,
        "baseline_pooled_accuracy": baseline_total / total,
        "nested_cv_delta_percentage_points": 100 * (nested_total - baseline_total) / total,
        "selected_shared_bias": list(full_bias),
        "selected_shared_bias_pooled_validation_accuracy": selected_total / total,
        "per_seed": per_seed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=(-0.2, -0.1, 0.0, 0.1, 0.2))
    args = parser.parse_args()
    report = calibrate(args.predictions, args.output, args.folds,
                       args.seed, args.levels)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
