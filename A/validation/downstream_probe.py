from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


A_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A_ROOT))

from src.utils import load_config, load_labels, resolve_config_paths


def aggregate_modalities(data: dict) -> dict[str, np.ndarray]:
    features = {}
    for key, length_key in (
        ("text", "text_len"),
        ("audio", "audio_len"),
        ("vision", "vision_len"),
    ):
        arrays = np.asarray(data[key], dtype=np.float64)
        lengths = np.asarray(data[length_key], dtype=int)
        features[key] = np.stack(
            [arrays[i, : lengths[i]].mean(axis=0) for i in range(arrays.shape[0])]
        )
    features["concat"] = np.concatenate(
        [features["text"], features["audio"], features["vision"]], axis=1
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_config_paths(cfg)
    labels = load_labels(cfg).set_index("sample_id")
    output_dir = A_ROOT / "validation" / "downstream"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (paths["output_root"] / "features" / "final_features.pkl").open("rb") as handle:
        data = pickle.load(handle)

    features = aggregate_modalities(data)
    sample_ids = list(data["id"])
    y_class = labels.loc[sample_ids, "annotation"].to_numpy()
    y_reg = labels.loc[sample_ids, "label"].to_numpy(dtype=float)
    groups = labels.loc[sample_ids, "video_id"].to_numpy()
    splitter = GroupKFold(n_splits=args.folds)

    fold_rows = []
    summary_rows = []
    baseline_fold_rows = []
    for modality, feature in features.items():
        classifier = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=20.0)),
            ]
        )
        class_pred = cross_val_predict(
            classifier,
            feature,
            y_class,
            groups=groups,
            cv=splitter,
            method="predict",
        )
        reg_pred = cross_val_predict(
            regressor,
            feature,
            y_reg,
            groups=groups,
            cv=splitter,
            method="predict",
        )
        for fold_index, (_, test_index) in enumerate(
            splitter.split(feature, y_class, groups), start=1
        ):
            fold_rows.append(
                {
                    "modality": modality,
                    "fold": fold_index,
                    "test_samples": len(test_index),
                    "accuracy": accuracy_score(
                        y_class[test_index], class_pred[test_index]
                    ),
                    "macro_f1": f1_score(
                        y_class[test_index],
                        class_pred[test_index],
                        average="macro",
                        zero_division=0,
                    ),
                    "mae": mean_absolute_error(
                        y_reg[test_index], reg_pred[test_index]
                    ),
                    "pearson": np.corrcoef(
                        y_reg[test_index], reg_pred[test_index]
                    )[0, 1]
                    if np.std(reg_pred[test_index]) > 0
                    else np.nan,
                }
            )
        summary_rows.append(
            {
                "modality": modality,
                "accuracy": accuracy_score(y_class, class_pred),
                "macro_f1": f1_score(
                    y_class, class_pred, average="macro", zero_division=0
                ),
                "mae": mean_absolute_error(y_reg, reg_pred),
                "pearson": np.corrcoef(y_reg, reg_pred)[0, 1]
                if np.std(reg_pred) > 0
                else np.nan,
            }
        )

    from collections import Counter

    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(features["text"], y_class, groups), start=1
    ):
        majority = Counter(y_class[train_index]).most_common(1)[0][0]
        majority_pred = np.full(len(test_index), majority)
        mean_pred = np.full(len(test_index), float(np.mean(y_reg[train_index])))
        baseline_fold_rows.append(
            {
                "fold": fold_index,
                "majority_accuracy": accuracy_score(
                    y_class[test_index], majority_pred
                ),
                "majority_macro_f1": f1_score(
                    y_class[test_index],
                    majority_pred,
                    average="macro",
                    zero_division=0,
                ),
                "mean_mae": mean_absolute_error(y_reg[test_index], mean_pred),
                "mean_pearson": np.corrcoef(y_reg[test_index], mean_pred)[0, 1]
                if np.std(mean_pred) > 0 and np.std(y_reg[test_index]) > 0
                else np.nan,
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(
        output_dir / "downstream_probe_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        fold_frame.groupby("modality")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
        )
        .reset_index()
    )
    summary["modality"] = pd.Categorical(
        summary["modality"],
        categories=["text", "audio", "vision", "concat"],
        ordered=True,
    )
    summary = summary.sort_values("modality")
    summary.to_csv(
        output_dir / "downstream_probe_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline = pd.DataFrame(baseline_fold_rows)
    baseline.to_csv(
        output_dir / "downstream_baseline_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_summary = baseline[
        [
            "majority_accuracy",
            "majority_macro_f1",
            "mean_mae",
            "mean_pearson",
        ]
    ].agg(["mean", "std"]).T.reset_index()
    baseline_summary.columns = ["metric", "mean", "std"]
    baseline_summary.to_csv(
        output_dir / "downstream_baseline_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))
    print(baseline_summary.to_string(index=False))


if __name__ == "__main__":
    main()
