from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_features(root: Path) -> dict:
    with (root / "outputs" / "features" / "final_features.pkl").open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--output", default="validation/reproducibility")
    args = parser.parse_args()

    reference_path = Path(args.reference).resolve()
    rerun_path = Path(args.rerun).resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_features(reference_path)
    rerun = load_features(rerun_path)
    rows = []
    for key in ("audio", "vision", "text"):
        left = np.asarray(reference[key])
        right = np.asarray(rerun[key])
        same_shape = left.shape == right.shape
        max_abs = float(np.max(np.abs(left - right))) if same_shape else np.nan
        rows.append(
            {
                "modality": key,
                "reference_shape": str(left.shape),
                "rerun_shape": str(right.shape),
                "same_shape": same_shape,
                "max_abs_diff": max_abs,
                "mean_abs_diff": float(np.mean(np.abs(left - right)))
                if same_shape
                else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["id_order_match"] = reference["id"] == rerun["id"]
    for modality in ("audio", "vision", "text"):
        result.loc[result["modality"] == modality, "id_order_match"] = (
            reference["id"] == rerun["id"]
        )
    result.to_csv(
        output_dir / "reproducibility_diff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

