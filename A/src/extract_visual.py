from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    align_timestamps,
    load_config,
    load_labels,
    prepare_output_dirs,
    resolve_config_paths,
    video_path,
)


def _run_openface(executable: Path, video: Path, output_dir: Path) -> None:
    if not executable.exists():
        raise FileNotFoundError(f"OpenFace executable not found: {executable}")
    command = [
        str(executable),
        "-f",
        str(video),
        "-out_dir",
        str(output_dir),
        "-2Dfp",
        "-aus",
        "-pose",
        "-gaze",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _read_openface_features(csv_path: Path, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    if "timestamp" not in frame.columns:
        raise ValueError("OpenFace CSV does not contain a timestamp column.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"OpenFace CSV is missing columns: {missing}")
    return (
        frame["timestamp"].to_numpy(dtype=np.float64),
        frame[feature_columns].to_numpy(dtype=np.float32),
    )


def main(config_path: str | None = None, overwrite: bool = False) -> None:
    cfg = load_config(config_path)
    paths = resolve_config_paths(cfg)
    dirs = prepare_output_dirs(cfg)
    labels = load_labels(cfg)
    align_cfg = cfg["alignment"]
    visual_cfg = cfg["visual"]
    feature_columns = list(visual_cfg["au_cols"]) + list(visual_cfg["pose_cols"])

    records: list[dict] = []
    failures: list[dict] = []
    raw_root = dirs["work"] / "openface_csv"
    raw_root.mkdir(parents=True, exist_ok=True)

    for index, row in labels.iterrows():
        sample_id = row["sample_id"]
        output_path = dirs["visual"] / f"{sample_id}.npy"
        video = video_path(paths["video_root"], row["video_id"], row["clip_id"])
        if output_path.exists() and not overwrite:
            records.append(
                {
                    "sample_id": sample_id,
                    "n_frames": None,
                    "n_dims": visual_cfg["feature_dim"],
                    "valid_segments": None,
                    "duration_sec": None,
                    "feature_file": output_path.relative_to(paths["project_root"]).as_posix(),
                    "status": "existing",
                }
            )
            continue
        if not video.exists():
            failures.append({"sample_id": sample_id, "error": f"Missing video: {video}"})
            continue

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"openface_{sample_id}_", dir=raw_root
            ) as tmp:
                raw_dir = Path(tmp)
                _run_openface(paths["openface_exe"], video, raw_dir)
                csv_files = sorted(raw_dir.glob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError("OpenFace did not create a CSV file.")
                timestamps, features = _read_openface_features(
                    csv_files[0], feature_columns
                )
                aligned, valid_segments = align_timestamps(
                    timestamps,
                    features,
                    float(align_cfg["segment_len"]),
                    int(align_cfg["max_segments"]),
                )
                np.save(output_path, aligned)
                records.append(
                    {
                        "sample_id": sample_id,
                        "n_frames": int(features.shape[0]),
                        "n_dims": int(features.shape[1]),
                        "valid_segments": valid_segments,
                        "duration_sec": round(float(timestamps[-1]), 3),
                        "feature_file": output_path.relative_to(paths["project_root"]).as_posix(),
                        "status": "ok",
                    }
                )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
        if (index + 1) % 20 == 0:
            print(f"Visual: processed {index + 1}/{len(labels)}")

    pd.DataFrame(records).to_csv(
        dirs["visual"] / "visual_aligned_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if failures:
        pd.DataFrame(failures).to_csv(
            dirs["logs"] / "visual_failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"Visual extraction complete: {len(records)} records, {len(failures)} failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config, args.overwrite)

