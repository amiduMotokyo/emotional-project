from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    align_frame_index,
    load_config,
    load_labels,
    prepare_output_dirs,
    resolve_config_paths,
    video_path,
)


def _ffmpeg_command(configured: Path, video: Path, wav: Path, sample_rate: int) -> list[str]:
    configured_text = str(configured)
    if configured.is_absolute() and configured.exists():
        executable = configured_text
    else:
        executable = shutil.which(configured_text) or configured_text
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav),
    ]


def _extract_smile_features(wav_path: Path, cfg: dict) -> np.ndarray:
    import opensmile

    audio_cfg = cfg["audio"]
    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, audio_cfg["feature_set"]),
        feature_level=getattr(opensmile.FeatureLevel, audio_cfg["feature_level"]),
    )
    values = smile.process_file(str(wav_path)).values.astype(np.float32)
    expected_dim = int(audio_cfg["feature_dim"])
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected audio feature shape {values.shape}; expected (*, {expected_dim})."
        )
    return values


def main(config_path: str | None = None, overwrite: bool = False) -> None:
    cfg = load_config(config_path)
    paths = resolve_config_paths(cfg)
    dirs = prepare_output_dirs(cfg)
    labels = load_labels(cfg)
    align_cfg = cfg["alignment"]
    audio_cfg = cfg["audio"]

    records: list[dict] = []
    failures: list[dict] = []
    for index, row in labels.iterrows():
        sample_id = row["sample_id"]
        output_path = dirs["audio"] / f"{sample_id}.npy"
        video = video_path(paths["video_root"], row["video_id"], row["clip_id"])
        if output_path.exists() and not overwrite:
            records.append(
                {
                    "sample_id": sample_id,
                    "n_frames": None,
                    "n_dims": audio_cfg["feature_dim"],
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
            with tempfile.TemporaryDirectory(prefix="problem1_audio_") as tmp:
                wav_path = Path(tmp) / f"{sample_id}.wav"
                command = _ffmpeg_command(
                    paths["ffmpeg_exe"],
                    video,
                    wav_path,
                    int(audio_cfg["sample_rate"]),
                )
                subprocess.run(command, check=True, capture_output=True, text=True)
                features = _extract_smile_features(wav_path, cfg)

            aligned, valid_segments = align_frame_index(
                features,
                float(audio_cfg["frame_shift_ms"]) / 1000.0,
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
                    "duration_sec": round(features.shape[0] * audio_cfg["frame_shift_ms"] / 1000.0, 3),
                    "feature_file": output_path.relative_to(paths["project_root"]).as_posix(),
                    "status": "ok",
                }
            )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
        if (index + 1) % 20 == 0:
            print(f"Audio: processed {index + 1}/{len(labels)}")

    pd.DataFrame(records).to_csv(
        dirs["audio"] / "audio_aligned_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if failures:
        pd.DataFrame(failures).to_csv(
            dirs["logs"] / "audio_failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"Audio extraction complete: {len(records)} records, {len(failures)} failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config, args.overwrite)

