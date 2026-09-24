from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    load_config,
    load_labels,
    prepare_output_dirs,
    resolve_config_paths,
)


def _read_summary(
    path: Path, sample_column: str, fallback_sample_column: str | None = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    frame = pd.read_csv(path)
    if sample_column not in frame.columns and fallback_sample_column is not None:
        if fallback_sample_column not in frame.columns:
            raise ValueError(
                f"{path} does not contain {sample_column!r} or "
                f"{fallback_sample_column!r}."
            )
        frame[sample_column] = (
            frame[fallback_sample_column]
            .astype(str)
            .str.replace(r"\.wav$", "", regex=True)
        )
    if sample_column not in frame.columns:
        raise ValueError(f"{path} does not contain {sample_column!r}.")
    frame[sample_column] = frame[sample_column].astype(str)
    return frame


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    paths = resolve_config_paths(cfg)
    dirs = prepare_output_dirs(cfg)
    align_cfg = cfg["alignment"]
    labels = load_labels(cfg).set_index("sample_id")

    audio_summary = _read_summary(
        dirs["audio"] / "audio_aligned_summary.csv",
        "sample_id",
        fallback_sample_column="filename",
    )
    visual_summary = _read_summary(
        dirs["visual"] / "visual_aligned_summary.csv",
        "sample_id",
        fallback_sample_column="sample",
    )
    text_summary = _read_summary(
        dirs["text"] / "text_feature_summary.csv", "sample_id"
    )

    sample_ids = sorted(
        set(audio_summary["sample_id"])
        & set(visual_summary["sample_id"])
        & set(text_summary["sample_id"])
    )
    expected_ids = sorted(labels.index)
    if sample_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(sample_ids))
        extra = sorted(set(sample_ids) - set(expected_ids))
        raise ValueError(
            f"Sample coverage mismatch. Missing={missing[:10]}, extra={extra[:10]}"
        )

    audio_len_map = dict(
        zip(audio_summary["sample_id"], audio_summary["valid_segments"])
    )
    visual_len_map = dict(
        zip(visual_summary["sample_id"], visual_summary["valid_segments"])
    )
    text_len_map = dict(
        zip(text_summary["sample_id"], text_summary["valid_tokens"])
    )
    if "duration_sec" not in audio_summary.columns:
        if "original_frames" not in audio_summary.columns:
            raise ValueError("Audio summary lacks duration_sec and original_frames.")
        audio_summary["duration_sec"] = (
            audio_summary["original_frames"].astype(float)
            * float(cfg["audio"]["frame_shift_ms"])
            / 1000.0
        )
    if "duration_sec" not in visual_summary.columns:
        if "n_frames" not in visual_summary.columns:
            raise ValueError("Visual summary lacks duration_sec and n_frames.")
        visual_summary["duration_sec"] = (
            visual_summary["n_frames"].astype(float) / 30.0
        )
    audio_duration_map = dict(
        zip(audio_summary["sample_id"], audio_summary["duration_sec"])
    )
    visual_duration_map = dict(
        zip(visual_summary["sample_id"], visual_summary["duration_sec"])
    )

    audio_all: list[np.ndarray] = []
    visual_all: list[np.ndarray] = []
    text_all: list[np.ndarray] = []
    audio_lengths: list[int] = []
    visual_lengths: list[int] = []
    text_lengths: list[int] = []

    for sample_id in sample_ids:
        audio = np.load(dirs["audio"] / f"{sample_id}.npy")
        visual = np.load(dirs["visual"] / f"{sample_id}.npy")
        text = np.load(dirs["text"] / f"{sample_id}.npy")
        expected_shapes = {
            "audio": (int(align_cfg["max_segments"]), int(cfg["audio"]["feature_dim"])),
            "visual": (
                int(align_cfg["max_segments"]),
                int(cfg["visual"]["feature_dim"]),
            ),
            "text": (int(cfg["text"]["max_length"]), int(cfg["text"]["feature_dim"])),
        }
        for name, array in (("audio", audio), ("visual", visual), ("text", text)):
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{sample_id} {name} shape {array.shape} != "
                    f"{expected_shapes[name]}"
                )
        if np.isnan(audio).any() or np.isnan(visual).any() or np.isnan(text).any():
            raise ValueError(f"{sample_id} contains NaN values.")
        audio_all.append(audio.astype(np.float32, copy=False))
        visual_all.append(visual.astype(np.float32, copy=False))
        text_all.append(text.astype(np.float32, copy=False))
        audio_lengths.append(int(audio_len_map[sample_id]))
        visual_lengths.append(int(visual_len_map[sample_id]))
        text_lengths.append(int(text_len_map[sample_id]))

    final_features = {
        "audio": np.stack(audio_all),
        "text": np.stack(text_all),
        "vision": np.stack(visual_all),
        "id": sample_ids,
        "text_len": text_lengths,
        "audio_len": audio_lengths,
        "vision_len": visual_lengths,
    }
    final_path = dirs["features"] / "final_features.pkl"
    with final_path.open("wb") as handle:
        pickle.dump(final_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary_rows: list[dict] = []
    for index, sample_id in enumerate(sample_ids):
        label_row = labels.loc[sample_id]
        summary_rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "video_id": label_row["video_id"],
                    "clip_id": label_row["clip_id"],
                    "modality": "text",
                    "original_duration_or_length": len(label_row["text"]),
                    "unit": "characters",
                    "feature_dim": int(cfg["text"]["feature_dim"]),
                    "valid_length": text_lengths[index],
                    "aligned_granularity": "token_position",
                    "padding_rule": "BERT padding vector",
                },
                {
                    "sample_id": sample_id,
                    "video_id": label_row["video_id"],
                    "clip_id": label_row["clip_id"],
                    "modality": "audio",
                    "original_duration_or_length": round(
                        float(audio_duration_map[sample_id]), 3
                    ),
                    "unit": "seconds",
                    "feature_dim": int(cfg["audio"]["feature_dim"]),
                    "valid_length": audio_lengths[index],
                    "aligned_granularity": "0.5s",
                    "padding_rule": "zero padding",
                },
                {
                    "sample_id": sample_id,
                    "video_id": label_row["video_id"],
                    "clip_id": label_row["clip_id"],
                    "modality": "vision",
                    "original_duration_or_length": round(
                        float(visual_duration_map[sample_id]), 3
                    ),
                    "unit": "seconds",
                    "feature_dim": int(cfg["visual"]["feature_dim"]),
                    "valid_length": visual_lengths[index],
                    "aligned_granularity": "0.5s",
                    "padding_rule": "zero padding",
                },
            ]
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["sample_id", "modality"], kind="stable"
    )
    summary.to_csv(
        dirs["summary"] / "full_result_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segment_len = float(align_cfg["segment_len"])
    max_segments = int(align_cfg["max_segments"])
    time_rows: list[dict] = []
    token_rows: list[dict] = []
    tokenizer = None
    try:
        from transformers import BertTokenizer

        if paths["bert_path"].exists():
            tokenizer = BertTokenizer.from_pretrained(str(paths["bert_path"]))
    except ImportError:
        tokenizer = None

    for index, sample_id in enumerate(sample_ids):
        for modality, valid_length in (
            ("audio", audio_lengths[index]),
            ("vision", visual_lengths[index]),
        ):
            for position in range(max_segments):
                time_rows.append(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "position": position,
                        "start_sec": round(position * segment_len, 3),
                        "end_sec": round((position + 1) * segment_len, 3),
                        "is_valid": position < valid_length,
                    }
                )

        tokens: list[str]
        if tokenizer is not None:
            encoded = tokenizer(
                labels.loc[sample_id, "text"],
                max_length=max_segments,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        else:
            tokens = ["[UNKNOWN]"] * max_segments
        for position, token in enumerate(tokens):
            token_rows.append(
                {
                    "sample_id": sample_id,
                    "modality": "text",
                    "position": position,
                    "token": token,
                    "is_valid": position < text_lengths[index],
                }
            )

    pd.DataFrame(time_rows).to_csv(
        dirs["alignment"] / "alignment_mapping_time.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(token_rows).to_csv(
        dirs["alignment"] / "alignment_mapping_token.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"Export complete: {final_path.name}, {len(summary)} summary rows, "
        f"{len(time_rows)} time rows, {len(token_rows)} token rows."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
