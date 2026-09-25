"""Audit Attachment-4 identity, token mapping, and possible media time axes."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from B.src.attachment4 import attachment4_directories, mp4_duration, stream_metadata

def match_quality(aligned: np.ndarray, unaligned: np.ndarray, valid_length: int) -> dict:
    """Nearest unaligned frame is diagnostic only, not a time alignment."""
    source = aligned[np.any(np.isfinite(aligned) & (aligned != 0), axis=1)]
    target_indices = np.flatnonzero(np.any(np.isfinite(unaligned) & (unaligned != 0), axis=1))
    target = unaligned[target_indices]
    if len(source) == 0 or len(target) == 0:
        return {"n_aligned": int(len(source)), "median_relative_distance": None,
                "ordered_nearest_fraction": None,
                "declared_unaligned_length": int(valid_length),
                "actual_nonzero_unaligned_count": int(len(target))}
    source = np.nan_to_num(source.astype(np.float64))
    target = np.nan_to_num(target.astype(np.float64))
    scale = np.maximum(target.std(axis=0), 1e-3)
    norm_source = source / scale
    norm_target = target / scale
    distance = np.square(norm_source[:, None, :] - norm_target[None, :, :]).mean(axis=2)
    nearest = target_indices[distance.argmin(axis=1)]
    denominator = np.maximum(np.square(norm_source).mean(axis=1), 1e-6)
    return {
        "n_aligned": int(len(source)),
        "declared_unaligned_length": int(valid_length),
        "actual_nonzero_unaligned_count": int(len(target)),
        "median_relative_distance": float(np.median(np.sqrt(distance.min(axis=1) / denominator))),
        "ordered_nearest_fraction": float(np.mean(np.diff(nearest) >= 0)) if len(nearest) > 1 else 1.0,
        "nearest_indices": nearest.tolist(),
    }


def audit(data_root: Path, tokenizer_path: Path, ffprobe: Path | None = None) -> dict:
    aligned_dir, unaligned_dir = attachment4_directories(data_root)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                              use_fast=True)
    rows = []
    paths = sorted(aligned_dir.glob("*.pkl"))
    for path in paths:
        with path.open("rb") as handle:
            aligned = pickle.load(handle)
        with (unaligned_dir / path.name).open("rb") as handle:
            unaligned = pickle.load(handle)
        video = aligned_dir / "videos" / f"{path.stem}.mp4"
        duration = mp4_duration(video)
        streams = stream_metadata(video, ffprobe)
        raw_text = str(aligned["raw_text"])
        encoded = tokenizer(raw_text, padding="max_length", truncation=True,
                            max_length=50, return_offsets_mapping=True)
        token_ids = np.rint(aligned["text_bert"][0]).astype(int).tolist()
        row = {
            "sample_id": path.stem,
            "id_matches_file": str(aligned["id"]) == path.stem == str(unaligned["id"]),
            "same_transcript_both_versions": raw_text == str(unaligned["raw_text"]),
            "token_ids_match_raw_text": token_ids == encoded["input_ids"],
            "aligned_shapes": {name: list(aligned[name].shape)
                               for name in ("text_bert", "audio", "vision")},
            "unaligned_shapes": {name: list(unaligned[name].shape)
                                 for name in ("audio", "vision")},
            "duration_seconds": duration,
            "streams": streams,
            "audio_length": int(unaligned["audio_lengths"]),
            "vision_length": int(unaligned["vision_lengths"]),
        }
        if duration:
            row["audio_frames_per_second"] = row["audio_length"] / duration
            row["vision_frames_per_second"] = row["vision_length"] / duration
        if "audio" in streams:
            row["audio_nominal_20hz_gap_seconds"] = abs(
                row["audio_length"] / 20 - streams["audio"]["duration_seconds"])
        if "video" in streams and row["vision_length"] > 1:
            row["vision_nominal_15hz_gap_seconds"] = abs(
                row["vision_length"] / 15 - streams["video"]["duration_seconds"])
        row["audio_nearest_frame_diagnostic"] = match_quality(
            aligned["audio"], unaligned["audio"], row["audio_length"])
        row["vision_nearest_frame_diagnostic"] = match_quality(
            aligned["vision"], unaligned["vision"], row["vision_length"])
        rows.append(row)
    return {
        "n_aligned": len(paths),
        "n_videos": len(list((aligned_dir / "videos").glob("*.mp4"))),
        "all_id_match": all(row["id_matches_file"] for row in rows),
        "all_transcripts_match": all(row["same_transcript_both_versions"] for row in rows),
        "all_token_ids_match": all(row["token_ids_match_raw_text"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.tokenizer, args.ffprobe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
