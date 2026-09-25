"""Attachment-4 sample pairing and shared evidence-location interface."""
from __future__ import annotations

import pickle
from pathlib import Path

from A.src.evidence_locator import (exact_source_frames, localize_window,
                                    mp4_duration, stream_metadata)

ATTACHMENT4 = "附件4-可解释专项视频样本与特征文件"


def attachment4_directories(data_root: Path) -> tuple[Path, Path]:
    base = data_root / ATTACHMENT4 / ATTACHMENT4
    return base / "对齐版本", base / "未对齐版本"


def load_pair(data_root: Path, sample_id: str) -> tuple[dict, dict, Path]:
    aligned_dir, unaligned_dir = attachment4_directories(data_root)
    aligned_path = aligned_dir / f"{sample_id}.pkl"
    unaligned_path = unaligned_dir / f"{sample_id}.pkl"
    video = aligned_dir / "videos" / f"{sample_id}.mp4"
    with aligned_path.open("rb") as handle:
        aligned = pickle.load(handle)
    with unaligned_path.open("rb") as handle:
        unaligned = pickle.load(handle)
    if str(aligned["id"]) != sample_id or str(unaligned["id"]) != sample_id:
        raise ValueError(f"Attachment-4 identity mismatch: {sample_id}")
    if str(aligned["raw_text"]) != str(unaligned["raw_text"]):
        raise ValueError(f"Attachment-4 transcript mismatch: {sample_id}")
    if not video.exists():
        raise FileNotFoundError(video)
    return aligned, unaligned, video
