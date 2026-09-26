"""Map aligned Attachment-4 evidence to original text, unaligned AV frames and media."""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np

MODALITY_RATE_HZ = {"audio": 20.0, "vision": 15.0}

def mp4_duration(path: Path) -> float | None:
    """Read container movie duration as a fallback; stream duration is preferred."""
    def walk(handle, end: int):
        while handle.tell() + 8 <= end:
            start = handle.tell()
            size, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                header = 16
            if size == 0:
                size = end - start
            if size < header or start + size > end:
                return None
            if kind == b"moov":
                value = walk(handle, start + size)
                if value is not None:
                    return value
            elif kind == b"mvhd":
                raw = handle.read(min(size - header, 40))
                if raw[0] == 0 and len(raw) >= 20:
                    scale, duration = struct.unpack_from(">II", raw, 12)
                elif raw[0] == 1 and len(raw) >= 32:
                    scale, duration = struct.unpack_from(">IQ", raw, 20)
                else:
                    return None
                return duration / scale if scale else None
            handle.seek(start + size)
        return None

    with path.open("rb") as handle:
        return walk(handle, path.stat().st_size)


def stream_metadata(path: Path, ffprobe: Path | None) -> dict:
    if ffprobe is None:
        return {}
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries",
         "stream=codec_type,duration,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    streams = json.loads(result.stdout).get("streams", [])
    return {item["codec_type"]: {
        "duration_seconds": float(item["duration"]),
        "frame_rate": item.get("r_frame_rate"),
    } for item in streams if item.get("duration")}


def exact_source_frames(aligned_vector: np.ndarray, source: np.ndarray) -> list[int]:
    """Return every frame with the same supplied feature vector; preserve ambiguity."""
    if not np.any(np.isfinite(aligned_vector) & (aligned_vector != 0)):
        return []
    candidates = np.flatnonzero(np.all(np.isclose(source, aligned_vector[None],
                                                  rtol=1e-6, atol=1e-6), axis=1))
    return candidates.astype(int).tolist()


def localize_window(modality: str, positions: list[int], aligned: dict,
                    unaligned: dict, video: Path, tokenizer,
                    streams: dict) -> dict:
    positions = sorted(set(int(position) for position in positions))
    result = {"modality": modality, "aligned_positions_zero_based": positions,
              "source_video_file": video.name}
    if not positions:
        return {**result, "mapping_status": "no_valid_positions"}
    if modality == "text":
        text = str(aligned["raw_text"])
        encoding = tokenizer(text, padding="max_length", truncation=True,
                             max_length=50, return_offsets_mapping=True)
        ids = np.rint(aligned["text_bert"][0]).astype(int).tolist()
        if ids != encoding["input_ids"]:
            return {**result, "mapping_status": "tokenizer_id_mismatch"}
        offsets = [encoding["offset_mapping"][index] for index in positions]
        offsets = [(start, end) for start, end in offsets if end > start]
        if not offsets:
            return {**result, "mapping_status": "no_text_char_span"}
        start, end = min(item[0] for item in offsets), max(item[1] for item in offsets)
        return {**result, "mapping_status": "exact_tokenizer_offset",
                "char_start_zero_based": start, "char_end_exclusive": end,
                "text_excerpt": text[start:end]}
    if modality not in MODALITY_RATE_HZ:
        raise ValueError(f"unknown modality: {modality}")
    source = np.asarray(unaligned[modality])
    frames = []
    ambiguous = False
    for position in positions:
        candidates = exact_source_frames(aligned[modality][position], source)
        if not candidates:
            return {**result, "mapping_status": "feature_frame_not_found",
                    "unmapped_position": position}
        frames.extend(candidates)
        ambiguous |= len(candidates) > 1
    first, last = min(frames), max(frames)
    rate = MODALITY_RATE_HZ[modality]
    start, end = first / rate, (last + 1) / rate
    stream_type = "audio" if modality == "audio" else "video"
    stream = streams.get(stream_type, {})
    stream_duration = stream.get("duration_seconds")
    status = "exact_feature_match_nominal_rate"
    if stream_duration is None or end > stream_duration + 0.2:
        status = "exact_feature_match_time_unverified"
    result.update({
        "mapping_status": status,
        "unaligned_frame_start_zero_based": first,
        "unaligned_frame_end_inclusive": last,
        "matched_frames_ambiguous": ambiguous,
        "nominal_feature_rate_hz": rate,
        "time_start_seconds": round(start, 3),
        "time_end_seconds": round(end, 3),
        "media_stream_duration_seconds": stream_duration,
    })
    if modality == "vision":
        result["keyframe_time_seconds"] = round((start + end) / 2, 3)
        numerator, denominator = (stream.get("frame_rate") or "30/1").split("/")
        video_rate = float(numerator) / max(float(denominator), 1.0)
        result["approx_video_frame_zero_based"] = round(result["keyframe_time_seconds"] * video_rate)
    return result
