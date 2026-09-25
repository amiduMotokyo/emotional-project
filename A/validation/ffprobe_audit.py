from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


A_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A_ROOT))

from src.utils import load_config, load_labels, resolve_config_paths


def probe(path: Path, ffprobe: Path) -> dict:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration,start_time:stream=index,codec_type,start_time,duration,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def stream_value(probe_result: dict, codec_type: str, key: str) -> float | None:
    for stream in probe_result.get("streams", []):
        if stream.get("codec_type") == codec_type and stream.get(key) is not None:
            return float(stream[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument(
        "--audio-dir",
        default=r"E:\E题\E题数据\问题1\音频特征提取\音频文件",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_config_paths(cfg)
    labels = load_labels(cfg)
    output_root = paths["output_root"]
    output_dir = A_ROOT / "validation" / "stream_sync"
    output_dir.mkdir(parents=True, exist_ok=True)
    ffprobe = (
        Path(args.ffprobe)
        if args.ffprobe
        else paths["ffmpeg_exe"].with_name("ffprobe.exe")
    )
    if not ffprobe.exists():
        raise FileNotFoundError(f"ffprobe not found: {ffprobe}")

    summary = pd.read_csv(output_root / "summary" / "full_result_summary.csv")
    valid = summary.pivot(
        index="sample_id", columns="modality", values="valid_length"
    )
    rows = []
    for row in labels.itertuples(index=False):
        sample_id = row.sample_id
        video = paths["video_root"] / row.video_id / f"{row.clip_id}.mp4"
        wav = Path(args.audio_dir) / f"{sample_id}.wav"
        video_probe = probe(video, ffprobe)
        wav_probe = probe(wav, ffprobe) if wav.exists() else {}
        video_start = stream_value(video_probe, "video", "start_time")
        audio_start = stream_value(video_probe, "audio", "start_time")
        video_duration = stream_value(video_probe, "video", "duration")
        audio_duration = stream_value(video_probe, "audio", "duration")
        wav_duration = float(wav_probe.get("format", {}).get("duration", 0.0))
        expected_audio = (
            min(50, math.ceil(audio_duration / 0.5))
            if audio_duration is not None
            else None
        )
        expected_vision = (
            min(50, math.ceil(video_duration / 0.5))
            if video_duration is not None
            else None
        )
        rows.append(
            {
                "sample_id": sample_id,
                "video_start_sec": video_start,
                "audio_start_sec": audio_start,
                "start_offset_sec": (
                    None
                    if video_start is None or audio_start is None
                    else video_start - audio_start
                ),
                "video_duration_sec": video_duration,
                "audio_duration_sec": audio_duration,
                "wav_duration_sec": wav_duration,
                "duration_delta_sec": (
                    None
                    if audio_duration is None
                    else audio_duration - wav_duration
                ),
                "expected_audio_segments": expected_audio,
                "expected_vision_segments": expected_vision,
                "stored_audio_segments": int(valid.loc[sample_id, "audio"]),
                "stored_vision_segments": int(valid.loc[sample_id, "vision"]),
                "audio_rule_ok": (
                    expected_audio is not None
                    and int(valid.loc[sample_id, "audio"]) == expected_audio
                ),
                "vision_rule_ok": (
                    expected_vision is not None
                    and int(valid.loc[sample_id, "vision"]) == expected_vision
                ),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(
        output_dir / "stream_sync_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_row = {
        "samples": len(audit),
        "max_abs_start_offset_sec": audit["start_offset_sec"].abs().max(),
        "mean_abs_start_offset_sec": audit["start_offset_sec"].abs().mean(),
        "max_abs_duration_delta_sec": audit["duration_delta_sec"].abs().max(),
        "audio_rule_ok_rate": audit["audio_rule_ok"].mean(),
        "vision_rule_ok_rate": audit["vision_rule_ok"].mean(),
    }
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "stream_sync_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame([summary_row]).to_string(index=False))


if __name__ == "__main__":
    main()
