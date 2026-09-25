from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
import whisper


def load_wav_as_float32(path: Path, target_rate: int = 16000) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if sample_rate != target_rate:
        raise ValueError(
            f"{path.name} has sample rate {sample_rate}; expected {target_rate}."
        )
    if sample_width != 2:
        raise ValueError(
            f"{path.name} has sample width {sample_width} bytes; expected 16-bit PCM."
        )

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default="en")
    parser.add_argument("--sample-list", default=None)
    parser.add_argument(
        "--use-ffmpeg",
        action="store_true",
        help="Pass the WAV path directly to Whisper. Requires ffmpeg.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model, device=device)

    if args.sample_list:
        sample_ids = [
            line.strip()
            for line in Path(args.sample_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        wav_files = [audio_dir / f"{sample_id}.wav" for sample_id in sample_ids]
    else:
        wav_files = sorted(audio_dir.glob("*.wav"))

    combined = []
    for index, wav_path in enumerate(wav_files, start=1):
        sample_id = wav_path.stem
        audio_input = (
            str(wav_path)
            if args.use_ffmpeg
            else load_wav_as_float32(wav_path)
        )
        result = model.transcribe(
            audio_input,
            language=args.language,
            word_timestamps=True,
            verbose=False,
        )
        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                item = {
                    "word": str(word.get("word", "")).strip(),
                    "start": round(float(word.get("start", 0.0)), 3),
                    "end": round(float(word.get("end", 0.0)), 3),
                    "probability": round(float(word.get("probability", 0.0)), 4),
                }
                words.append(item)
                combined.append(
                    {
                        "sample_id": sample_id,
                        **item,
                    }
                )
        (output_dir / f"{sample_id}_words.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{index}/{len(wav_files)}] {sample_id}: {len(words)} words", flush=True)

    with (output_dir / "whisper_words.jsonl").open("w", encoding="utf-8") as handle:
        for item in combined:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
