"""Use the compact Q2 package to infer all aligned Attachment-3 samples."""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import LABELS, apply_audio_vision_scale, assemble_sample
from B.src.fusion import Fusion, predict_one


def infer(data_root: Path, package: Path, output: Path, device: str):
    session = ort.InferenceSession(str(package / "text_encoder_int8.onnx"),
                                   providers=["CPUExecutionProvider"])
    with np.load(package / "audio_vision_normalization.npz") as arrays:
        scale = {"audio": (arrays["audio_mean"], arrays["audio_std"]),
                 "vision": (arrays["vision_mean"], arrays["vision_std"])}
    model = Fusion().to(device)
    checkpoint = torch.load(package / "robust.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    attachment3 = data_root / "附件3-模态缺失特征样本" / "对齐版本"
    rows = []
    for path in sorted(attachment3.glob("*.pkl")):
        with path.open("rb") as handle:
            source = pickle.load(handle)["test"]
        bert = source["text_bert"]
        ids = np.rint(bert[:, 0]).astype(np.int64)
        attention = np.rint(bert[:, 1]).astype(np.int64)
        text = session.run(None, {
            "input_ids": ids,
            "attention_mask": attention,
            "token_type_ids": np.zeros_like(ids),
        })[0].astype(np.float16)
        sample = assemble_sample(bert, text, source["audio"], source["vision"])
        apply_audio_vision_scale(sample, scale)
        probabilities, score, gates = predict_one(model, sample, device)
        rows.append({
            "sample_id": path.stem,
            "polarity": LABELS[int(probabilities.argmax())],
            "intensity": round(score, 6),
            "prob_negative": round(float(probabilities[0]), 6),
            "prob_neutral": round(float(probabilities[1]), 6),
            "prob_positive": round(float(probabilities[2]), 6),
            "text_observed_fraction_of_50": round(float(sample["tmask"].mean()), 4),
            "audio_observed_fraction_of_50": round(float(sample["amask"].mean()), 4),
            "vision_observed_fraction_of_50": round(float(sample["vmask"].mean()), 4),
            "gate_text": round(float(gates[0]), 4),
            "gate_audio": round(float(gates[1]), 4),
            "gate_vision": round(float(gates[2]), 4),
        })
    if len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30:
        raise ValueError(f"expected 30 unique Attachment-3 samples, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} predictions to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    infer(args.data_root, args.package, args.output, args.device)


if __name__ == "__main__":
    main()
