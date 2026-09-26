"""Predict all aligned Attachment-3 samples with the selected Q2 package."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import LABELS, apply_audio_vision_scale, assemble_sample, encode_text
from C.src.reliability_imputation import ReliabilityImputationFusion
from B.src.fusion import Fusion
from B.src.temporal_fusion import TemporalFusion
from C.src.q2_protocol import coherent_score, support_mask


def infer(data_root: Path, package: Path, output: Path, device: str):
    session = ort.InferenceSession(str(package / "text_encoder_int8.onnx"),
                                   providers=["CPUExecutionProvider"])
    with np.load(package / "audio_vision_normalization.npz") as arrays:
        scale = {"audio": (arrays["audio_mean"], arrays["audio_std"]),
                 "vision": (arrays["vision_mean"], arrays["vision_std"])}
    saved = torch.load(package / "model.pt", map_location=device, weights_only=False)
    bias_path = package / "class_bias.json"
    bias_data = json.loads(bias_path.read_text()) if bias_path.exists() else {}
    class_bias = torch.tensor(bias_data.get("class_bias", [0., 0., 0.]), device=device)
    if class_bias.shape != (3,):
        raise ValueError("class_bias must contain three offsets")
    architecture = saved["architecture"]
    kwargs = saved.get('model_kwargs', {})
    cls = {"fusion": Fusion, "temporal": TemporalFusion, "impute": ReliabilityImputationFusion}[architecture]
    model = cls(**kwargs).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    directory = data_root / "附件3-模态缺失特征样本" / "对齐版本"
    rows = []
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as handle:
            source = pickle.load(handle)["test"]
        text = encode_text(session, source["text_bert"], "cpu", batch_size=1)
        sample = assemble_sample(source["text_bert"], text, source["audio"], source["vision"])
        apply_audio_vision_scale(sample, scale)
        support = support_mask(sample)
        inputs = [torch.from_numpy(sample[key].copy()).to(device)
                  for key in ("text", "audio", "vision", "tmask", "amask", "vmask")]
        with torch.inference_mode():
            if architecture == "temporal":
                logits, raw, gates = model(*inputs, torch.from_numpy(support.copy()).to(device))
            else:
                logits, raw, gates = model(*inputs)
            logits = logits + class_bias.to(dtype=logits.dtype)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            raw_score = float(raw.cpu().numpy()[0])
            gate = gates.cpu().numpy()[0]
        polarity = int(probs.argmax())
        score = float(coherent_score(np.array([polarity]), np.array([raw_score]))[0])
        denominator = max(1, int(support[0].sum()))
        rows.append({
            "sample_id": path.stem,
            "polarity": LABELS[polarity],
            "intensity": round(score, 6),
            "prob_negative": round(float(probs[0]), 6),
            "prob_neutral": round(float(probs[1]), 6),
            "prob_positive": round(float(probs[2]), 6),
            "text_observed_fraction": round(float((sample["tmask"][0] & support[0]).sum()) / denominator, 4),
            "audio_observed_fraction": round(float((sample["amask"][0] & support[0]).sum()) / denominator, 4),
            "vision_observed_fraction": round(float((sample["vmask"][0] & support[0]).sum()) / denominator, 4),
            "gate_text": round(float(gate[0]), 4),
            "gate_audio": round(float(gate[1]), 4),
            "gate_vision": round(float(gate[2]), 4),
        })
    if len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30:
        raise ValueError(f"expected 30 unique Attachment-3 samples, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} predictions with {architecture} to {output}", flush=True)


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
