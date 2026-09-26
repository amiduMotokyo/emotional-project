"""Write clean Attachment-2 validation probabilities for bias calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from C.scripts.finetune_q2_minilm import (FineTuneModel, RawTextDataset,
                                         forward_corrupted, load_data)


def predict(cache: Path, encoder: Path, checkpoint_path: Path,
            output: Path, device: str) -> list[dict]:
    _, valid = load_data(cache)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "fusion")
    model = FineTuneModel(encoder, checkpoint["unfreeze_layers"], architecture).to(device)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.fusion.load_state_dict(checkpoint["fusion_state_dict"])
    model.eval()
    loader = DataLoader(RawTextDataset(valid), batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=device.startswith("cuda"))
    rows = []
    with torch.inference_mode():
        for batch in loader:
            logits, _, _, classes, _ = forward_corrupted(model, batch, device)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            rows.extend({"true_class": int(label), "probabilities": values}
                        for label, values in zip(classes.cpu().tolist(), probabilities))
    if len(rows) != len(valid["cls"]):
        raise ValueError(f"expected {len(valid['cls'])} validation predictions, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = predict(args.cache, args.encoder, args.checkpoint,
                   args.output, args.device)
    print(f"wrote {len(rows)} clean validation probabilities to {args.output}", flush=True)


if __name__ == "__main__":
    main()
