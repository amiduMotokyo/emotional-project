"""Export a fine-tuned MiniLM checkpoint and create its dynamic-int8 ONNX form."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel


class EncoderOutput(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids).last_hidden_state


def export(encoder_path: Path, checkpoint_path: Path, output_dir: Path,
           onnxruntime_path: Path | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = AutoModel.from_pretrained(str(encoder_path), local_files_only=True,
                                        attn_implementation="eager")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model = EncoderOutput(encoder).eval()
    example_ids = torch.full((1, 50), 0, dtype=torch.long)
    example_ids[0, :3] = torch.tensor([101, 2057, 102])
    example_attention = torch.zeros((1, 50), dtype=torch.long)
    example_attention[0, :3] = 1
    example_types = torch.zeros_like(example_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    float_path = output_dir / "text_encoder_finetuned_fp32.onnx"
    int8_path = output_dir / "text_encoder_finetuned_int8.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example_ids, example_attention, example_types),
            str(float_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            dynamo=False,
        )
    if onnxruntime_path is not None:
        import sys
        sys.path.insert(0, str(onnxruntime_path))
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(float_path), str(int8_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    return {
        "source": str(checkpoint_path),
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "float_onnx": str(float_path),
        "int8_onnx": str(int8_path),
        "float_bytes": float_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "quantization": "ONNX Runtime dynamic QInt8 for MatMul/Gemm/Gather",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnxruntime-path", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.encoder, args.checkpoint, args.output,
                    args.onnxruntime_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
