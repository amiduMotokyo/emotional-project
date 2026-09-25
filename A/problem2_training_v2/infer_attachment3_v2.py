"""Five-model ensemble inference for the aligned Attachment-3 samples."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers.models.bert.modeling_bert import BertModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_lib
from ensemble_q2 import calibrated_predict


def load_pickle(path: Path) -> dict:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as handle:
        return pickle.load(handle)


@torch.inference_mode()
def encode_text(model: BertModel, text_bert: np.ndarray, device: torch.device) -> np.ndarray:
    ids = np.rint(text_bert[:, 0]).astype(np.int64)
    attention = np.rint(text_bert[:, 1]).astype(np.int64)
    output = model(
        input_ids=torch.from_numpy(ids).to(device),
        attention_mask=torch.from_numpy(attention).to(device),
    ).last_hidden_state
    return output.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\E题\E题数据"))
    parser.add_argument("--bert", type=Path, default=Path(r"E:\E题\bert-base-uncased"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bias", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.bias is None:
        args.bias = args.output / "reports" / "类别偏置.json"
    device = torch.device(args.device)
    model_paths = [
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed43\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed44\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed45\models\best_model.pt"),
        Path(r"C:\Users\30741\Documents\Codex\2026-09-24\wo\outputs\problem2_training_v2\seed46\models\best_model.pt"),
    ]
    models = []
    for path in model_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = model_lib.MissingAwareFusion(checkpoint["config"]["hidden_dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    bert = BertModel.from_pretrained(args.bert, local_files_only=True).to(device)
    bert.eval()
    with np.load(args.output / "reports" / "normalization.npz") as arrays:
        scale = {
            "audio": (arrays["audio_mean"], arrays["audio_std"]),
            "vision": (arrays["vision_mean"], arrays["vision_std"]),
        }
    bias_data = json.loads(args.bias.read_text(encoding="utf-8"))
    bias = np.asarray(bias_data["bias_negative_neutral_positive"], dtype=np.float32)

    rows = []
    source_dir = args.data_root / "附件3-模态缺失特征样本" / "对齐版本"
    for path in sorted(source_dir.glob("*.pkl")):
        item = load_pickle(path)["test"]
        text = encode_text(bert, item["text_bert"], device)
        split = {
            "text": text,
            "audio": item["audio"].astype(np.float32),
            "vision": item["vision"].astype(np.float32),
            "text_bert": item["text_bert"].astype(np.int64),
            "classification_labels": np.zeros(len(text), dtype=np.float32),
            "regression_labels": np.zeros(len(text), dtype=np.float32),
            "id": [path.stem],
            "raw_text": [""],
        }
        sample = model_lib.prepare_split(split)
        for name, mask_name in (("audio", "amask"), ("vision", "vmask")):
            mean, std = scale[name]
            value = np.clip((sample[name] - mean) / std, -5.0, 5.0)
            sample[name] = np.where(
                sample[mask_name][..., None], value, 0.0
            ).astype(np.float32)
        probabilities = []
        raw_scores = []
        gates = []
        for model in models:
            with torch.inference_mode():
                logits, raw, gate, _, _ = model(
                    torch.from_numpy(sample["text"]).to(device),
                    torch.from_numpy(sample["audio"]).to(device),
                    torch.from_numpy(sample["vision"]).to(device),
                    torch.from_numpy(sample["tmask"]).to(device),
                    torch.from_numpy(sample["amask"]).to(device),
                    torch.from_numpy(sample["vmask"]).to(device),
                )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            raw_scores.append(raw.cpu().numpy())
            gates.append(gate.cpu().numpy())
        mean_probability = np.mean(probabilities, axis=0)
        mean_raw = np.mean(raw_scores, axis=0)
        mean_gate = np.mean(gates, axis=0)
        predicted, final_score = calibrated_predict(mean_probability, bias, mean_raw)
        polarity = int(predicted[0])
        rows.append(
            {
                "sample_id": path.stem,
                "polarity": model_lib.LABELS[polarity],
                "intensity": round(float(final_score[0]), 6),
                "raw_intensity": round(float(mean_raw[0]), 6),
                "prob_negative": round(float(mean_probability[0, 0]), 6),
                "prob_neutral": round(float(mean_probability[0, 1]), 6),
                "prob_positive": round(float(mean_probability[0, 2]), 6),
                "gate_text": round(float(mean_gate[0, 0]), 4),
                "gate_audio": round(float(mean_gate[0, 1]), 4),
                "gate_vision": round(float(mean_gate[0, 2]), 4),
            }
        )
    if len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30:
        raise ValueError("Attachment-3 must contain exactly 30 unique samples")
    output_path = args.output / "reports" / "附件3预测_五模型集成.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {label: sum(row["polarity"] == label for row in rows) for label in model_lib.LABELS}
    summary = {
        "样本数": len(rows),
        "极性数量": counts,
        "平均强度": float(np.mean([row["intensity"] for row in rows])),
        "平均门控": {
            "文本": float(np.mean([row["gate_text"] for row in rows])),
            "音频": float(np.mean([row["gate_audio"] for row in rows])),
            "视觉": float(np.mean([row["gate_vision"] for row in rows])),
        },
    }
    (args.output / "reports" / "附件3预测汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    figure_path = args.output / "figures" / "附件3预测分布.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(list(counts), list(counts.values()), color=["#4472C4", "#ED7D31", "#70AD47"])
    axes[0].set_title("附件3情感极性预测")
    axes[0].set_ylabel("样本数")
    axes[1].scatter(
        [row["gate_text"] for row in rows],
        [row["intensity"] for row in rows],
        c=[model_lib.LABELS.index(row["polarity"]) for row in rows],
        cmap="viridis",
        alpha=0.8,
    )
    axes[1].set_xlabel("文本模态门控")
    axes[1].set_ylabel("预测强度")
    axes[1].set_title("附件3预测与文本门控")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("wrote", output_path, flush=True)


if __name__ == "__main__":
    main()
