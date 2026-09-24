"""Aligned MOSEI data preparation and shared batch interface for B and C."""
from __future__ import annotations

import gc
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LABELS = ("Negative", "Neutral", "Positive")


def encode_text(model, text_bert: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """Encode supplied BERT token IDs with the frozen local MiniLM or ONNX model."""
    ids = np.rint(text_bert[:, 0]).astype(np.int64)
    attention = np.rint(text_bert[:, 1]).astype(np.int64)
    encoded = []
    if hasattr(model, "run"):
        for start in range(0, len(ids), batch_size):
            x, mask = ids[start:start + batch_size], attention[start:start + batch_size]
            value = model.run(None, {
                "input_ids": x,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(x),
            })[0]
            encoded.append(value.astype(np.float16))
    else:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                x = torch.from_numpy(ids[start:start + batch_size]).to(device)
                mask = torch.from_numpy(attention[start:start + batch_size]).to(device)
                value = model(input_ids=x, attention_mask=mask).last_hidden_state
                encoded.append(value.float().cpu().numpy().astype(np.float16))
    return np.concatenate(encoded, axis=0)


def assemble_sample(text_bert: np.ndarray, text: np.ndarray, audio: np.ndarray,
                    vision: np.ndarray, classes=None, scores=None) -> dict:
    """Create the common aligned batch schema used by the fusion model."""
    token_ids = np.rint(text_bert[:, 0]).astype(np.int32)
    attention = text_bert[:, 1] > 0
    text_mask = attention & (token_ids != 101) & (token_ids != 102) & (token_ids != 100)
    audio = np.nan_to_num(audio.astype(np.float32))
    vision = np.nan_to_num(vision.astype(np.float32))
    audio_mask = attention & np.any(audio != 0, axis=-1)
    vision_mask = attention & np.any(vision != 0, axis=-1)
    result = {
        "text": text, "audio": audio, "vision": vision,
        "tmask": text_mask, "amask": audio_mask, "vmask": vision_mask,
        "token_ids": token_ids, "attention": attention,
    }
    if classes is not None:
        result["cls"] = np.asarray(classes, np.int64)
        result["score"] = np.asarray(scores, np.float32)
    return result


def prepare_cache(data_root: Path, text_model: Path, cache: Path, device: str,
                  onnx_model: Path | None = None) -> int:
    """Encode aligned train/valid and Attachment 3; return test sample count."""
    cache.mkdir(parents=True, exist_ok=True)
    source_path = data_root / "附件2-数据集特征文件" / "aligned_50.pkl"
    with source_path.open("rb") as handle:
        source = pickle.load(handle)
    if onnx_model is not None:
        import onnxruntime as ort
        encoder = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
    else:
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(str(text_model), local_files_only=True).to(device)
    for split_name in ("train", "valid"):
        split = source[split_name]
        text = encode_text(encoder, split["text_bert"], device)
        result = assemble_sample(split["text_bert"], text, split["audio"], split["vision"],
                                 split["classification_labels"], split["regression_labels"])
        np.savez_compressed(cache / f"{split_name}.npz", **result)
    del source
    gc.collect()
    attachment3 = data_root / "附件3-模态缺失特征样本" / "对齐版本"
    paths = sorted(attachment3.glob("*.pkl"))
    for path in paths:
        with path.open("rb") as handle:
            item = pickle.load(handle)["test"]
        text = encode_text(encoder, item["text_bert"], device)
        result = assemble_sample(item["text_bert"], text, item["audio"], item["vision"])
        np.savez_compressed(cache / f"q2_{path.stem}.npz", **result)
    return len(paths)


def load_npz(path: Path) -> dict:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def fit_audio_vision_scale(train: dict) -> dict:
    """Fit standardization using valid training positions only."""
    result = {}
    for feature, mask_name in (("audio", "amask"), ("vision", "vmask")):
        values = train[feature][train[mask_name]]
        mean = values.mean(axis=0).astype(np.float32)
        std = np.maximum(values.std(axis=0).astype(np.float32), 1e-4)
        result[feature] = (mean, std)
    return result


def apply_audio_vision_scale(data: dict, scale: dict) -> None:
    for feature, mask_name in (("audio", "amask"), ("vision", "vmask")):
        mean, std = scale[feature]
        normalized = np.clip((data[feature] - mean) / std, -5, 5)
        data[feature] = np.where(data[mask_name][..., None], normalized, 0).astype(np.float32)


class MultimodalDataset(Dataset):
    """Rows: three features, three masks, class ID, and intensity."""
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data["text"])

    def __getitem__(self, index):
        row = self.data
        inputs = tuple(torch.from_numpy(row[key][index].copy())
                       for key in ("text", "audio", "vision", "tmask", "amask", "vmask"))
        return inputs + (torch.tensor(int(row["cls"][index])),
                         torch.tensor(float(row["score"][index]), dtype=torch.float32))
