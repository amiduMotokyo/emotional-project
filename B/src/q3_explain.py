"""Model-faithful modality and local evidence by input-level deletion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from B.src.data import LABELS
from B.src.fusion import Fusion
from B.src.temporal_fusion import TemporalFusion
from C.src.q2_protocol import coherent_score

MODALITIES = ("text", "audio", "vision")
MASK_NAMES = ("tmask", "amask", "vmask")


class Q3Predictor:
    """Use the selected Q2 checkpoint unchanged, with deterministic Q3 explanations."""

    def __init__(self, package: Path, device: str = "cpu", ort_threads: int | None = None):
        import onnxruntime as ort

        self.package = package
        self.device = device
        session_options = ort.SessionOptions()
        if ort_threads is not None:
            session_options.intra_op_num_threads = ort_threads
            session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(package / "text_encoder_int8.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        saved = torch.load(package / "model.pt", map_location=device, weights_only=False)
        self.architecture = saved["architecture"]
        if self.architecture == "fusion":
            self.model = Fusion(**saved.get("model_kwargs", {}))
        elif self.architecture == "temporal":
            self.model = TemporalFusion()
        elif self.architecture == "temporal_primary":
            from B.src.q3_temporal_primary_fusion import TemporalPrimaryFusion
            self.model = TemporalPrimaryFusion(**saved.get("model_kwargs", {}))
        elif self.architecture == "impute":
            from C.src.reliability_imputation import ReliabilityImputationFusion
            self.model = ReliabilityImputationFusion()
        else:
            raise ValueError(f"unsupported Q3 fusion architecture: {self.architecture}")
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(device).eval()
        bias_path = package / "class_bias.json"
        bias = json.loads(bias_path.read_text(encoding="utf-8")) if bias_path.exists() else {}
        self.class_bias = np.asarray(bias.get("class_bias", [0.0, 0.0, 0.0]),
                                     dtype=np.float32)
        if self.class_bias.shape != (3,):
            raise ValueError("class bias must contain exactly three values")
        with np.load(package / "audio_vision_normalization.npz") as scale:
            self.audio_mean = scale["audio_mean"].copy()
            self.audio_std = scale["audio_std"].copy()
            self.vision_mean = scale["vision_mean"].copy()
            self.vision_std = scale["vision_std"].copy()

    def _encode(self, ids: np.ndarray, attention: np.ndarray) -> np.ndarray:
        # The deployed ONNX encoder changes numerically with batch size; stay at batch 1.
        inputs = {
            "input_ids": ids.astype(np.int64, copy=False)[None],
            "attention_mask": attention.astype(np.int64, copy=False)[None],
            "token_type_ids": np.zeros((1, len(ids)), dtype=np.int64),
        }
        return self.session.run(None, inputs)[0][0].astype(np.float32)

    def prepare(self, token_ids: np.ndarray, attention: np.ndarray,
                audio: np.ndarray, vision: np.ndarray) -> dict:
        ids = np.rint(token_ids).astype(np.int64).reshape(-1)
        attention = np.rint(attention).astype(bool).reshape(-1)
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        vision = np.nan_to_num(np.asarray(vision, dtype=np.float32))
        if ids.shape != (50,) or audio.shape != (50, 74) or vision.shape != (50, 35):
            raise ValueError("Q3 requires aligned 50-step 384/74/35 input")
        tmask = attention & (ids != 0) & (ids != 100) & (ids != 101) & (ids != 102)
        amask = attention & np.any(audio != 0, axis=1)
        vmask = attention & np.any(vision != 0, axis=1)
        audio = np.where(amask[:, None], np.clip((audio - self.audio_mean) /
                                                 self.audio_std, -5, 5), 0).astype(np.float32)
        vision = np.where(vmask[:, None], np.clip((vision - self.vision_mean) /
                                                   self.vision_std, -5, 5), 0).astype(np.float32)
        return {"token_ids": ids, "attention": attention,
                "audio": audio, "vision": vision,
                "tmask": tmask, "amask": amask, "vmask": vmask,
                "text_embedding": self._encode(ids, attention)}

    def predict(self, sample: dict, removed: dict[str, tuple[int, ...]] | None = None) -> dict:
        removed = removed or {}
        ids = sample["token_ids"].copy()
        masks = [sample[name].copy() for name in MASK_NAMES]
        audio = sample["audio"].copy()
        vision = sample["vision"].copy()
        for index, modality in enumerate(MODALITIES):
            positions = tuple(removed.get(modality, ()))
            if any(position < 0 or position >= 50 or not masks[index][position]
                   for position in positions):
                raise ValueError(f"cannot delete invalid {modality} position: {positions}")
            if positions:
                masks[index][list(positions)] = False
                if modality == "text":
                    ids[list(positions)] = 100
                elif modality == "audio":
                    audio[list(positions)] = 0
                else:
                    vision[list(positions)] = 0
        text = (self._encode(ids, sample["attention"])
                if removed.get("text") else sample["text_embedding"])
        tensors = [torch.from_numpy(np.asarray(value)[None].copy()).to(self.device)
                   for value in (text, audio, vision, *masks)]
        with torch.inference_mode():
            if self.architecture == "temporal":
                support = (sample["attention"] & ~np.isin(sample["token_ids"],
                                                           [0, 101, 102]))
                output = self.model(*tensors, torch.from_numpy(support[None].copy()).to(self.device))
            else:
                output = self.model(*tensors)
        raw_logits = output[0][0].float().cpu().numpy()
        biased_logits = raw_logits + self.class_bias
        exp_logits = np.exp(biased_logits - biased_logits.max())
        probabilities = exp_logits / exp_logits.sum()
        predicted = int(biased_logits.argmax())
        raw_score = float(output[1][0].float().cpu())
        served_score = float(coherent_score(
            np.asarray([predicted]), np.asarray([raw_score]))[0])
        return {
            "predicted_class": predicted,
            "predicted_polarity": LABELS[predicted],
            "predicted_intensity": served_score,
            "raw_intensity": raw_score,
            "raw_logits": raw_logits.astype(float).tolist(),
            "biased_logits": biased_logits.astype(float).tolist(),
            "probabilities": probabilities.astype(float).tolist(),
            "gate_weights": output[2][0].float().cpu().numpy().astype(float).tolist(),
        }

    def modality_effects(self, sample: dict, base: dict | None = None) -> dict:
        base = base or self.predict(sample)
        target = base["predicted_class"]
        effects = {}
        for modality, mask_name in zip(MODALITIES, MASK_NAMES):
            observed = tuple(np.flatnonzero(sample[mask_name]).astype(int))
            if not observed:
                effects[modality] = {"observed": False, "delta_logit": 0.0,
                                     "delta_probability": 0.0, "abs_share": 0.0}
                continue
            ablated = self.predict(sample, {modality: observed})
            effects[modality] = {
                "observed": True,
                "delta_logit": float(base["biased_logits"][target] -
                                     ablated["biased_logits"][target]),
                "delta_probability": float(base["probabilities"][target] -
                                           ablated["probabilities"][target]),
                "abs_share": 0.0,
                "ablated_prediction": ablated["predicted_polarity"],
            }
        total = sum(abs(item["delta_logit"]) for item in effects.values())
        if total > 1e-8:
            for item in effects.values():
                item["abs_share"] = abs(item["delta_logit"]) / total
        positive = [m for m in MODALITIES if effects[m]["observed"] and
                    effects[m]["delta_logit"] > 1e-8]
        if positive:
            main = max(positive, key=lambda m: effects[m]["delta_logit"])
            main_basis = "largest_positive_logit_drop"
        else:
            available = [m for m in MODALITIES if effects[m]["observed"]]
            main = max(available, key=lambda m: abs(effects[m]["delta_logit"])) if available else None
            main_basis = "largest_absolute_effect_no_positive_support" if main else "no_observed_modality"
        return {"effects": effects, "main_modality": main, "main_basis": main_basis}

    def local_effects(self, sample: dict, base: dict, modality: str,
                      window_width: int = 3) -> dict:
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality: {modality}")
        mask = sample[MASK_NAMES[MODALITIES.index(modality)]]
        positions = np.flatnonzero(mask).astype(int).tolist()
        target = base["predicted_class"]
        scores = [None] * 50
        for position in positions:
            ablated = self.predict(sample, {modality: (position,)})
            scores[position] = float(base["biased_logits"][target] -
                                     ablated["biased_logits"][target])
        if not positions:
            return {"signed_logit_drop_by_position": scores,
                    "positive_importance_by_position": [None] * 50,
                    "top_window": None}
        positive = sum(max(0.0, scores[position]) for position in positions)
        normalized = [(max(0.0, value) / positive if positive > 1e-8 else 0.0)
                      if value is not None else None for value in scores]
        windows = {}
        for width in range(1, window_width + 1):
            for start in positions:
                window = tuple(position for position in range(start, min(50, start + width))
                               if mask[position])
                if not window or window in windows:
                    continue
                if len(window) == 1:
                    effect = scores[window[0]]
                else:
                    joint = self.predict(sample, {modality: window})
                    effect = float(base["biased_logits"][target] -
                                   joint["biased_logits"][target])
                windows[window] = effect
        best_window = max(windows, key=lambda item: windows[item])
        if windows[best_window] <= 0:
            best_window = max(windows, key=lambda item: abs(windows[item]))
        joint_effect = windows[best_window]
        return {
            "signed_logit_drop_by_position": scores,
            "positive_importance_by_position": normalized,
            "top_window": {
                "aligned_positions_zero_based": list(best_window),
                "summed_single_position_effect": float(
                    sum(scores[position] for position in best_window)),
                "joint_delta_logit": joint_effect,
                "direction": "supports_prediction" if joint_effect > 1e-8 else
                             ("opposes_prediction" if joint_effect < -1e-8 else "neutral"),
            },
        }

    def explain(self, sample: dict, include_local: bool = True) -> dict:
        base = self.predict(sample)
        global_result = self.modality_effects(sample, base)
        result = {"prediction": base, **global_result}
        if include_local:
            result["local"] = {
                modality: self.local_effects(sample, base, modality)
                for modality in MODALITIES
            }
        return result
