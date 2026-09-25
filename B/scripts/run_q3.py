"""Q3 validation analysis and full Attachment-4 prediction/explanation export."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_recall_fscore_support)
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.attachment4 import (attachment4_directories, load_pair, localize_window,
                               stream_metadata)
from B.src.data import LABELS, load_npz
from B.src.q3_explain import MODALITIES, Q3Predictor


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flat_prediction(row_id: str, explanation: dict) -> dict:
    prediction = explanation["prediction"]
    effects = explanation["effects"]
    row = {
        "sample_id": row_id,
        "predicted_polarity": prediction["predicted_polarity"],
        "predicted_intensity": round(prediction["predicted_intensity"], 6),
        "prob_negative": round(prediction["probabilities"][0], 6),
        "prob_neutral": round(prediction["probabilities"][1], 6),
        "prob_positive": round(prediction["probabilities"][2], 6),
        "main_reference_modality": explanation["main_modality"] or "undetermined",
        "main_reference_basis": explanation["main_basis"],
    }
    for index, modality in enumerate(MODALITIES):
        row[f"{modality}_gate"] = round(prediction["gate_weights"][index], 6)
        row[f"{modality}_delta_logit"] = round(effects[modality]["delta_logit"], 6)
        row[f"{modality}_delta_probability"] = round(
            effects[modality]["delta_probability"], 6)
        row[f"{modality}_abs_effect_share"] = round(effects[modality]["abs_share"], 6)
        row[f"{modality}_observed"] = effects[modality]["observed"]
    return row


def _summarize_validation(rows: list[dict]) -> dict:
    truth = np.asarray([row["true_class"] for row in rows], dtype=np.int64)
    predicted = np.asarray([row["predicted_class"] for row in rows], dtype=np.int64)
    actual = np.asarray([row["true_intensity"] for row in rows], dtype=np.float64)
    estimated = np.asarray([row["predicted_intensity"] for row in rows], dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1, 2], zero_division=0)
    pearson = (float(np.corrcoef(actual, estimated)[0, 1])
               if actual.std() > 1e-8 and estimated.std() > 1e-8 else 0.0)
    correct = truth == predicted
    main_counts = {"all": Counter(), "correct": Counter(), "incorrect": Counter()}
    for row, is_correct in zip(rows, correct):
        key = row["main_reference_modality"]
        main_counts["all"][key] += 1
        main_counts["correct" if is_correct else "incorrect"][key] += 1
    short = np.asarray([row["text_valid_tokens"] <= 5 for row in rows])
    vision_missing = np.asarray([row["vision_valid_steps"] == 0 for row in rows])
    return {
        "n": len(rows), "accuracy": float(accuracy_score(truth, predicted)),
        "correct": int(correct.sum()),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(actual, estimated)), "pearson": pearson,
        "confusion_matrix_true_by_pred": confusion_matrix(
            truth, predicted, labels=[0, 1, 2]).tolist(),
        "class_metrics": {LABELS[i]: {"precision": float(precision[i]),
                                     "recall": float(recall[i]),
                                     "f1": float(f1[i]), "support": int(support[i])}
                          for i in range(3)},
        "main_modality_counts": {key: dict(value) for key, value in main_counts.items()},
        "neutral_to_positive_errors": int(np.sum((truth == 1) & (predicted == 2))),
        "negative_to_positive_errors": int(np.sum((truth == 0) & (predicted == 2))),
        "high_confidence_errors_over_0_7": int(sum(
            not correct[i] and max(rows[i][f"prob_{name.lower()}"]
                                   for name in LABELS) >= 0.7
            for i in range(len(rows)))),
        "short_text_le_5": {"n": int(short.sum()),
                             "accuracy": float(correct[short].mean()) if short.any() else None},
        "longer_text": {"n": int((~short).sum()),
                        "accuracy": float(correct[~short].mean()) if (~short).any() else None},
        "vision_unavailable": {"n": int(vision_missing.sum()),
                               "accuracy": float(correct[vision_missing].mean())
                               if vision_missing.any() else None},
        "mean_abs_effect_share": {
            modality: float(np.mean([row[f"{modality}_abs_effect_share"] for row in rows]))
            for modality in MODALITIES
        },
        "mean_gate": {
            modality: float(np.mean([row[f"{modality}_gate"] for row in rows]))
            for modality in MODALITIES
        },
    }


def run_validation(predictor: Q3Predictor, data_root: Path, cache: Path,
                   output: Path) -> tuple[list[dict], dict]:
    valid = load_npz(cache / "valid.npz")
    with (data_root / "附件2-数据集特征文件" / "aligned_50.pkl").open("rb") as handle:
        original = pickle.load(handle)["valid"]
    if len(valid["token_ids"]) != len(original["classification_labels"]):
        raise ValueError("validation cache size differs from original split")
    raw_ids = np.rint(original["text_bert"][:, 0]).astype(np.int64)
    if not np.array_equal(raw_ids, valid["token_ids"]):
        raise ValueError("validation cache order/token IDs differ from Attachment 2")
    rows = []
    for index in range(len(valid["token_ids"])):
        sample = predictor.prepare(valid["token_ids"][index], valid["attention"][index],
                                   valid["audio"][index], valid["vision"][index])
        base = predictor.predict(sample)
        explanation = {"prediction": base, **predictor.modality_effects(sample, base)}
        row = _flat_prediction(str(original["id"][index]), explanation)
        row.update({"validation_row_zero_based": index,
                    "true_class": int(valid["cls"][index]),
                    "true_polarity": LABELS[int(valid["cls"][index])],
                    "true_intensity": float(valid["score"][index]),
                    "predicted_class": base["predicted_class"],
                    "correct": int(base["predicted_class"] == int(valid["cls"][index])),
                    "text_valid_tokens": int(sample["tmask"].sum()),
                    "audio_valid_steps": int(sample["amask"].sum()),
                    "vision_valid_steps": int(sample["vmask"].sum()),
                    "raw_text": str(original["raw_text"][index])})
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f"validation {index + 1}/{len(valid['token_ids'])}", flush=True)
    write_csv(output / "validation_predictions.csv", rows)
    summary = _summarize_validation(rows)
    (output / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, summary


def _extract_media(ffmpeg: Path | None, video: Path, time_seconds: float,
                   output: Path, audio: bool = False) -> str | None:
    if ffmpeg is None:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    if audio:
        start = max(0.0, time_seconds - 0.5)
        command = [str(ffmpeg), "-nostdin", "-y", "-ss", f"{start:.3f}",
                   "-i", str(video), "-t", "1.000", "-vn", "-ac", "1",
                   "-ar", "16000", "-c:a", "pcm_s16le", str(output)]
    else:
        command = [str(ffmpeg), "-nostdin", "-y", "-ss", f"{time_seconds:.3f}",
                   "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    return output.name if result.returncode == 0 and output.exists() else None


def _card(sample_id: str, record: dict) -> str:
    prediction = record["prediction"]
    parts = [f"# 附件4样本 {sample_id} 解释卡", "",
             f"原文：{record['raw_text']}", "",
             f"预测：**{prediction['predicted_polarity']}**；强度 **{prediction['predicted_intensity']:.3f}**。",
             f"三类概率（负/中/正）：{', '.join(f'{value:.3f}' for value in prediction['probabilities'])}。",
             "", "| 模态 | 删除后目标类logit下降 | 作用绝对占比 | 门控权重 | 关键片段 |",
             "|---|---:|---:|---:|---|"]
    for index, modality in enumerate(MODALITIES):
        effect = record["effects"][modality]
        local = record["localized_evidence"].get(modality, {})
        local_window = record["local"][modality]["top_window"]
        local_delta = (local_window["joint_delta_logit"] if local_window else 0.0)
        if local.get("mapping_status") == "exact_tokenizer_offset":
            evidence = f"原文 {local['char_start_zero_based']}:{local['char_end_exclusive']} ‘{local['text_excerpt']}’"
        elif local.get("mapping_status") == "exact_feature_match_nominal_rate":
            evidence = (f"对齐位置 {local['aligned_positions_zero_based']}，"
                        f"约 {local['time_start_seconds']:.2f}–{local['time_end_seconds']:.2f} 秒")
        else:
            evidence = f"对齐位置 {local.get('aligned_positions_zero_based', [])}；{local.get('mapping_status', 'unavailable')}"
        evidence += f"；局部遮挡Δ={local_delta:+.3f}"
        parts.append(f"| {modality} | {effect['delta_logit']:+.3f} | "
                     f"{effect['abs_share']:.1%} | {prediction['gate_weights'][index]:.3f} | {evidence} |")
    parts.extend(["", f"主要参考模态：**{record['main_modality']}**（{record['main_basis']}）。",
                  "正的logit下降表示该模态支持当前预测；负值表示它抑制当前预测。门控权重仅作模型内部诊断，不作为贡献值。"])
    main = record["main_modality"]
    if main and record["local"][main]["top_window"]:
        global_effect = record["effects"][main]["delta_logit"]
        local_effect = record["local"][main]["top_window"]["joint_delta_logit"]
        if global_effect > 0 and local_effect / global_effect < 0.1:
            parts.append("主要模态的作用分散在多个位置；单个最多3步的片段只解释了其中较小部分。")
    if record.get("visual_preview"):
        parts.extend(["", f"![视觉关键帧](../frames/{record['visual_preview']})"])
    if record.get("audio_preview"):
        parts.extend(["", f"[播放语音证据](../audio/{record['audio_preview']})"])
    parts.extend(["", "局部时间由对齐特征与未对齐帧精确匹配后，按音频20 Hz、视觉15 Hz估计；视频流时长用于核查。",
                  "遮挡效应反映模型对输入变化的敏感性，不等同于人类情绪的因果来源。", ""])
    return "\n".join(parts)


def run_attachment4(predictor: Q3Predictor, data_root: Path, tokenizer,
                    output: Path, ffprobe: Path | None,
                    ffmpeg: Path | None) -> list[dict]:
    aligned_dir, _ = attachment4_directories(data_root)
    paths = sorted(aligned_dir.glob("*.pkl"))
    if len(paths) != 20:
        raise ValueError(f"expected 20 Attachment-4 aligned samples, got {len(paths)}")
    records, flat_rows = [], []
    for path in paths:
        sample_id = path.stem
        aligned, unaligned, video = load_pair(data_root, sample_id)
        sample = predictor.prepare(aligned["text_bert"][0], aligned["text_bert"][1],
                                   aligned["audio"], aligned["vision"])
        explanation = predictor.explain(sample)
        streams = stream_metadata(video, ffprobe)
        localized = {}
        for modality in MODALITIES:
            window = explanation["local"][modality]["top_window"]
            positions = window["aligned_positions_zero_based"] if window else []
            localized[modality] = localize_window(
                modality, positions, aligned, unaligned, video, tokenizer, streams)
        record = {"sample_id": sample_id, "raw_text": str(aligned["raw_text"]),
                  **explanation, "localized_evidence": localized,
                  "audio_preview": None, "visual_preview": None}
        if localized["vision"].get("mapping_status") == "exact_feature_match_nominal_rate":
            record["visual_preview"] = _extract_media(
                ffmpeg, video, localized["vision"]["keyframe_time_seconds"],
                output / "frames" / f"{sample_id}_vision.jpg")
        if localized["audio"].get("mapping_status") == "exact_feature_match_nominal_rate":
            center = (localized["audio"]["time_start_seconds"] +
                      localized["audio"]["time_end_seconds"]) / 2
            record["audio_preview"] = _extract_media(
                ffmpeg, video, center, output / "audio" / f"{sample_id}_audio.wav",
                audio=True)
        records.append(record)
        flat = _flat_prediction(sample_id, explanation)
        main = localized.get(explanation["main_modality"], {})
        flat.update({"raw_text": record["raw_text"],
                     "main_evidence_aligned_positions": json.dumps(
                         main.get("aligned_positions_zero_based", [])),
                     "main_evidence_text": main.get("text_excerpt", ""),
                     "main_evidence_start_seconds": main.get("time_start_seconds", ""),
                     "main_evidence_end_seconds": main.get("time_end_seconds", ""),
                     "main_evidence_video_frame": main.get("approx_video_frame_zero_based", ""),
                     "main_evidence_mapping_status": main.get("mapping_status", ""),
                     "main_local_joint_delta_logit": (
                         explanation["local"][explanation["main_modality"]]["top_window"]["joint_delta_logit"]
                         if explanation["main_modality"] and
                         explanation["local"][explanation["main_modality"]]["top_window"] else 0.0)})
        flat_rows.append(flat)
        (output / "cards").mkdir(parents=True, exist_ok=True)
        (output / "cards" / f"{sample_id}.md").write_text(
            _card(sample_id, record), encoding="utf-8")
        print(f"Attachment 4 {sample_id}/{len(paths)}", flush=True)
    write_csv(output / "attachment4_predictions_explanations.csv", flat_rows)
    (output / "attachment4_explanations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--onnxruntime-path", type=Path,
                        help="directory that contains an existing onnxruntime installation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.onnxruntime_path is not None:
        sys.path.insert(0, str(args.onnxruntime_path))
    predictor = Q3Predictor(args.package, args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer),
                                              local_files_only=True, use_fast=True)
    rows, summary = run_validation(predictor, args.data_root, args.cache, args.output)
    records = run_attachment4(predictor, args.data_root, tokenizer, args.output,
                              args.ffprobe, args.ffmpeg)
    report = {"validation": summary, "attachment4_n": len(records),
              "source_package": str(args.package), "class_bias": predictor.class_bias.tolist(),
              "architecture": predictor.architecture,
              "explanation": "input deletion on fixed predicted-class biased logit",
              "validation_labels_used_for_checkpoint_selection_and_class_bias": True,
              "validation_labels_not_backpropagated": True,
              "attachment4_has_no_labels": True}
    (args.output / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation_accuracy": summary["accuracy"],
                      "validation_macro_f1": summary["macro_f1"],
                      "attachment4_n": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
