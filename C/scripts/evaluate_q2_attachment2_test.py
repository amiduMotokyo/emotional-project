"""Evaluate locked Q2 candidates on Attachment 2's untouched labeled test split."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from B.src.data import (LABELS, apply_audio_vision_scale, assemble_sample,
                        encode_text, load_npz)
from C.src.missingness import corrupt_masks
from C.src.q2_protocol import (MODES, POSITIONS, RATES, ViewDataset,
                               _encode_with_masks, base_masks, case_name,
                               evaluate_loader)
from C.scripts.run_q2_corrected import load_checkpoint, loader_for_case


METRICS = ("accuracy", "macro_f1", "mae", "pearson")
CLASS_IDS = {"Negative": 0, "Neutral": 1, "Positive": 2}


def _cell_value(cell, shared: list[str], ns: dict[str, str]):
    value = cell.find("m:v", ns)
    if value is None:
        inline = cell.find("m:is", ns)
        if inline is None:
            return None
        return "".join(item.text or "" for item in inline.findall(".//m:t", ns))
    result = value.text
    if cell.attrib.get("t") == "s":
        return shared[int(result)]
    return result


def validate_test_labels(label_path: Path, sample_ids: list[str], classes: np.ndarray,
                         scores: np.ndarray) -> None:
    """Check that aligned test rows match label.xlsx by video/clip ID and labels."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(label_path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//m:t", ns))
                      for item in shared_root.findall("m:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//m:sheetData/m:row", ns)[1:]:
        cells = {}
        for cell in row.findall("m:c", ns):
            col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            cells[col] = _cell_value(cell, shared, ns)
        if cells.get("F") == "test":
            rows.append(cells)
    expected_ids = [f"{row['A']}$_${row['B']}" for row in rows]
    if len(expected_ids) != len(sample_ids) or set(expected_ids) != set(sample_ids):
        raise ValueError("aligned_50 test IDs do not exactly match label.xlsx mode=test IDs")
    by_id = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    for row, sample_id in zip(rows, expected_ids):
        index = by_id[sample_id]
        expected_class = CLASS_IDS[row["E"]]
        expected_score = float(row["D"])
        if int(classes[index]) != expected_class or not np.isclose(
                float(scores[index]), expected_score, atol=1e-5):
            raise ValueError(f"test labels disagree with label.xlsx for {sample_id}")


def load_attachment2_test(data_root: Path, run: Path,
                          session: ort.InferenceSession) -> tuple[dict, list[str]]:
    data_path = data_root / "附件2-数据集特征文件" / "aligned_50.pkl"
    with data_path.open("rb") as handle:
        split = pickle.load(handle)["test"]
    sample_ids = list(split["id"])
    text = encode_text(session, split["text_bert"], "cpu", batch_size=1)
    data = assemble_sample(split["text_bert"], text, split["audio"], split["vision"],
                           split["classification_labels"], split["regression_labels"])
    validate_test_labels(data_root / "附件2-数据集特征文件" / "label.xlsx",
                         sample_ids, data["cls"], data["score"])
    with np.load(run / "audio_vision_normalization.npz") as scale:
        norms = {"audio": (scale["audio_mean"], scale["audio_std"]),
                 "vision": (scale["vision_mean"], scale["vision_std"])}
    apply_audio_vision_scale(data, norms)
    print(f"loaded {len(sample_ids)} Attachment-2 test samples; label.xlsx IDs/classes/scores verified",
          flush=True)
    return data, sample_ids


def prepare_test_cases(data: dict, session: ort.InferenceSession,
                       directory: Path) -> list[tuple[str, float, str, Path]]:
    """Make the controlled 63-case grid; reuse each text mask encoding once."""
    directory.mkdir(parents=True, exist_ok=True)
    text_views: dict[tuple[float, str], np.ndarray] = {}
    original = base_masks(data)
    for rate in RATES:
        for position in POSITIONS:
            path = directory / case_name("text", rate, position)
            if path.exists():
                with np.load(path) as archive:
                    text_views[(rate, position)] = archive["text"].copy()
                continue
            text_masks = corrupt_masks(original, "text", rate, position,
                                       rng=random.Random(20260924))
            stacked = np.stack([item.numpy() for item in text_masks], axis=1)
            text_views[(rate, position)] = _encode_with_masks(session, data, stacked)
            np.savez_compressed(path, text=text_views[(rate, position)], masks=stacked)
            print(f"encoded test text mask {rate:.0%} {position}", flush=True)
    case_paths = []
    for mode in MODES:
        for rate in RATES:
            for position in POSITIONS:
                path = directory / case_name(mode, rate, position)
                if path.exists():
                    case_paths.append((mode, rate, position, path))
                    continue
                masks = corrupt_masks(original, mode, rate, position,
                                      rng=random.Random(20260924))
                stacked = np.stack([item.numpy() for item in masks], axis=1)
                text = (text_views[(rate, position)] if "text" in mode.split("+")
                        else data["text"])
                np.savez_compressed(path, text=text, masks=stacked)
                case_paths.append((mode, rate, position, path))
    if len(case_paths) != 63:
        raise AssertionError(f"expected 63 test corruption conditions, got {len(case_paths)}")
    return case_paths


def _mean_metrics(rows: list[dict]) -> dict:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def _metric_spread(rows: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in rows]
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def resolve_checkpoint(value: str, run: Path) -> Path:
    path = Path(value)
    candidates = ([path] if path.is_absolute() else [run / path])
    candidates.append(run / "checkpoints" / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find checkpoint {value} under {run}")


def evaluate(run: Path, output: Path, device: str, data_root: Path,
             encoder_path: Path) -> dict:
    session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    data, sample_ids = load_attachment2_test(data_root, run, session)
    case_paths = prepare_test_cases(data, session, output / "test_cases")
    records = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selected_model.json").read_text(encoding="utf-8"))
    candidate_rows = []
    condition_rows = []
    for record in records:
        model = load_checkpoint(resolve_checkpoint(record["checkpoint"], run), device)
        clean = evaluate_loader(model, DataLoader(ViewDataset(data),
                                   batch_size=128, shuffle=False), device)
        grid = []
        for mode, rate, position, path in case_paths:
            result = evaluate_loader(model, loader_for_case(data, path), device)
            row = {"architecture": record["architecture"], "seed": record["seed"],
                   "mode": mode, "rate": rate, "position": position, **result}
            condition_rows.append(row)
            grid.append(result)
        candidate_rows.append({
            "architecture": record["architecture"], "seed": record["seed"],
            **{f"clean_{key}": clean[key] for key in METRICS},
            **{f"grid_mean_{key}": value for key, value in _mean_metrics(grid).items()},
            "grid_worst_macro_f1": min(row["macro_f1"] for row in grid),
        })
        print("evaluated test candidate", record["architecture"], record["seed"], flush=True)
        del model

    with (output / "attachment2_test_candidates.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = list(candidate_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (output / "attachment2_test_candidate_grid.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = ("architecture", "seed", "mode", "rate", "position", *METRICS, "n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in condition_rows])

    architecture_summary = {}
    for architecture in ("clean_gate", "robust_gate", "temporal"):
        members = [row for row in candidate_rows if row["architecture"] == architecture]
        architecture_summary[architecture] = {
            field: _metric_spread(members, field)
            for field in ("clean_macro_f1", "clean_mae", "grid_mean_macro_f1",
                          "grid_mean_mae", "grid_mean_pearson", "grid_worst_macro_f1")
        }

    selected_path = resolve_checkpoint(selection["checkpoint"], run)
    selected_model = load_checkpoint(selected_path, device)
    clean_coherent = evaluate_loader(selected_model, DataLoader(
        ViewDataset(data), batch_size=128, shuffle=False), device,
        coherent=True, include_rows=True)
    clean_rows = clean_coherent.pop("rows")
    predictions = []
    for i, sample_id in enumerate(sample_ids):
        predicted = int(clean_rows["predicted"][i])
        predictions.append({
            "sample_id": sample_id,
            "actual_polarity": LABELS[int(clean_rows["actual_class"][i])],
            "predicted_polarity": LABELS[predicted],
            "actual_intensity": round(float(clean_rows["actual"][i]), 6),
            "predicted_intensity": round(float(clean_rows["score"][i]), 6),
            "prob_negative": round(float(clean_rows["probabilities"][i, 0]), 6),
            "prob_neutral": round(float(clean_rows["probabilities"][i, 1]), 6),
            "prob_positive": round(float(clean_rows["probabilities"][i, 2]), 6),
        })
    with (output / "attachment2_test_predictions.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    served_grid = []
    for mode, rate, position, path in case_paths:
        result = evaluate_loader(selected_model, loader_for_case(data, path),
                                 device, coherent=True)
        served_grid.append({"mode": mode, "rate": rate, "position": position, **result})
    with (output / "attachment2_test_selected_grid.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = ("mode", "rate", "position", *METRICS, "n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in served_grid])
    summary = {
        "split": "Attachment 2 label.xlsx mode=test",
        "n": len(sample_ids),
        "labels_verified_against_label_xlsx": True,
        "test_used_for_training_or_model_selection": False,
        "test_corruption_evaluation": "controlled 63-case grid; clean test is also reported",
        "candidate_comparison_raw_head_mean_std_across_three_seeds": architecture_summary,
        "selected_model": {"architecture": selection["architecture"],
                           "seed": selection["seed"],
                           "checkpoint": selected_path.name,
                           "architecture_selection": selection["architecture_selection"]},
        "selected_model_clean_coherent": clean_coherent,
        "selected_model_63_case_coherent_mean": _mean_metrics(served_grid),
        "selected_model_63_case_worst_macro_f1": float(
            min(row["macro_f1"] for row in served_grid)),
    }
    (output / "attachment2_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_comparison": architecture_summary,
                      "selected_clean": clean_coherent,
                      "selected_grid_mean": summary["selected_model_63_case_coherent_mean"],
                      "selected_grid_worst_f1": summary["selected_model_63_case_worst_macro_f1"]},
                     ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True,
                        help="training run with normalization, reports, and candidate checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluate(args.run, args.output, args.device, args.data_root, args.encoder)


if __name__ == "__main__":
    main()
