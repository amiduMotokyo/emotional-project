"""Assemble a portable, identity-free Q2 model and results archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(repo: Path, run: Path, encoder: Path, output: Path):
    model = run / "model.pt"
    predictions = run / "attachment3_predictions.csv"
    with predictions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30:
        raise ValueError("Attachment-3 output must contain 30 unique sample IDs")
    for row in rows:
        score = float(row["intensity"])
        if row["polarity"] == "Neutral" and score != 0:
            raise ValueError("neutral predictions must have intensity zero")
        if row["polarity"] == "Positive" and score < 0:
            raise ValueError("positive predictions must have nonnegative intensity")
        if row["polarity"] == "Negative" and score > 0:
            raise ValueError("negative predictions must have nonpositive intensity")
    selected = json.loads((run / "selected_model.json").read_text(encoding="utf-8"))
    validation = json.loads((run / "selected_validation.json").read_text(encoding="utf-8"))
    error_analysis = json.loads((run / "error_analysis.json").read_text(encoding="utf-8"))
    served = json.loads((run / "served_grid.json").read_text(encoding="utf-8"))
    test = json.loads((run / "attachment2_test_summary.json").read_text(encoding="utf-8"))
    test_predictions_path = run / "attachment2_test_predictions.csv"
    with test_predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        test_predictions = list(csv.DictReader(handle))
    if len(test_predictions) != 727 or len({row["sample_id"] for row in test_predictions}) != 727:
        raise ValueError("Attachment-2 test output must contain 727 unique sample IDs")
    for row in test_predictions:
        score = float(row["predicted_intensity"])
        if row["predicted_polarity"] == "Neutral" and score != 0:
            raise ValueError("neutral test predictions must have intensity zero")
        if row["predicted_polarity"] == "Positive" and score < 0:
            raise ValueError("positive test predictions must have nonnegative intensity")
        if row["predicted_polarity"] == "Negative" and score > 0:
            raise ValueError("negative test predictions must have nonpositive intensity")
    grid_summary = {key: error_analysis[key] for key in (
        "served_missing_grid_mean", "served_missing_grid_worst_macro_f1")}
    dest = run / "question2_corrected"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "B" / "src").mkdir(parents=True)
    (dest / "B" / "scripts").mkdir(parents=True)
    (dest / "C" / "src").mkdir(parents=True)
    (dest / "C" / "scripts").mkdir(parents=True)
    (dest / "checkpoints").mkdir(parents=True)
    for name in ("__init__.py", "data.py", "fusion.py", "temporal_fusion.py"):
        shutil.copy2(repo / "B" / "src" / name, dest / "B" / "src" / name)
    for name in ("__init__.py", "reencode_q2_cache.py", "prepare_q2_cache.py"):
        shutil.copy2(repo / "B" / "scripts" / name, dest / "B" / "scripts" / name)
    for name in ("__init__.py", "missingness.py", "q2_protocol.py"):
        shutil.copy2(repo / "C" / "src" / name, dest / "C" / "src" / name)
    for name in ("__init__.py", "run_q2_corrected.py", "infer_q2_corrected.py",
                 "analyze_q2_corrected.py", "evaluate_q2_attachment2_test.py",
                 "package_q2_corrected.py"):
        shutil.copy2(repo / "C" / "scripts" / name, dest / "C" / "scripts" / name)
    for name in ("attachment3_predictions.csv", "validation_predictions.csv",
                 "missing_grid.csv", "served_grid.csv", "error_analysis.json",
                 "fig_confusion.png", "fig_regression.png",
                 "fig_missing_macro_f1.png", "fig_missing_mae.png"):
        shutil.copy2(run / name, dest / name)
    for name in ("attachment2_test_candidates.csv", "attachment2_test_candidate_grid.csv",
                 "attachment2_test_predictions.csv", "attachment2_test_selected_grid.csv"):
        shutil.copy2(run / name, dest / name)
    portable_test = dict(test)
    portable_test["selected_model"] = {
        "architecture": selected["architecture"], "seed": selected["seed"],
        "checkpoint": Path(selected["checkpoint"]).name,
        "architecture_selection": selected["architecture_selection"],
    }
    (dest / "attachment2_test_summary.json").write_text(
        json.dumps(portable_test, ensure_ascii=False, indent=2), encoding="utf-8")
    training_report = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
    for entry in training_report:
        entry["checkpoint"] = f"checkpoints/{Path(entry['checkpoint']).name}"
    (dest / "training_report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2), encoding="utf-8")
    for checkpoint in sorted((run / "checkpoints").glob("*.pt")):
        shutil.copy2(checkpoint, dest / "checkpoints" / checkpoint.name)
    shutil.copy2(model, dest / "model.pt")
    shutil.copy2(encoder, dest / "text_encoder_int8.onnx")
    shutil.copy2(run / "audio_vision_normalization.npz", dest / "audio_vision_normalization.npz")
    shutil.copy2(repo / "C" / "configs" / "q2_corrected.json", dest / "q2_corrected.json")
    clean_selected = {
        "architecture": selected["architecture"], "seed": selected["seed"],
        "checkpoint": "model.pt", "text_encoder_sha256": sha256(encoder),
        "decoder": "predicted-class-consistent signed intensity; Neutral is exactly zero",
        "architecture_selection": selected["architecture_selection"],
    }
    (dest / "model_metadata.json").write_text(
        json.dumps(clean_selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "selected_validation.json").write_text(json.dumps({
        "raw": validation["raw"], "coherent": validation["coherent"],
        "selected": "model.pt",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_selected_run = {
        "architecture": selected["architecture"], "seed": selected["seed"],
        "checkpoint": f"checkpoints/{Path(selected['checkpoint']).name}",
        "architecture_selection": selected["architecture_selection"],
        "seed_selection": selected["seed_selection"],
    }
    (dest / "selected_model.json").write_text(
        json.dumps(clean_selected_run, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "data": {"source": "Attachment 2 aligned_50.pkl; labels cross-checked with label.xlsx",
                 "train": 3395, "valid": 728, "test": 727, "attachment3": 30,
                 "test_used_for_training_or_selection": False,
                 "test_labels_verified_against_label_xlsx": True,
                 "attachment3_used_for_training_or_selection": False},
        "text_encoder": {"name": "sentence-transformers/all-MiniLM-L6-v2",
                         "artifact": "text_encoder_int8.onnx",
                         "batch_size": 1,
                         "sha256": clean_selected["text_encoder_sha256"]},
        "selected_model": clean_selected,
        "validation_raw_heads": validation["raw"],
        "validation_delivered_decoder": validation["coherent"],
        "missing_grid_delivered_decoder": grid_summary,
        "missing_grid_conditions": len(served),
        "attachment2_test_clean_coherent": test["selected_model_clean_coherent"],
        "attachment2_test_missing_grid_coherent_mean": test[
            "selected_model_63_case_coherent_mean"],
        "attachment2_test_missing_grid_worst_macro_f1": test[
            "selected_model_63_case_worst_macro_f1"],
        "attachment2_test_candidate_comparison": test[
            "candidate_comparison_raw_head_mean_std_across_three_seeds"],
        "class_counts_attachment3": {label: sum(row["polarity"] == label for row in rows)
                                      for label in ("Negative", "Neutral", "Positive")},
        "attachment3_neutral_nonzero": sum(
            row["polarity"] == "Neutral" and float(row["intensity"]) != 0 for row in rows),
    }
    (dest / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "training_summary.json").write_text(json.dumps([
        {"architecture": entry["architecture"], "seed": entry["seed"],
         "best_epoch": entry["best_epoch"], "best_selection": entry["best_selection"],
         "clean_validation_raw": entry["final"]["clean"],
         "selected_missing_conditions_raw": entry["final"]["damaged"]}
        for entry in json.loads((run / "training_report.json").read_text(encoding="utf-8"))
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "requirements.txt").write_text(
        "numpy>=1.26\ntorch>=2.7\nscikit-learn>=1.5\nonnxruntime>=1.20\nmatplotlib>=3.8\n",
        encoding="utf-8")
    (dest / "README.md").write_text(readme(report), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dest.parent))
    print(json.dumps({"archive": str(output), "size_bytes": output.stat().st_size,
                      "files": len(zipfile.ZipFile(output).namelist()),
                      "included_neutral_predictions": 0}, ensure_ascii=False), flush=True)


def readme(report: dict) -> str:
    raw, delivered = report["validation_raw_heads"], report["validation_delivered_decoder"]
    grid = report["missing_grid_delivered_decoder"]
    test = report["attachment2_test_clean_coherent"]
    test_grid = report["attachment2_test_missing_grid_coherent_mean"]
    return f'''# 问题二：连续局部缺失下的鲁棒情感预测

本包使用附件二对齐版训练集3395条学习参数、验证集728条选模型，并在独立的727条测试集上评估。测试集没有参与训练或模型选择；测试样本ID、类别和强度与附件二 `label.xlsx` 逐条核对。附件三30条另作最终推理。文本、音频和视觉输入固定为50个对齐位置；三种模态特征在相同原始位置进行连续区间缺失。

## 方法

输入为 `text_bert`、74维音频和35维视觉。文本经冻结的 `all-MiniLM-L6-v2` int8 ONNX编码为384维。缺失文本在编码前将对应词元替换成 `[UNK]`，并保留原attention mask；音频、视觉按独立观测掩码屏蔽。所有文本都以单条样本调用ONNX编码器，使训练、验证和部署保持相同的输入形状。音视频标准化参数只由训练集拟合。

最终选用带掩码池化和样本级可靠性门控的模型。候选模型为完整输入门控基线、缺失增强门控模型和带位置卷积及双向GRU的时序模型。缺失增强对每条训练样本独立抽取，概率为0.8，区间长度为有效序列跨度的10%、20%、30%或50%，模态组合覆盖全部七种非空子集。8组受损缓存版本循环用于训练。目标函数是平方根类别频率加权交叉熵加0.8倍SmoothL1。优化器AdamW，学习率0.0008、权重衰减0.01、batch size 64、最多35轮、早停7轮，三个随机种子。

## 验证结果

选中结构：`{report['selected_model']['architecture']}`，种子 `{report['selected_model']['seed']}`。分类结果不变；强度解码按预测极性约束符号，中性输出严格为0。

| 指标 | 原始双头 | 最终一致性解码 |
|---|---:|---:|
| Accuracy | {raw['accuracy']:.4f} | {delivered['accuracy']:.4f} |
| Macro-F1 | {raw['macro_f1']:.4f} | {delivered['macro_f1']:.4f} |
| MAE | {raw['mae']:.4f} | {delivered['mae']:.4f} |
| Pearson | {raw['pearson']:.4f} | {delivered['pearson']:.4f} |

在63种模态、比例和位置组合的验证网格上，所交付解码器的平均Macro-F1为{grid['served_missing_grid_mean']['macro_f1']:.4f}，平均MAE为{grid['served_missing_grid_mean']['mae']:.4f}，最差Macro-F1为{grid['served_missing_grid_worst_macro_f1']:.4f}。附件三预测类别数为{report['class_counts_attachment3']}；中性强度非零为{report['attachment3_neutral_nonzero']}条。附件三无标签，不计算准确率。

## 附件二留出测试结果

此前锁定的 `robust_gate` 种子模型在727条测试样本上的干净输入结果为：Accuracy {test['accuracy']:.4f}、Macro-F1 {test['macro_f1']:.4f}、MAE {test['mae']:.4f}、Pearson {test['pearson']:.4f}。63种连续缺失场景的平均Macro-F1为{test_grid['macro_f1']:.4f}、MAE为{test_grid['mae']:.4f}，最差Macro-F1为{report['attachment2_test_missing_grid_worst_macro_f1']:.4f}。

为核验缺失增强收益，在测试集上对训练策略作事后比较（不据此改选模型）：`clean_gate` 未加入人工缺失增强，`robust_gate` 使用局部缺失增强。三种子平均的63场景Macro-F1分别为{report['attachment2_test_candidate_comparison']['clean_gate']['grid_mean_macro_f1']['mean']:.4f}±{report['attachment2_test_candidate_comparison']['clean_gate']['grid_mean_macro_f1']['std']:.4f}和{report['attachment2_test_candidate_comparison']['robust_gate']['grid_mean_macro_f1']['mean']:.4f}±{report['attachment2_test_candidate_comparison']['robust_gate']['grid_mean_macro_f1']['std']:.4f}；平均MAE分别为{report['attachment2_test_candidate_comparison']['clean_gate']['grid_mean_mae']['mean']:.4f}±{report['attachment2_test_candidate_comparison']['clean_gate']['grid_mean_mae']['std']:.4f}和{report['attachment2_test_candidate_comparison']['robust_gate']['grid_mean_mae']['mean']:.4f}±{report['attachment2_test_candidate_comparison']['robust_gate']['grid_mean_mae']['std']:.4f}。最终模型仍按验证集选定。

## 运行附件三推理

安装 `requirements.txt`，将赛题附件数据解压到 `<data-root>` 后运行：

```bash
python C/scripts/infer_q2_corrected.py \\
  --data-root /path/to/E题数据 \\
  --package /path/to/question2_corrected \\
  --output attachment3_predictions.csv \\
  --device cpu
```

在附件二留出测试集上复核全部候选模型及63种缺失条件：

```bash
python C/scripts/evaluate_q2_attachment2_test.py \\
  --data-root /path/to/E题数据 \\
  --encoder text_encoder_int8.onnx \\
  --run . \\
  --output attachment2_test_eval \\
  --device cpu
```

## 重建训练与验证

先用包内ONNX编码器准备基础缓存，再准备缺失视图、训练、全量验证网格和分析：

```bash
python B/scripts/prepare_q2_cache.py --data-root /path/to/E题数据 --text-model unused --onnx-model text_encoder_int8.onnx --cache cache --device cpu
python C/scripts/run_q2_corrected.py --phase prepare --cache cache --encoder text_encoder_int8.onnx --output experiment
python C/scripts/run_q2_corrected.py --phase train --cache cache --encoder text_encoder_int8.onnx --output experiment
python C/scripts/run_q2_corrected.py --phase evaluate --cache cache --encoder text_encoder_int8.onnx --output experiment
python C/scripts/analyze_q2_corrected.py --cache cache --output experiment --device cpu
```

完整缺失视图会占用较多临时磁盘空间；基础缓存和实验缓存可在复现后删除，不需放入本包。验证网格完整结果在 `missing_grid.csv` 和 `served_grid.csv`；附件二测试候选比较、逐条件网格、最终预测和汇总分别在 `attachment2_test_candidates.csv`、`attachment2_test_candidate_grid.csv`、`attachment2_test_predictions.csv`、`attachment2_test_selected_grid.csv` 和 `attachment2_test_summary.json`。模型和运行环境版本见 `q2_corrected.json` 与 `requirements.txt`。
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.repo, args.run, args.encoder, args.output)


if __name__ == "__main__":
    main()
