"""Read-only archive review; writes derived evidence, never trains a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'C/outputs/q2_accuracy_screen_20260925'


def main():
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    checks = {}
    for name, expected in read(SOURCE / 'minilm_manifest.json').items():
        raw = (SOURCE / name).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        variants = {'exact': raw, 'LF_normalized': lf, 'CRLF_normalized': lf.replace(b'\n', b'\r\n')}
        match = next((k for k, v in variants.items()
                      if len(v) == expected['bytes'] and hashlib.sha256(v).hexdigest() == expected['sha256']), None)
        assert match is not None, name
        checks[name] = match
    metrics = {}
    labels = None
    for name in ['validation_clean_predictions_seed20260925.json',
                 'validation_clean_predictions_calibrated_seed20260925.json']:
        rows = read(SOURCE / 'minilm_finetune' / name)
        assert len(rows) == 728 and [r['row'] for r in rows] == list(range(728))
        y = np.array([r['true_class'] for r in rows])
        if labels is not None:
            assert np.array_equal(labels, y)
        labels = y
        pred = np.array([r['predicted_class'] for r in rows])
        target = np.array([r['target_score'] for r in rows])
        score = np.array([r['served_score'] for r in rows])
        prob = np.asarray([r['probabilities'] for r in rows])
        bias = [0.2, 0, 0] if 'calibrated' in name else [0, 0, 0]
        assert np.array_equal((np.log(np.maximum(prob, 1e-12)) + bias).argmax(axis=1), pred)
        metrics[name] = dict(n=len(rows), correct=int((y == pred).sum()),
            accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average='macro')),
            per_class_f1=f1_score(y, pred, labels=[0, 1, 2], average=None).tolist(),
            mae=float(mean_absolute_error(target, score)), pearson=float(np.corrcoef(target, score)[0, 1]),
            confusion_matrix=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())
    grid = read(SOURCE / 'minilm_finetune/validation_grid_full_calibrated_seed20260925.json')['metrics']
    assert len(grid) == 64
    report = dict(source_commit='226b84fd00eba160511a8723b5f01cd45c553bdc',
        scope='Saved archive integrity and prediction arithmetic only; no training or inference rerun.',
        artifact_checks=checks, metrics=metrics,
        missing63_mean_accuracy=float(np.mean([r['accuracy'] for r in grid[1:]])),
        historical_log_damaged_conditions=3, current_training_script_damaged_conditions=6,
        original_fp32_checkpoint_in_archive=False,
        calibrated_prediction_probabilities='Raw probabilities; predicted_class applies [0.2,0,0] to log probabilities.')
    out = ROOT / 'docs/C/experiments/assets/round6_redesign/archive_review.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'verified_files': len(checks), 'metrics': metrics}, ensure_ascii=False))


if __name__ == '__main__':
    main()
