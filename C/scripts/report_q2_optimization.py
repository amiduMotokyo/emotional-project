"""Export paper tables and figures from a completed optimization evaluation; no training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import confusion_matrix
from C.src.q2_protocol import MODES, RATES, POSITIONS, case_name

METRICS = ('accuracy', 'macro_f1', 'mae', 'pearson')
GROUPS = dict(E0='Default', E1='Random HPO', E2='DE', E3='DE + random subset',
              E4='DE + top-k', E5='DE + BPSO')
CASES = {'clean': ('clean', 0, 'none'), **{
    case_name(m, r, p): (m, r, p) for m in MODES for r in RATES for p in POSITIONS}}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_out(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    return dict(mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None)


def report(root, split):
    grid_path = root / f'{split}_grid.json'
    grid, final = read(grid_path), read(root / 'final.json')
    if not grid.get('complete') or set(final) != set(GROUPS):
        raise ValueError('paper export requires completed grid and all six groups')
    seeds = {g: [r['spec']['seed'] for r in final[g]['runs']] for g in GROUPS}
    if any(len(v) != len(set(v)) or set(v) != set(seeds['E0']) for v in seeds.values()):
        raise ValueError('groups must have identical distinct seeds')
    expected = {(g, s, c) for g in GROUPS for s in seeds[g] for c in CASES}
    lookup = {(r['group'], r['seed'], r['case']): r for r in grid['rows']}
    if set(lookup) != expected or len(grid['rows']) != len(expected):
        raise ValueError('missing or duplicate group/seed/condition in evaluation grid')
    # Validate predictions before exporting, including alignment with aggregate metrics.
    sources = [grid_path, root / 'final.json', root / 'locked_selection.json']
    predictions = {}
    for g in GROUPS:
        for seed in seeds[g]:
            path = root / 'predictions' / split / f'{g}_{seed}_clean.npz'
            with np.load(path) as archive:
                data = {k: archive[k] for k in archive.files}
            n = lookup[g, seed, 'clean']['raw']['n']
            if any(len(v) != n for v in data.values()) or len(set(data['sample_id'])) != n:
                raise ValueError(f'prediction identities/lengths invalid: {path}')
            reference = predictions.get(('E0', seeds['E0'][0]))
            if reference is not None and any(not np.array_equal(data[k], reference[k])
                                             for k in ('sample_id', 'actual_class', 'actual')):
                raise ValueError('groups/seeds use different sample order or labels')
            for decoder, key in (('raw', 'raw_score'), ('coherent', 'coherent_score')):
                metric = lookup[g, seed, 'clean'][decoder]
                accuracy = np.mean(data['actual_class'] == data['predicted'])
                mae = np.mean(np.abs(data['actual'] - data[key]))
                if not np.isclose(accuracy, metric['accuracy']) or not np.isclose(mae, metric['mae']):
                    raise ValueError('saved predictions disagree with grid metrics')
            predictions[g, seed] = data
            sources.append(path)
    out = root / 'paper' / split
    out.mkdir(parents=True, exist_ok=True)
    flat, summaries, per_seed = [], [], []
    scopes = {'clean': ['clean'], 'missing_mean': [c for c in CASES if c != 'clean']}
    scopes.update({f'mode:{m}': [c for c, x in CASES.items() if x[0] == m] for m in MODES})
    scopes.update({f'rate:{r}': [c for c, x in CASES.items() if x[1] == r] for r in RATES})
    scopes.update({f'position:{p}': [c for c, x in CASES.items() if x[2] == p] for p in POSITIONS})
    for g in GROUPS:
        for decoder in ('raw', 'coherent'):
            for seed in seeds[g]:
                for case, (mode, rate, position) in CASES.items():
                    metrics = lookup[g, seed, case][decoder]
                    flat.append(dict(group=g, seed=seed, decoder=decoder, case=case,
                        mode=mode, rate=rate, position=position,
                        **{k: metrics[k] for k in METRICS},
                        **{f'f1_{label}': metrics['per_class_f1'][i]
                           for i, label in enumerate(('negative', 'neutral', 'positive'))}, n=metrics['n']))
            for scope, cases in scopes.items():
                for seed in seeds[g]:
                    per_seed.append(dict(group=g, seed=seed, decoder=decoder, scope=scope,
                        **{k: float(np.mean([lookup[g, seed, c][decoder][k] for c in cases]))
                           for k in METRICS}))
                for metric in METRICS:
                    values = [float(np.mean([lookup[g, s, c][decoder][metric] for c in cases]))
                              for s in seeds[g]]
                    summaries.append(dict(group=g, decoder=decoder, scope=scope, metric=metric,
                                          seeds=len(values), **stats(values)))
            values = [min(lookup[g, s, c][decoder]['macro_f1'] for c in scopes['missing_mean'])
                      for s in seeds[g]]
            summaries.append(dict(group=g, decoder=decoder, scope='missing_worst', metric='macro_f1',
                                  seeds=len(values), **stats(values)))
    csv_out(out / 'grid.csv', flat)
    csv_out(out / 'per_seed.csv', per_seed)
    csv_out(out / 'summary.csv', summaries)
    class_summary = []
    for g in GROUPS:
        for i, label in enumerate(('negative', 'neutral', 'positive')):
            class_summary.append(dict(group=g, label=label, **stats([
                lookup[g, s, 'clean']['raw']['per_class_f1'][i] for s in seeds[g]])))
    csv_out(out / 'per_class_f1.csv', class_summary)
    locked = read(root / 'locked_selection.json')
    selected_group, selected_seed = locked['group'], locked['selected']['spec']['seed']
    selected = []
    for decoder in ('raw', 'coherent'):
        for scope in ('clean', 'missing_mean'):
            selected.append(dict(group=selected_group, seed=selected_seed, decoder=decoder, scope=scope,
                **{k: float(np.mean([lookup[selected_group, selected_seed, c][decoder][k]
                                    for c in scopes[scope]])) for k in METRICS},
                worst_macro_f1=min(lookup[selected_group, selected_seed, c][decoder]['macro_f1']
                                   for c in scopes[scope])))
    csv_out(out / 'selected_model.csv', selected)
    paired = []
    for g in list(GROUPS)[1:]:
        for decoder in ('raw', 'coherent'):
            for scope in ('clean', 'missing_mean'):
                cases = scopes[scope]
                for metric in METRICS:
                    values = [float(np.mean([lookup[g, s, c][decoder][metric] -
                              lookup['E0', s, c][decoder][metric] for c in cases])) for s in seeds[g]]
                    paired.append(dict(group=g, baseline='E0', decoder=decoder, scope=scope,
                                       metric=metric, **stats(values)))
    csv_out(out / 'paired_delta_vs_E0.csv', paired)
    configurations = []
    for g in GROUPS:
        for run in final[g]['runs']:
            spec = run['spec']
            configurations.append(dict(group=g, seed=spec['seed'], trial_id=run['trial_id'],
                best_epoch=run['best_epoch'], epochs=spec['epochs'],
                audio_kept=len(spec['mask'].get('audio_indices', range(74))),
                vision_kept=len(spec['mask'].get('vision_indices', range(35))),
                parameters=json.dumps(spec['parameters'], sort_keys=True),
                mask=json.dumps(spec['mask'], sort_keys=True)))
    csv_out(out / 'configurations.csv', configurations)
    # Each unique trial is counted once. Resume histories contain prior epochs,
    # so use measured invocation wall time instead of summing cumulative histories.
    costs = []
    for path in sorted((root / 'trials').glob('*/result.json')):
        r = read(path)
        peaks = [h['peak_cuda_bytes'] for h in r.get('history', []) if h.get('peak_cuda_bytes') is not None]
        costs.append(dict(trial_id=r['trial_id'], status=r['status'],
            requested_epochs=r['spec']['epochs'], wall_seconds=r.get('wall_seconds'),
            peak_cuda_bytes=max(peaks) if peaks else None))
    csv_out(out / 'unique_trial_costs.csv', costs)
    requests = root / 'requests.jsonl'
    if requests.exists():
        csv_out(out / 'requests.csv', [dict(trial_id=r['trial_id'], phase=r['phase'],
            cache_hit=r['cache_hit'], status=r['status'])
            for r in (json.loads(line) for line in requests.read_text().splitlines() if line.strip())])
    draw(out, split, final, lookup, seeds, predictions, root)
    sources.extend(p for p in (root / 'prepared.json', root / 'hpo.json', root / 'features.json') if p.exists())
    provenance = dict(split=split, source_sha256={str(p.relative_to(root)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        f'# Optimization report: {split}\n\n'
        'E0 Default; E1 Random HPO; E2 DE; E3 DE + random subset; E4 DE + top-k; E5 DE + BPSO.\n\n'
        'Raw and coherent outputs are separate. All errors bars are sample standard deviations across training seeds. '
        'Conditions are averaged within each seed first; 63 conditions are not 63 independent repeats. '
        'Single-seed standard deviations are blank. Paired deltas are method minus E0 (negative MAE is better).\n\n'
        'Confusion matrices and scatter plots use each group\'s best VALIDATION selection seed, even on test. '
        'The representative seed is in the filename; no seed is selected by test performance. '
        'Means across seeds and the selected single-model metrics must not be conflated.\n\n'
        'Validation is used for selection and is not an independent test. '
        'B historical 61.68% used up to 35 epochs/early stopping; E0 is retrained at the current equal budget. '
        'Identical configurations after fallback may share trials and are not separate improvements. '
        'Feature masks do not reduce dense-model FLOPs. '
        'Search is a single search seed; training repeats do not establish search-seed stability.\n\n'
        'Trial costs exclude preparation, filtering, full-grid evaluation and unreported interrupted attempts; '
        'cached requests are not additional training cost. No significance claim follows from these tables alone.\n',
        encoding='utf-8')
    return out


def draw(out, split, final, lookup, seeds, predictions, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{extension}', dpi=180)
        plt.close(fig)

    for g in GROUPS:
        seed = max(final[g]['runs'], key=lambda r: r['final']['selection_score'])['spec']['seed']
        data = predictions[g, seed]
        cm = confusion_matrix(data['actual_class'], data['predicted'], labels=[0, 1, 2])
        csv_out(out / f'{g}_{seed}_confusion.csv', [dict(actual=i, negative=row[0], neutral=row[1],
                                                      positive=row[2]) for i, row in enumerate(cm)])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=['Neg', 'Neu', 'Pos'],
               yticklabels=['Neg', 'Neu', 'Pos'], xlabel='Predicted', ylabel='Actual',
               title=f'{split}: {g}, seed {seed}')
        save(fig, f'{g}_{seed}_confusion')
        for decoder, key in (('raw', 'raw_score'), ('coherent', 'coherent_score')):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(data['actual'], data[key], alpha=.3, s=12)
            ax.plot([-3, 3], [-3, 3], color='red')
            ax.set(xlabel='Actual intensity', ylabel='Predicted intensity',
                   title=f'{split}: {g}, {decoder}, seed {seed}')
            save(fig, f'{g}_{seed}_{decoder}_regression')
    for decoder in ('raw', 'coherent'):
        for metric in ('macro_f1', 'mae'):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
            for ax, mode in zip(axes, ('text', 'audio', 'vision')):
                for g in GROUPS:
                    values = np.array([[np.mean([lookup[g, s, case_name(mode, rate, p)][decoder][metric]
                                                 for p in POSITIONS]) for rate in RATES] for s in seeds[g]])
                    ax.errorbar(RATES, values.mean(0),
                        yerr=values.std(0, ddof=1) if len(values) > 1 else None, label=g, marker='o')
                ax.set(title=f'{split}: {mode}, {decoder}', xlabel='Missing fraction', ylabel=metric)
            axes[-1].legend()
            save(fig, f'{decoder}_missing_{metric}')
    if (root / 'hpo.json').exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, item in read(root / 'hpo.json')['methods'].items():
            best = -np.inf
            values = []
            for r in item['records']:
                best = max(best, r['score'] if r['score'] is not None else -np.inf)
                values.append(best if np.isfinite(best) else np.nan)
            ax.plot(np.arange(1, len(values) + 1), values, label=method)
        ax.set(xlabel='Logical candidate requests (includes failures/cache hits)',
               ylabel='Best terminal validation selection score', title='HPO screening budget')
        ax.legend()
        save(fig, 'hpo_search')
    if (root / 'features.json').exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, item in read(root / 'features.json').items():
            for c in item['candidates']:
                subset = c['subset']
                trace = subset.get('best_trace', [subset['filter_score']])
                step = subset['evaluations'] // len(trace)
                ax.plot(np.arange(1, len(trace) + 1) * step, trace,
                        label=f"{method}, q={c['ratio']}", marker='.' if len(trace) == 1 else None)
        ax.set(xlabel='Filter evaluations', ylabel='Train-only filter score (not prediction F1)',
               title='Feature search')
        ax.legend()
        save(fig, 'feature_search')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='existing optimization run directory')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    print(report(args.output.resolve(), args.split))
