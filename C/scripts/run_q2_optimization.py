"""C optimization orchestration on the shared corrected Q2 training implementation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from B.src.data import load_npz, encode_text, apply_audio_vision_scale, assemble_sample
from C.src.intelligent_search import decode, parameter_search, filter_statistics, feature_search
from C.src.q2_protocol import (SELECT_CASES, MODES, RATES, POSITIONS, ViewDataset,
                              prepare_views, prepare_case, case_name, evaluate_loader)
from C.scripts.run_q2_corrected import (load_data, loader_for_case, train_one, load_checkpoint)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_hashes():
    files = ['B/src/data.py', 'B/src/fusion.py', 'C/src/missingness.py',
             'C/src/q2_protocol.py', 'C/src/intelligent_search.py',
             'C/scripts/run_q2_corrected.py', 'C/scripts/run_q2_optimization.py']
    return {f: digest(ROOT / f) for f in files}


def fingerprint(config, args):
    if (args.cache is None) == (args.aligned is None) or args.encoder is None:
        raise ValueError('prepare requires exactly one of --cache/--aligned, plus --encoder')
    source = ({'aligned_pickle': digest(args.aligned)} if args.aligned is not None else
              {name: digest(args.cache / f'{name}.npz') for name in ('train', 'valid')})
    return dict(config=config, source=source,
                encoder_sha256=digest(args.encoder), code=code_hashes())


def validate_data(data, expected=None):
    n = len(data['cls'])
    if expected is not None and n != expected:
        raise ValueError(f'expected {expected} rows, got {n}')
    for key, shape in [('text', (n, 50, 384)), ('audio', (n, 50, 74)),
                       ('vision', (n, 50, 35)), ('token_ids', (n, 50)),
                       ('attention', (n, 50)), ('tmask', (n, 50)),
                       ('amask', (n, 50)), ('vmask', (n, 50))]:
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'invalid {key} shape/values')
    if not np.isin(data['cls'], [0, 1, 2]).all():
        raise ValueError('unexpected labels')
    if not np.isfinite(data['score']).all() or np.any(np.abs(data['score']) > 3):
        raise ValueError('invalid intensity labels')


def session_for(encoder):
    import onnxruntime as ort
    return ort.InferenceSession(str(encoder), providers=['CPUExecutionProvider'])


def reencode(data, session):
    bert = np.stack([data['token_ids'].astype(np.int64), data['attention'].astype(np.int64),
                     np.zeros_like(data['token_ids'], dtype=np.int64)], axis=1)
    data['text'] = encode_text(session, bert, 'cpu', batch_size=1)
    return data


def prepare(args, config):
    signature = fingerprint(config, args)
    manifest = args.output / 'prepared.json'
    if manifest.exists():
        if read(manifest)['signature'] != signature:
            raise ValueError('prepared inputs/config/code changed; use a new output directory')
        print('Prepared manifest matches. Use preflight to verify artifacts.')
        return
    directory = args.output / 'prepared'
    if directory.exists():
        raise ValueError('partial preparation exists; preserve it and choose a fresh output directory')
    directory.mkdir(parents=True)
    session = session_for(args.encoder)
    source = None
    if args.aligned is not None:
        with args.aligned.open('rb') as handle:
            source = pickle.load(handle)
    for split, expected in (('train', 3395), ('valid', 728)):
        if source is None:
            data = reencode(load_npz(args.cache / f'{split}.npz'), session)
        else:
            row = source[split]
            text = encode_text(session, row['text_bert'], 'cpu', batch_size=1)
            data = assemble_sample(row['text_bert'], text, row['audio'], row['vision'],
                                   row['classification_labels'], row['regression_labels'])
        # Stable row identity tied to immutable source-file hash; not video IDs.
        data['sample_id'] = np.array([f'{split}:{i}' for i in range(expected)])
        validate_data(data, expected)
        np.savez_compressed(directory / f'{split}.npz', **data)
    source = None
    data = None
    if args.aligned is not None:
        del row, text
    gc.collect()
    train, valid, scale = load_data(directory)
    np.savez_compressed(directory / 'audio_vision_normalization.npz',
                        audio_mean=scale['audio'][0], audio_std=scale['audio'][1],
                        vision_mean=scale['vision'][0], vision_std=scale['vision'][1])
    shutil.copy2(args.encoder, directory / 'text_encoder_int8.onnx')
    prepare_views(train, session, directory / 'views', config['views'], config['mask_seed'])
    for mode, rate, position in SELECT_CASES:
        prepare_case(valid, session, directory / 'valid_cases', mode, rate, position)
    artifacts = {str(p.relative_to(directory)): digest(p)
                 for p in directory.rglob('*') if p.is_file()}
    write(manifest, dict(signature=signature, artifacts=artifacts,
                         versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__)))
    print('Prepared train/valid views only. Test split is not used; Attachment-3 is not loaded.')


def preflight(args, config):
    manifest = read(args.output / 'prepared.json')
    if manifest['signature']['config'] != config or manifest['signature']['code'] != code_hashes():
        raise ValueError('configuration/code differs from preparation; use a new run directory')
    directory = args.output / 'prepared'
    for relative, expected in manifest['artifacts'].items():
        if digest(directory / relative) != expected:
            raise ValueError(f'prepared artifact changed: {relative}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    train, valid, _ = load_data(directory)
    validate_data(train, 3395)
    validate_data(valid, 728)
    print('Preflight passed: immutable corrected cache, 3395/728 rows, 384/74/35 dimensions.')
    return train, valid


class Runner:
    def __init__(self, args, config, train, valid):
        self.args, self.config, self.train, self.valid = args, config, train, valid
        self.prepared = args.output / 'prepared'
        self.clean = loader_for_case(valid, None)
        self.cases = [loader_for_case(valid, self.prepared / 'valid_cases' / case_name(*case))
                      for case in SELECT_CASES]

    def log_request(self, key, spec, cache_hit, status):
        with (self.args.output / 'requests.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(trial_id=key, spec=spec, cache_hit=cache_hit,
                phase=getattr(self.args, 'phase', 'synthetic_check'), status=status)) + '\n')

    def trial(self, parameters, epochs, mask=None, seed=None, resume=None):
        seed = self.config['train_seed'] if seed is None else seed
        mask = mask or {}
        kwargs = dict(dropout=parameters['dropout'], **mask)
        hp = {k: v for k, v in parameters.items() if k != 'dropout'}
        spec = dict(parameters=parameters, epochs=epochs, mask=mask, seed=seed)
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
        directory = self.args.output / 'trials' / key
        result_path = directory / 'result.json'
        if result_path.exists():
            result = read(result_path)
            if result.get('status') == 'ok' and not Path(result['checkpoint']).exists():
                raise ValueError('cached result missing checkpoint')
            self.log_request(key, spec, True, result['status'])
            return result
        write(directory / 'spec.json', spec)
        started = time.perf_counter()
        last = directory / 'checkpoints' / f'robust_gate_{seed}_last.pt'
        if last.exists():
            # A missing result after the final save needs inspection, not silent overwrite.
            state = torch.load(last, map_location='cpu', weights_only=False)
            if state['epoch'] >= epochs:
                raise ValueError(f'completed checkpoint without report: {last}')
            resume = last
        try:
            result = train_one('robust_gate', seed, self.train, self.clean, self.cases,
                               self.prepared / 'views', directory, self.args.device,
                               epochs, epochs + 1, hyperparameters=hp, model_kwargs=kwargs,
                               fixed_budget=True, resume=Path(resume) if resume else None)
            result.update(status='ok', spec=spec, trial_id=key,
                          wall_seconds=time.perf_counter() - started)
        except (torch.OutOfMemoryError, FloatingPointError) as error:
            result = dict(status='failed', spec=spec, trial_id=key, error=str(error),
                          wall_seconds=time.perf_counter() - started)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write(result_path, result)
        self.log_request(key, spec, False, result['status'])
        return result


def value(result, terminal=False):
    return (result['terminal' if terminal else 'final']['selection_score']
            if result['status'] == 'ok' else -float('inf'))


def eligible(result, reference, config):
    if result['status'] != 'ok' or reference['status'] != 'ok':
        return False
    x, y = result['final']['clean'], reference['final']['clean']
    return (x['macro_f1'] >= y['macro_f1'] - config['max_clean_f1_drop'] and
            x['mae'] <= y['mae'] + config['max_clean_mae_increase'])


def pilot(runner):
    c = runner.config
    rng = np.random.default_rng(c['search_seed'])
    parameters = [c['default_parameters']] + [decode(x) for x in rng.random((3, 4))]
    results = [runner.trial(p, c['full_epochs']) for p in parameters]
    if any(r['status'] != 'ok' for r in results):
        raise ValueError('pilot failed: inspect trial logs before searching')
    low = [r['history'][c['low_epochs'] - 1]['selection_score'] for r in results]
    high = [r['terminal']['selection_score'] for r in results]
    from scipy.stats import spearmanr
    correlation = float(spearmanr(low, high).statistic)
    overlap = len(set(np.argsort(low)[-2:]) & set(np.argsort(high)[-2:]))
    safe = np.isfinite(correlation) and correlation > 0 and overlap > 0
    epoch_seconds = [h['seconds'] for r in results for h in r['history']]
    total = (4 * c['full_epochs'] + 2 * (c['search_population'] *
             (c['search_generations'] + 1) * c['low_epochs'] +
             c['promote'] * (c['full_epochs'] - c['low_epochs'])) +
             3 * len(c['feature_ratios']) * c['full_epochs'] +
             6 * len(c['final_seeds']) * c['full_epochs'])
    write(runner.args.output / 'pilot.json', dict(results=results,
          spearman=correlation if np.isfinite(correlation) else None, top2_overlap=overlap,
          low_budget_screening_allowed=bool(safe), budget_epochs=total,
          estimated_hours_including_per_epoch_validation=total * max(epoch_seconds) / 3600,
          note='Four configurations are diagnostic only; estimate excludes preparation and full-grid evaluation.'))


def hpo(runner):
    c, root = runner.config, runner.args.output
    p = read(root / 'pilot.json')
    if not p['low_budget_screening_allowed'] and c['low_epochs'] != c['full_epochs']:
        raise ValueError('pilot ranking failed; set equal low/full epochs in a new configured run')
    baseline = runner.trial(c['default_parameters'], c['full_epochs'])
    if baseline['status'] != 'ok':
        raise ValueError('default baseline failed')
    output = {'baseline': baseline, 'methods': {}}
    for method in ('random', 'de'):
        def evaluate(parameters):
            return value(runner.trial(parameters, c['low_epochs']), terminal=True)
        records = parameter_search(evaluate, method, c['search_population'],
                                   c['search_generations'], c['search_seed'])
        unique = {}
        for record in records:
            if np.isfinite(record['score']):
                unique[json.dumps(record['parameters'], sort_keys=True)] = record
        shortlist = sorted(unique.values(), key=lambda r: r['score'], reverse=True)[:c['promote']]
        promoted = []
        for record in shortlist:
            short = runner.trial(record['parameters'], c['low_epochs'])
            promoted.append(runner.trial(record['parameters'], c['full_epochs'],
                            resume=short['last_checkpoint']) if c['full_epochs'] > c['low_epochs'] else short)
        good = [r for r in promoted if eligible(r, baseline, c)]
        selected = max(good, key=value) if good else baseline
        for record in records:
            if not np.isfinite(record['score']):
                record['score'] = None
        output['methods'][method] = dict(records=records, promoted=promoted, selected=selected,
                                        fallback=not bool(good))
        write(root / 'hpo.json', output)


def features(runner):
    c, root = runner.config, runner.args.output
    h = read(root / 'hpo.json')
    reference = h['methods']['de']['selected']
    parameters = reference['spec']['parameters']
    relevance, redundancy = filter_statistics(runner.train)
    np.savez_compressed(root / 'filter_statistics.npz', relevance=relevance, redundancy=redundancy)
    output = {}
    for method in ('random', 'topk', 'bpso'):
        candidates = []
        for ratio in c['feature_ratios']:
            subset = feature_search(relevance, redundancy, method, ratio, c['search_seed'],
                                    c['feature_population'], c['feature_updates'])
            mask = {k: subset[k] for k in ('audio_indices', 'vision_indices')}
            result = runner.trial(parameters, c['full_epochs'], mask)
            candidates.append(dict(ratio=ratio, subset=subset, result=result))
        good = [x['result'] for x in candidates if eligible(x['result'], reference, c)]
        output[method] = dict(candidates=candidates, selected=max(good, key=value) if good else reference,
                              fallback=not bool(good))
        write(root / 'features.json', output)


def finalize(runner):
    c, root = runner.config, runner.args.output
    h, f = read(root / 'hpo.json'), read(root / 'features.json')
    chosen = {'E0': h['baseline'], 'E1': h['methods']['random']['selected'],
              'E2': h['methods']['de']['selected'], 'E3': f['random']['selected'],
              'E4': f['topk']['selected'], 'E5': f['bpso']['selected']}
    output = {}
    for group, selected in chosen.items():
        spec = selected['spec']
        runs = [runner.trial(spec['parameters'], c['full_epochs'], spec['mask'], seed)
                for seed in c['final_seeds']]
        if any(r['status'] != 'ok' for r in runs):
            raise ValueError(f'{group} final run failed; do not silently discard seed')
        metrics = {k: [r['final']['clean'][k] for r in runs]
                   for k in ('accuracy', 'macro_f1', 'mae', 'pearson')}
        output[group] = dict(runs=runs, selection_mean=float(np.mean([value(r) for r in runs])),
                            clean={k: {'mean': float(np.mean(v)),
                                       'std': float(np.std(v, ddof=1)) if len(v) > 1 else None}
                                   for k, v in metrics.items()})
        write(root / 'final.json', output)
    baseline = output['E0']['clean']
    feasible = [g for g, item in output.items()
                if item['clean']['macro_f1']['mean'] >= baseline['macro_f1']['mean'] - c['max_clean_f1_drop']
                and item['clean']['mae']['mean'] <= baseline['mae']['mean'] + c['max_clean_mae_increase']]
    group = max(feasible, key=lambda g: output[g]['selection_mean'])
    selected = max(output[group]['runs'], key=value)
    write(root / 'locked_selection.json', dict(group=group, selected=selected,
          rule='validation mean score across seeds, clean guards; best validation seed within group'))
    package = root / 'inference'
    package.mkdir(exist_ok=True)
    shutil.copy2(selected['checkpoint'], package / 'model.pt')
    for filename in ('audio_vision_normalization.npz', 'text_encoder_int8.onnx'):
        shutil.copy2(runner.prepared / filename, package / filename)


def evaluate_grid(runner, test=False):
    root = runner.args.output
    final = read(root / 'final.json')
    read(root / 'locked_selection.json')
    report = root / ('test_grid.json' if test else 'validation_grid.json')
    if test and report.exists():
        raise ValueError('test already evaluated; no automatic repeated test use')
    session = session_for(runner.prepared / 'text_encoder_int8.onnx')
    if test:
        if (runner.args.test_cache is None) == (runner.args.aligned is None):
            raise ValueError('test requires exactly one of --test-cache/--aligned')
        if runner.args.aligned is not None:
            source_hash = digest(runner.args.aligned)
            original_hash = read(root / 'prepared.json')['signature']['source'].get('aligned_pickle')
            if original_hash is not None and original_hash != source_hash:
                raise ValueError('test aligned pickle differs from preparation source')
            with runner.args.aligned.open('rb') as handle:
                source = pickle.load(handle)
            row = source['test']
            text = encode_text(session, row['text_bert'], 'cpu', batch_size=1)
            data = assemble_sample(row['text_bert'], text, row['audio'], row['vision'],
                                   row['classification_labels'], row['regression_labels'])
            del source, row, text
            gc.collect()
        else:
            source_hash = digest(runner.args.test_cache)
            data = reencode(load_npz(runner.args.test_cache), session)
        validate_data(data, 727)
        with np.load(runner.prepared / 'audio_vision_normalization.npz') as a:
            scale = {m: (a[f'{m}_mean'], a[f'{m}_std']) for m in ('audio', 'vision')}
        apply_audio_vision_scale(data, scale)
        directory = root / ('test_cases_' + source_hash[:12])
    else:
        data, directory = runner.valid, runner.prepared / 'valid_cases'
    cases = [('clean', None)]
    for mode in MODES:
        for rate in RATES:
            for position in POSITIONS:
                cases.append((case_name(mode, rate, position),
                              prepare_case(data, session, directory, mode, rate, position)))
    # Persist progressively; validation may be resumed, test is deliberately locked on first output.
    rows = []
    split = 'test' if test else 'validation'
    prediction_dir = root / 'predictions' / split
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for group, item in final.items():
        for run in item['runs']:
            model = load_checkpoint(Path(run['checkpoint']), runner.args.device)
            for case, path in cases:
                loader = loader_for_case(data, path)
                raw = evaluate_loader(model, loader, runner.args.device, include_rows=case == 'clean')
                coherent = evaluate_loader(model, loader, runner.args.device, coherent=True,
                                           include_rows=case == 'clean')
                if case == 'clean':
                    predictions = raw.pop('rows')
                    predictions['coherent_score'] = coherent.pop('rows')['score']
                    predictions['sample_id'] = data.get('sample_id', np.array(
                        [f'{split}:{i}' for i in range(len(data['cls']))]))
                    np.savez_compressed(prediction_dir / f"{group}_{run['spec']['seed']}_clean.npz",
                                        **predictions)
                rows.append(dict(group=group, seed=run['spec']['seed'], case=case,
                                 raw=raw, coherent=coherent))
            write(report, dict(complete=False, rows=rows))
    write(report, dict(complete=True, rows=rows,
                      test_source_sha256=source_hash if test else None,
                      note='Raw heads and coherent decoder reported separately; no test-based reselection.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True,
                        choices=['prepare', 'preflight', 'pilot', 'hpo', 'features', 'final', 'evaluate', 'test'])
    parser.add_argument('--config', type=Path, default=ROOT / 'C/configs/q2_optimization.json')
    parser.add_argument('--cache', type=Path, help='source train/valid NPZ cache, used only for preparation')
    parser.add_argument('--aligned', type=Path, help='original aligned_50.pkl for preparation or locked test')
    parser.add_argument('--encoder', type=Path, help='same deployed int8 ONNX encoder')
    parser.add_argument('--test-cache', type=Path, help='raw unnormalized test NPZ, only for locked test phase')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.output = args.output.resolve()
    c = read(args.config)
    if not 1 <= c['low_epochs'] <= c['full_epochs'] or c['views'] < 1:
        raise ValueError('invalid training budget')
    if c['search_population'] < 4 or c['search_generations'] < 0 or c['promote'] < 1:
        raise ValueError('invalid search budget')
    if not c['final_seeds'] or len(set(c['final_seeds'])) != len(c['final_seeds']):
        raise ValueError('final seeds must be distinct and nonempty')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        prepare(args, c)
        return
    train, valid = preflight(args, c)
    if args.phase == 'preflight':
        return
    if (args.output / 'test_grid.json').exists() and args.phase not in ('evaluate', 'test'):
        raise ValueError('test has been opened: this run is locked against further optimization')
    runner = Runner(args, c, train, valid)
    {'pilot': pilot, 'hpo': hpo, 'features': features, 'final': finalize,
     'evaluate': evaluate_grid, 'test': lambda r: evaluate_grid(r, test=True)}[args.phase](runner)


if __name__ == '__main__':
    main()
