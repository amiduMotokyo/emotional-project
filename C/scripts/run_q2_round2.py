"""Resumable complete C round-two experiment. Run --help for stage commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from B.src.data import load_npz, fit_audio_vision_scale, apply_audio_vision_scale, assemble_sample, encode_text
from C.src.q2_protocol import (ViewDataset, SELECT_CASES, MODES, RATES, POSITIONS,
    prepare_views, prepare_case, read_case, case_name, evaluate_loader, base_masks)
from C.src.missingness import corrupt_masks
from C.src.ensemble_search import fit, predict, predict_mean, metrics, select_method
from C.src.q2_group_metadata import read_group_ids
from C.scripts.run_q2_corrected import train_one, load_checkpoint, loader_for_case, selection_result
from C.scripts.run_q2_optimization import digest, read, write, validate_data

METHODS = [f'R{i}' for i in range(8)]
SOURCES = ['C/scripts/run_q2_round2.py', 'C/src/ensemble_search.py', 'C/src/q2_group_metadata.py',
           'C/scripts/run_q2_corrected.py', 'C/scripts/run_q2_optimization.py',
           'C/src/q2_protocol.py', 'C/src/missingness.py', 'B/src/data.py',
           'B/src/fusion.py', 'B/src/temporal_fusion.py']


def session_for(encoder):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(encoder), sess_options=options, providers=['CPUExecutionProvider'])


def subset(data, indices):
    return {key: value[indices].copy() for key, value in data.items()}


class Experiment:
    def __init__(self, args):
        self.args, self.out, self.old = args, args.output.resolve(), args.source.resolve()
        self.cfg = read(args.config)
        self.out.mkdir(parents=True, exist_ok=True)
        self.prepared = self.old / 'prepared'
        self.encoder = self.prepared / 'text_encoder_int8.onnx'
        self.signature = dict(config=self.cfg, code={p: digest(ROOT / p) for p in SOURCES},
                              source=str(self.old), source_manifest=digest(self.old / 'prepared.json'),
                              device=args.device)
        saved = self.out / 'execution_manifest.json'
        if saved.exists():
            if read(saved)['signature'] != self.signature:
                raise ValueError('Execution/source/config changed: use a new output directory.')
        else:
            write(saved, dict(signature=self.signature, environment=dict(python=sys.version,
                platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))
        self.run_hash = digest(saved)
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.set_num_threads(1)

    def artifact(self, path):
        return str(Path(path).resolve())

    def done(self, task):
        marker = self.out / 'state' / f'{task}.json'
        if not marker.exists():
            return False
        state = read(marker)
        if state['run_hash'] != self.run_hash:
            raise ValueError('Task provenance mismatch')
        for path, expected in state['artifacts'].items():
            if not Path(path).is_file() or digest(path) != expected:
                raise ValueError(f'Artifact changed or missing: {path}')
        return True

    def finish(self, task, paths, started):
        write(self.out / 'state' / f'{task}.json', dict(run_hash=self.run_hash,
              completed_at=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=time.perf_counter()-started,
              artifacts={self.artifact(p): digest(p) for p in paths}))
        print('COMPLETE', task, round(time.perf_counter()-started, 2), 'seconds', flush=True)

    def require(self, task):
        if not self.done(task):
            raise ValueError(f'Run prerequisite stage: {task}')

    def verify_source(self):
        old = read(self.old / 'prepared.json')
        for name in ('train.npz', 'valid.npz', 'text_encoder_int8.onnx'):
            if digest(self.prepared / name) != old['artifacts'][name]:
                raise ValueError(f'Old cache integrity failure: {name}')
        # Encoding and mask semantics must match their original producer.
        for name in ('C/src/q2_protocol.py', 'C/src/missingness.py'):
            if digest(ROOT / name) != old['signature']['code'][name]:
                raise ValueError(f'Old cache producer changed: {name}')

    def load_development(self):
        self.verify_source()
        train, valid = [load_npz(self.prepared / f'{s}.npz') for s in ('train', 'valid')]
        return train, valid

    def scaled(self):
        train, valid = self.load_development()
        split = read(self.out / 'splits.json')
        base_fit, stop = subset(train, split['fit_indices']), subset(train, split['stop_indices'])
        scales = load_npz(self.out / 'normalization.npz')
        scale = {k: (scales[k+'_mean'], scales[k+'_std']) for k in ('audio', 'vision')}
        for item in (base_fit, stop, valid):
            apply_audio_vision_scale(item, scale)
        return base_fit, stop, valid, scale

    def prepare(self):
        if self.done('P0'):
            return
        started = time.perf_counter()
        train, valid = self.load_development()
        validate_data(train, 3395)
        validate_data(valid, 728)
        # Link prepared row IDs to original segment IDs; preserve official splits.
        for name, data in [('train', train), ('valid', valid)]:
            if data['sample_id'].tolist() != [f'{name}:{i}' for i in range(len(data['cls']))]:
                raise ValueError('Unexpected row identity; review grouping metadata first')
            for key in ('group_id', 'video_id', 'subject_id'):
                if key in data:
                    raise ValueError('Grouping metadata found: implement verified group split before running')
        expected = read(self.old / 'prepared.json')['signature']['source']['aligned_pickle']
        if digest(self.args.aligned) != expected:
            raise ValueError('ID metadata source differs from prepared source')
        groups = read_group_ids(self.args.aligned)
        if len(groups['train']['segment_ids']) != 3395 or len(groups['valid']['segment_ids']) != 728:
            raise ValueError('Metadata/prepared row count mismatch')
        indices = np.arange(len(train['cls']))
        train_groups = np.array(groups['train']['group_ids'])
        valid_groups = np.array(groups['valid']['group_ids'])
        fi, st = next(StratifiedGroupKFold(10, shuffle=True, random_state=20260925).split(
            indices, train['cls'], train_groups))
        fi, st = np.sort(fi), np.sort(st)
        fold = np.full(len(valid['cls']), -1)
        for k, (fitted, held) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=20260925).split(fold, valid['cls'], valid_groups)):
            assert not set(valid_groups[fitted]) & set(valid_groups[held])
            fold[held] = k
        assert not set(fi) & set(st) and len(fi) + len(st) == 3395
        assert not set(train_groups[fi]) & set(train_groups[st])
        fit_data, stop = subset(train, fi), subset(train, st)
        splits = dict(policy='stratified_video_groups_first_of_10_stop_folds',
                      group_audit='Original id=video_prefix$_$segment; numeric arrays replaced by metadata stubs, only train/valid IDs returned. Prefix is video grouping, not verified subject identity.',
                      source_metadata=groups,
                      official_train_valid_shared_groups=sorted(set(train_groups) & set(valid_groups)),
                      fit_indices=fi.tolist(), stop_indices=st.tolist(), valid_fold=fold.tolist(),
                      fit_ids=fit_data['sample_id'].tolist(), stop_ids=stop['sample_id'].tolist(),
                      valid_ids=valid['sample_id'].tolist(),
                      counts={key: np.bincount(d['cls'], minlength=3).tolist()
                              for key, d in [('fit', fit_data), ('stop', stop), ('valid', valid)]})
        write(self.out / 'splits.json', splits)
        scale = fit_audio_vision_scale(fit_data)
        np.savez_compressed(self.out / 'normalization.npz', **{
            f'{k}_{n}': v[i] for k, v in scale.items() for i, n in enumerate(('mean', 'std'))})
        session = session_for(self.encoder)
        # B's whole-file hash can change for unrelated cache/export functions.
        # Do not waive it: verify ALL reused clean embeddings and masks against
        # the current producer, recording both producer hashes and exact equality.
        old_b = read(self.old / 'prepared.json')['signature']['code']['B/src/data.py']
        current_b = digest(ROOT / 'B/src/data.py')
        if old_b != current_b:
            for data in (train, valid):
                bert = np.stack([data['token_ids'], data['attention'], np.zeros_like(data['token_ids'])], 1)
                encoded = encode_text(session, bert, 'cpu', batch_size=1)
                if not np.array_equal(encoded, data['text']):
                    raise ValueError('Reused clean embedding differs from current encoder')
                assembled = assemble_sample(bert, encoded, data['audio'], data['vision'])
                for key in ('tmask', 'amask', 'vmask', 'token_ids', 'attention'):
                    if not np.array_equal(assembled[key], data[key]):
                        raise ValueError(f'Reused field differs from current producer: {key}')
        write(self.out / 'source_compatibility.json', dict(old_B_hash=old_b, current_B_hash=current_b,
            verification='exact_all_train_valid_text_and_mask_equivalence' if old_b != current_b else 'producer_hash_identical'))
        # Separate fit view generation keeps all indices local and explicitly mapped.
        views = self.out / 'fit_views'
        prepare_views(fit_data, session, views, 8, 20260924)
        write(views / 'sample_ids.json', fit_data['sample_id'].tolist())
        paths = [self.out / 'splits.json', self.out / 'normalization.npz', self.out / 'source_compatibility.json',
                 views / 'train_text.npy', views / 'train_masks.npy', views / 'sample_ids.json']
        for mode, rate, position in SELECT_CASES:
            target = self.out / 'stop_cases' / case_name(mode, rate, position)
            # Incomplete P0 is rebuilt; never trust file existence from failed preparation.
            if target.exists():
                target.unlink()
            paths.append(prepare_case(stop, session, target.parent, mode, rate, position))
        masks = np.load(views / 'train_masks.npy', mmap_mode='r')
        original = np.stack([fit_data[k] for k in ('tmask', 'amask', 'vmask')], 1)
        if np.any(masks & ~original[None]):
            raise AssertionError('View restored an invalid/padded position')
        for case in paths[-3:]:
            _, mask = read_case(case)
            if np.any(mask & ~np.stack([stop[k] for k in ('tmask', 'amask', 'vmask')], 1)):
                raise AssertionError('Stop masks restored padding')
        # Real forward pass, zero training updates, both heads and IDs verified.
        apply_audio_vision_scale(fit_data, scale)
        from B.src.fusion import Fusion
        model = Fusion().to(self.args.device)
        batch = DataLoader(ViewDataset(subset(fit_data, np.arange(8))), batch_size=8)
        rows = evaluate_loader(model, batch, self.args.device, include_rows=True)['rows']
        assert np.allclose(rows['probabilities'].sum(1), 1, atol=1e-6)
        assert np.array_equal(rows['actual_class'], fit_data['cls'][:8])
        write(self.out / 'protocol_manifest.json', dict(run_hash=self.run_hash,
            encoder_sha256=digest(self.encoder), fit_count=len(fi), stop_count=len(st), valid_count=728,
            sample_ids_hash=digest(self.out / 'splits.json'), test_opened=False,
            checks=['source_hashes', 'row_identity', 'split_disjoint', 'fit_only_normalization',
                    'view_identity', 'mask_subset', 'real_forward_probability_and_label_alignment']))
        paths.append(self.out / 'protocol_manifest.json')
        self.finish('P0', paths, started)

    def models(self, library):
        return [(f'{library}_{family}_{seed}', lr, seed)
                for family, lr in zip(('L', 'D', 'H'), self.cfg['base']['learning_rates'])
                for seed in self.cfg['base']['library_seeds'][library]]

    def train(self):
        self.require('P0')
        if self.done('P1'):
            return
        started = time.perf_counter()
        base_fit, stop, _, _ = self.scaled()
        if read(self.out / 'fit_views/sample_ids.json') != base_fit['sample_id'].tolist():
            raise ValueError('Training view row mapping mismatch')
        clean = loader_for_case(stop, None)
        damaged = [loader_for_case(stop, self.out / 'stop_cases' / case_name(*c)) for c in SELECT_CASES]
        artifacts = []
        for library in ('A', 'B'):
            for mid, lr, seed in self.models(library):
                destination = self.out / 'models' / mid
                result = destination / 'result.json'
                if not self.done(mid):
                    t = time.perf_counter()
                    hp = dict(learning_rate=lr, weight_decay=.01, lambda_reg=.8)
                    last = destination / 'checkpoints' / f'robust_gate_{seed}_last.pt'
                    saved = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
                    if saved is not None and saved['epoch'] == 18:
                        checkpoint = last.with_name(f'robust_gate_{seed}.pt')
                        torch.save(dict(state_dict=saved['best_state'], architecture='robust_gate',
                            seed=seed, epoch=saved['best_epoch'], model_kwargs=saved['model_kwargs'],
                            hyperparameters=hp, protocol='input_unk_before_int8_encoder_v1'), checkpoint)
                        model = load_checkpoint(checkpoint, self.args.device)
                        record = dict(architecture='robust_gate', seed=seed, checkpoint=str(checkpoint),
                            best_epoch=saved['best_epoch'], best_selection=saved['best'],
                            epochs_completed=18, history=saved['history'],
                            final=selection_result(model, clean, damaged, self.args.device))
                    else:
                        record = train_one('robust_gate', seed, base_fit, clean, damaged,
                            self.out / 'fit_views', destination, self.args.device, 18, 18,
                            hyperparameters=hp, model_kwargs=dict(dropout=.25), fixed_budget=True,
                            resume=last if last.exists() else None)
                    record.update(model_id=mid, checkpoint_selection='base_stop_only',
                                  run_hash=self.run_hash, wall_seconds=time.perf_counter()-t)
                    write(result, record)
                    self.finish(mid, [result, Path(record['checkpoint']), last], t)
                    print('TRAINING TIMING', mid, record['wall_seconds'], flush=True)
                artifacts += [result, Path(read(result)['checkpoint'])]
        self.finish('P1', artifacts, started)

    def cache(self):
        self.require('P1')
        if self.done('P2'):
            return
        started = time.perf_counter()
        _, _, valid, _ = self.scaled()
        paths = [None] + [self.prepared / 'valid_cases' / case_name(*c) for c in SELECT_CASES]
        old = read(self.old / 'prepared.json')['artifacts']
        for path in paths[1:]:
            if digest(path) != old[str(path.relative_to(self.prepared))]:
                raise ValueError('Validation case hash mismatch')
        artifacts = []
        for library in ('A', 'B'):
            probs, regressions, hashes = [], [], {}
            for mid, _, _ in self.models(library):
                cp = Path(read(self.out / 'models' / mid / 'result.json')['checkpoint'])
                model = load_checkpoint(cp, self.args.device)
                p, r = [], []
                for case in paths:
                    rows = evaluate_loader(model, loader_for_case(valid, case), self.args.device, include_rows=True)['rows']
                    assert np.array_equal(rows['actual_class'], valid['cls'])
                    assert np.array_equal(rows['actual'], valid['score'])
                    p.append(rows['probabilities'])
                    r.append(rows['raw_score'])
                probs.append(p)
                regressions.append(r)
                hashes[mid] = digest(cp)
            cache = self.out / f'predictions_{library}.npz'
            np.savez_compressed(cache, probs=np.asarray(probs, dtype=np.float64), reg=np.asarray(regressions, dtype=np.float64),
                sample_id=valid['sample_id'], cls=valid['cls'], score=valid['score'],
                model_id=np.array([x[0] for x in self.models(library)]),
                scenario_id=np.array(['clean', 'T_middle_0.3', 'A_middle_0.3', 'V_middle_0.3']))
            write(cache.with_suffix('.json'), dict(run_hash=self.run_hash, checkpoints=hashes,
                source=digest(self.prepared / 'valid.npz'), split_hash=digest(self.out / 'splits.json'),
                encoder_sha256=digest(self.encoder), cases={str(p): digest(p) for p in paths[1:]}))
            artifacts.extend([cache, cache.with_suffix('.json')])
        self.finish('P2', artifacts, started)

    def fits(self, method, data, indices):
        seeds = [11, 23, 37] if method in METHODS[3:] else [11]
        return [fit(method, data['probs'][:, :, indices], data['reg'][:, :, indices],
                    data['cls'][indices], data['score'][indices], data['sample_id'][indices], seed) for seed in seeds]

    def search(self):
        self.require('P2')
        if self.done('P3'):
            return
        if (self.out / 'lock.json').exists():
            raise ValueError('Selection locked; no new search permitted')
        started = time.perf_counter()
        folds = np.array(read(self.out / 'splits.json')['valid_fold'])
        results, artifacts, requests = {}, [], 0
        for library in ('A', 'B'):
            data = load_npz(self.out / f'predictions_{library}.npz')
            results[library] = {}
            for method in METHODS:
                task = f'search_{library}_{method}'
                result_path = self.out / 'search' / f'{library}_{method}.json'
                pred_path = result_path.with_suffix('.npz')
                if not self.done(task):
                    t = time.perf_counter()
                    oofp = np.zeros_like(data['probs'][0])
                    oofr = np.zeros_like(data['reg'][0])
                    seedp = np.zeros((3 if method in METHODS[3:] else 1, *oofp.shape))
                    seedr = np.zeros(seedp.shape[:-1])
                    reports, supports, effective = [], [], []
                    for fold in range(3):
                        train_idx, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                        solutions = self.fits(method, data, train_idx)
                        p, r = predict_mean(solutions, data['probs'][:, :, held], data['reg'][:, :, held])
                        oofp[:, held], oofr[:, held] = p, r
                        for i, solution in enumerate(solutions):
                            sp, sr = predict(solution, data['probs'][:, :, held], data['reg'][:, :, held])
                            seedp[i][:, held], seedr[i][:, held] = sp, sr
                        fp, fr = predict_mean(solutions, data['probs'][:, :, train_idx], data['reg'][:, :, train_idx])
                        weights = np.mean([s['weights'] for s in solutions], axis=0)
                        supports.append(int(np.count_nonzero(weights)))
                        effective.append(float(1 / (weights @ weights)))
                        reports.append(dict(fold=fold, fit_ids=data['sample_id'][train_idx].tolist(),
                            held_ids=data['sample_id'][held].tolist(), solutions=solutions,
                            fit_metrics=metrics(fp, fr, data['cls'][train_idx], data['score'][train_idx]),
                            held_metrics=metrics(p, r, data['cls'][held], data['score'][held])))
                    report = dict(oof=metrics(oofp, oofr, data['cls'], data['score']), folds=reports,
                        seed_oof=[metrics(p, r, data['cls'], data['score']) for p, r in zip(seedp, seedr)],
                        mean_nonzero_models=float(np.mean(supports)), mean_effective_models=float(np.mean(effective)),
                        requests=sum(s['requests'] for f in reports for s in f['solutions']))
                    write(result_path, report)
                    np.savez_compressed(pred_path, probs=oofp, reg=oofr, sample_id=data['sample_id'],
                                        cls=data['cls'], score=data['score'], fold=folds)
                    self.finish(task, [result_path, pred_path], t)
                row = read(result_path)
                results[library][method] = {k: v for k, v in row.items() if k != 'folds'}
                if method in METHODS[3:]:
                    requests += row['requests']
                artifacts.extend([result_path, pred_path])
                print('OOF', library, method, row['oof']['selection_score'], flush=True)
        assert requests == 27000
        write(self.out / 'crossfit_summary.json', dict(results=results, search_requests=requests,
              note='OOF used for method selection; internal validation, not unbiased final evaluation'))
        self.finish('P3', artifacts + [self.out / 'crossfit_summary.json'], started)

    def lock(self):
        self.require('P3')
        if self.done('P4'):
            return
        started = time.perf_counter()
        path = self.out / 'lock.json'
        if path.exists():
            locked = read(path)
            if locked['run_hash'] != self.run_hash or locked['summary_hash'] != digest(self.out / 'crossfit_summary.json'):
                raise ValueError('Lock provenance mismatch')
        else:
            summary = read(self.out / 'crossfit_summary.json')
            locked = dict(**select_method(summary['results']), run_hash=self.run_hash,
                summary_hash=digest(self.out / 'crossfit_summary.json'), deployment_library='A',
                locked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                historical_fallback_hash=digest(self.old / 'locked_selection.json'),
                test_policy='post_lock_reassessment_not_new_blind_test')
            write(path, locked)
        data = load_npz(self.out / 'predictions_A.npz')
        indices = np.arange(len(data['cls']))
        final = {m: self.fits(m, data, indices) for m in ('R0', 'R2')}
        if locked['selected'] and locked['selected'] not in final:
            final[locked['selected']] = self.fits(locked['selected'], data, indices)
        write(self.out / 'final_ensembles.json', dict(lock_hash=digest(path), ensembles=final))
        self.finish('P4', [path, self.out / 'final_ensembles.json'], started)
        print('LOCKED', locked['decision'], locked['selected'], flush=True)

    def evaluate(self):
        self.require('P4')
        if self.done('P5'):
            return
        started = time.perf_counter()
        locked = read(self.out / 'lock.json')
        final = read(self.out / 'final_ensembles.json')
        if final['lock_hash'] != digest(self.out / 'lock.json'):
            raise ValueError('Ensemble/lock mismatch')
        _, _, valid, scale = self.scaled()
        session = session_for(self.encoder)
        expected = read(self.old / 'prepared.json')['signature']['source']['aligned_pickle']
        if digest(self.args.aligned) != expected:
            raise ValueError('Test source differs from original attachment')
        # First NUMERIC raw pickle deserialization happens only after P4 verification.
        with self.args.aligned.open('rb') as handle:
            original = pickle.load(handle)['test']
        text = encode_text(session, original['text_bert'], 'cpu', batch_size=1)
        test = assemble_sample(original['text_bert'], text, original['audio'], original['vision'],
                               original['classification_labels'], original['regression_labels'])
        test['sample_id'] = np.array([f'test:{i}' for i in range(len(test['cls']))])
        validate_data(test, 727)
        apply_audio_vision_scale(test, scale)
        del original
        models = [load_checkpoint(Path(read(self.out / 'models' / mid / 'result.json')['checkpoint']),
                                  self.args.device) for mid, _, _ in self.models('A')]
        scenarios = [None] + [(m, r, p) for m in MODES for r in RATES for p in POSITIONS]
        artifacts = []
        for split, data in [('valid', valid), ('test', test)]:
            reports = {}
            for scenario in scenarios:
                name = 'clean' if scenario is None else case_name(*scenario).removesuffix('.npz')
                task = f'eval_{split}_{name}'
                result_path = self.out / 'evaluation' / split / f'{name}.json'
                pred_path = result_path.with_suffix('.npz')
                if not self.done(task):
                    t = time.perf_counter()
                    if scenario is None:
                        changed = data
                    else:
                        masks = np.stack([x.numpy() for x in corrupt_masks(base_masks(data), *scenario)], 1)
                        # Encoder depends only on text mask, not audio/vision loss.
                        # Reuse identical text requests within this locked stage.
                        key = hashlib.sha256(masks[:, 0].tobytes()).hexdigest()
                        enc_path = self.out / 'evaluation' / split / 'text_cache' / f'{key}.npy'
                        enc_meta = enc_path.with_suffix('.json')
                        if enc_meta.exists():
                            if read(enc_meta) != dict(run_hash=self.run_hash, sha256=digest(enc_path)):
                                raise ValueError('Evaluation text cache mismatch')
                            damaged_text = np.load(enc_path)
                        else:
                            from C.src.q2_protocol import _encode_with_masks
                            damaged_text = _encode_with_masks(session, data, masks)
                            enc_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(enc_path, damaged_text)
                            write(enc_meta, dict(run_hash=self.run_hash, sha256=digest(enc_path)))
                        changed = dict(data, text=damaged_text)
                        for i, k in enumerate(('tmask', 'amask', 'vmask')):
                            changed[k] = masks[:, i]
                    prepared_at = time.perf_counter()
                    probs, reg = [], []
                    for model in models:
                        rows = evaluate_loader(model, loader_for_case(changed, None), self.args.device, include_rows=True)['rows']
                        assert np.array_equal(rows['actual_class'], data['cls'])
                        probs.append(rows['probabilities'])
                        reg.append(rows['raw_score'])
                    probs, reg = np.asarray(probs)[:, None], np.asarray(reg)[:, None]
                    forward_at = time.perf_counter()
                    report, predictions = {}, {}
                    for method, solutions in final['ensembles'].items():
                        p, r = predict_mean(solutions, probs, reg)
                        cls = p.argmax(-1)
                        coherent = np.where(cls == 1, 0, np.where(cls == 2, np.abs(r), -np.abs(r)))
                        report[method] = dict(raw=metrics(p, r, data['cls'], data['score']),
                                             coherent=metrics(p, coherent, data['cls'], data['score']))
                        predictions[method+'_probs'], predictions[method+'_reg'] = p, r
                    write(result_path, dict(metrics=report, lock_hash=digest(self.out / 'lock.json'),
                        preparation_seconds=prepared_at-t, forward_seconds=forward_at-prepared_at,
                        postprocess_seconds=time.perf_counter()-forward_at))
                    np.savez_compressed(pred_path, **predictions, sample_id=data['sample_id'], cls=data['cls'], score=data['score'])
                    self.finish(task, [result_path, pred_path], t)
                reports[name] = read(result_path)
                artifacts.extend([result_path, pred_path])
            write(self.out / f'{split}_grid.json', dict(lock_hash=digest(self.out / 'lock.json'),
                selected=locked['selected'], complete=True, scenarios=reports,
                note='Fixed R0/R2 references; no round2 deployment candidate if selected is null'))
            artifacts.append(self.out / f'{split}_grid.json')
        self.finish('P5', artifacts, started)

    def report(self):
        self.require('P5')
        if self.done('P6'):
            return
        started = time.perf_counter()
        summary = read(self.out / 'crossfit_summary.json')
        locked = read(self.out / 'lock.json')
        lines = ['# 第二轮完整优化实验结果', '', '状态：完整运行完成。', '',
                 '两套六模型库、18轮训练、三折验证、三个优化种子；R3—R7共27000次搜索请求。',
                 '基础训练仅使用train内部fit/stop；验证集折外分数参与方法选择，属于内部验证。',
                 '原始视频前缀用于fit/stop及valid三折分组，不保证主体隔离；test曾在第一轮查看，本轮仅作锁定后复评。', '',
                 '| 方法 | 库A折外S | 库B折外S | 平均ΔS对R0 |', '|---|---:|---:|---:|']
        for m in METHODS:
            a, b = [summary['results'][l][m]['oof']['selection_score'] for l in ('A', 'B')]
            delta = np.mean([summary['results'][l][m]['oof']['selection_score'] -
                             summary['results'][l]['R0']['oof']['selection_score'] for l in ('A', 'B')])
            lines.append(f'| {m} | {a:.6f} | {b:.6f} | {delta:+.6f} |')
        lines += ['', f"锁定决定：{locked['decision']}；第二轮候选：{locked['selected']}。", '',
                  'R0固定默认单模型，R1折内选单最佳，R2等权，R3贪心，R4随机，R5 DE，R6正则DE，R7正则DE加偏置。', '',
                  '## 锁定后64场景复评', '', '| 数据 | 方法 | 平均S(raw) | clean ACC | clean F1 | clean MAE |',
                  '|---|---|---:|---:|---:|---:|']
        for split in ('valid', 'test'):
            grid = read(self.out / f'{split}_grid.json')['scenarios']
            for m in grid['clean']['metrics']:
                score = np.mean([v['metrics'][m]['raw']['selection_score'] for v in grid.values()])
                c = grid['clean']['metrics'][m]['raw']['scenarios'][0]
                lines.append(f"| {split} | {m} | {score:.6f} | {c['accuracy']:.6f} | {c['macro_f1']:.6f} | {c['mae']:.6f} |")
        timings = {p.stem: read(p)['seconds'] for p in (self.out / 'state').glob('P*.json')}
        write(self.out / 'timings.json', timings)
        lines += ['', '## 阶段计时', '', '已完成任务断点复用时阶段墙钟时间可能不含此前失败尝试；各模型result.json另存训练耗时。', '',
                  *[f'- {k}: {v:.2f} 秒。' for k, v in sorted(timings.items())], '',
                  '来源：crossfit_summary.json、search/、lock.json、final_ensembles.json、valid_grid.json、test_grid.json。',
                  'raw与coherent的逐条件指标、逐样本预测均在evaluation/；判定只使用锁定前的raw四场景指标。', '']
        report = self.out / 'report.md'
        report.write_text('\n'.join(lines), encoding='utf-8')
        self.finish('P6', [report, self.out / 'timings.json'], started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'train', 'cache', 'search', 'lock', 'evaluate', 'report', 'all'], default='all')
    parser.add_argument('--config', type=Path, default=ROOT / 'C/configs/q2_round2_design.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'C/outputs/q2_round2_ensemble_v1')
    parser.add_argument('--source', type=Path, default=ROOT / 'C/outputs/q2_optimization_local_myenv_v1')
    parser.add_argument('--aligned', type=Path, default=ROOT / 'data/附件2-数据集特征文件/aligned_50.pkl')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    experiment = Experiment(args)
    try:
        for phase in ('prepare', 'train', 'cache', 'search', 'lock', 'evaluate', 'report'):
            if args.phase in ('all', phase):
                getattr(experiment, phase)()
    except Exception:
        write(experiment.out / 'failure.json', dict(phase=args.phase, time=time.strftime('%Y-%m-%d %H:%M:%S'),
              traceback=traceback.format_exc(), run_hash=experiment.run_hash))
        raise


if __name__ == '__main__':
    main()
