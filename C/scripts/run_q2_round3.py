"""Accuracy-led, grouped Q2 experiments. Old round-two artifacts are read-only."""
from __future__ import annotations
import argparse
import os
import sys
import time
import pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'C/outputs/q2_round3_env'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from B.src.data import load_npz, apply_audio_vision_scale, assemble_sample, encode_text
from B.src.fusion import Fusion
from C.src.q2_protocol import ViewDataset, base_masks, case_name, SELECT_CASES, MODES, RATES, POSITIONS, _encode_with_masks
from C.src.missingness import corrupt_masks
from C.src.ensemble_search import metrics
from C.scripts.run_q2_corrected import seed_all
from C.scripts.run_q2_round2 import subset, session_for
from C.scripts.run_q2_optimization import read, write, digest

OLD = ROOT / 'C/outputs/q2_optimization_local_myenv_v1'
R2 = ROOT / 'C/outputs/q2_round2_ensemble_v1'
OUT = ROOT / 'C/outputs/q2_round3_accuracy_v1'
CFG = ROOT / 'C/configs/q2_round3_accuracy.json'
MODEL = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DEVICE = 'cuda'


def masks_for(data, scenario):
    return np.stack([x.numpy() for x in (base_masks(data) if scenario is None else
                                        corrupt_masks(base_masks(data), *scenario))], 1)


class TrainData(Dataset):
    def __init__(self, data, probability, online=False):
        self.clean = ViewDataset(data)
        self.views = ViewDataset(data, R2 / 'fit_views')
        self.keep = np.load(OUT / 'keep_half.npy')
        self.probability, self.online, self.epoch = probability, online, 0

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        view = (i * 7 + self.epoch) % len(self.views.text)
        self.views.epoch = self.epoch
        sample = list((self.clean if self.probability == .4 and not self.keep[view, i]
                       else self.views)[i])
        if self.online:
            ids = self.clean.base['token_ids'][i].copy().astype(np.int64)
            ids[self.clean.base['tmask'][i] & ~sample[3].numpy()] = 100
            sample[0] = torch.from_numpy(ids)
            sample[6] = torch.from_numpy(self.clean.base['attention'][i].copy())
        return tuple(sample)


class TextFusion(torch.nn.Module):
    def __init__(self, tuned):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(str(MODEL), local_files_only=True, attn_implementation='eager')
        self.encoder.requires_grad_(False)
        self.tuned = tuned
        if tuned:
            self.encoder.encoder.layer[-2:].requires_grad_(True)
        self.fusion = Fusion()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.tuned and mode:
            self.encoder.encoder.layer[-2:].train()
        return self

    def forward(self, ids, audio, vision, tm, am, vm, attention):
        text = self.encoder(input_ids=ids.long(), attention_mask=attention.long()).last_hidden_state
        return self.fusion(text, audio, vision, tm, am, vm)


def forward(model, batch):
    b = [x.to(DEVICE) for x in batch]
    logits, reg, _ = model(*b[:7]) if isinstance(model, TextFusion) else model(*b[:6])
    return logits, reg, b[-2], b[-1]


def predict(model, data, masks, text=None):
    online = isinstance(model, TextFusion)
    changed = dict(data)
    for j, key in enumerate(('tmask', 'amask', 'vmask')):
        changed[key] = masks[:, j]
    if text is not None:
        changed['text'] = text
    ds = ViewDataset(changed)
    ps, rs = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ds), 64):
            samples = [ds[i] for i in range(start, min(start+64, len(ds)))]
            batch = [torch.stack([s[j] for s in samples]) for j in range(9)]
            if online:
                ids = data['token_ids'][start:start+64].copy().astype(np.int64)
                ids[data['tmask'][start:start+64] & ~masks[start:start+64, 0]] = 100
                batch[0] = torch.from_numpy(ids)
                batch[6] = torch.from_numpy(data['attention'][start:start+64].copy())
            logits, reg, _, _ = forward(model, batch)
            ps.append(logits.softmax(-1).cpu().numpy())
            rs.append(reg.cpu().numpy())
    return np.concatenate(ps), np.concatenate(rs)


def panel(model, data, cases):
    values = [predict(model, data, m, t) for m, t in cases]
    return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])


def key(result):
    row = result['scenarios'][0]
    return row['accuracy'], row['macro_f1'], -row['mae']


def guard(result, baseline, cfg):
    r, b = result['scenarios'], baseline['scenarios']
    return (r[0]['macro_f1'] >= b[0]['macro_f1']-cfg['guard_f1']-1e-12 and
            np.mean([x['macro_f1'] for x in r[1:]]) >= np.mean([x['macro_f1'] for x in b[1:]])-cfg['guard_f1']-1e-12 and
            np.mean([x['mae'] for x in r]) <= np.mean([x['mae'] for x in b])+cfg['guard_mae']+1e-12)


class Run:
    def __init__(self):
        self.cfg = read(CFG)
        OUT.mkdir(parents=True, exist_ok=True)
        sources = [Path(__file__), CFG, ROOT/'B/src/fusion.py', ROOT/'B/src/data.py',
                   ROOT/'C/src/q2_protocol.py', ROOT/'C/src/missingness.py', ROOT/'C/src/ensemble_search.py',
                   ROOT/'C/scripts/run_q2_corrected.py', ROOT/'C/scripts/run_q2_round2.py',
                   ROOT/'C/scripts/run_q2_optimization.py', R2/'splits.json', R2/'normalization.npz',
                   R2/'fit_views/train_masks.npy', R2/'fit_views/train_text.npy', R2/'fit_views/sample_ids.json',
                   OLD/'prepared/train.npz', OLD/'prepared/valid.npz', OLD/'prepared/text_encoder_int8.onnx',
                   MODEL/'config.json', MODEL/'model.safetensors']
        sources += [d/case_name(*s) for d in (R2/'stop_cases', OLD/'prepared/valid_cases') for s in SELECT_CASES]
        self.signature = {str(p): digest(p) for p in sources}
        manifest = OUT / 'manifest.json'
        if manifest.exists():
            assert read(manifest)['source_hashes'] == self.signature, 'Sources changed: use a fresh output'
        else:
            import transformers
            write(manifest, dict(source_hashes=self.signature, config=self.cfg, python=sys.version,
                  torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                  created=time.strftime('%Y-%m-%d %H:%M:%S')))
        self.run_hash = digest(manifest)
        torch.set_num_threads(1)
        self.split = read(R2 / 'splits.json')
        raw = load_npz(OLD / 'prepared/train.npz')
        self.fit = subset(raw, self.split['fit_indices'])
        self.stop = subset(raw, self.split['stop_indices'])
        self.valid = load_npz(OLD / 'prepared/valid.npz')
        z = load_npz(R2 / 'normalization.npz')
        self.scale = {k: (z[k+'_mean'], z[k+'_std']) for k in ('audio', 'vision')}
        for data in (self.fit, self.stop, self.valid):
            apply_audio_vision_scale(data, self.scale)
        assert read(R2/'fit_views/sample_ids.json') == self.fit['sample_id'].tolist()
        self.stop_cases = self.cases(self.stop, R2/'stop_cases')
        self.valid_cases = self.cases(self.valid, OLD/'prepared/valid_cases')

    def cases(self, data, directory):
        result = [(masks_for(data, None), data['text'])]
        for scenario in SELECT_CASES:
            z = load_npz(directory/case_name(*scenario))
            assert np.array_equal(z['masks'], masks_for(data, scenario))
            result.append((z['masks'], z['text']))
        return result

    def done(self, name):
        path = OUT/'state'/f'{name}.json'
        if not path.exists():
            return False
        obj = read(path)
        assert obj['run_hash'] == self.run_hash
        for p, h in obj['artifacts'].items():
            assert digest(p) == h, f'Artifact changed: {p}'
        return True

    def finish(self, name, files, start):
        write(OUT/'state'/f'{name}.json', dict(run_hash=self.run_hash, seconds=time.perf_counter()-start,
              artifacts={str(p): digest(p) for p in files}))
        print('COMPLETE', name, round(time.perf_counter()-start, 2), flush=True)

    def prepare(self):
        if self.done('prepare'):
            return
        start = time.perf_counter()
        old = read(OLD/'prepared.json')
        for name in ('train.npz', 'valid.npz', 'text_encoder_int8.onnx'):
            assert digest(OLD/'prepared'/name) == old['artifacts'][name]
        keep = np.random.default_rng(20262000).random((8, len(self.fit['cls']))) < .5
        np.save(OUT/'keep_half.npy', keep)
        masks = np.load(R2/'fit_views/train_masks.npy', mmap_mode='r')
        base = masks_for(self.fit, None)
        changed = np.any(masks != base[None], axis=(2, 3))
        write(OUT/'preparation.json', dict(original_changed=float(changed.mean()),
              thinned_changed=float((changed & keep).mean()), fit=len(self.fit['cls']), stop=len(self.stop['cls']),
              valid=len(self.valid['cls']), convention='Input-level UNK, same masks for int8 and full-precision'))
        self.finish('prepare', [OUT/'keep_half.npy', OUT/'preparation.json'], start)

    def train_one(self, name, recipe, seed, kind='int8'):
        if self.done(name):
            return
        start = time.perf_counter()
        seed_all(seed)
        online = kind != 'int8'
        model = (TextFusion(kind == 'tuned') if online else Fusion()).to(DEVICE)
        groups = [dict(params=model.fusion.parameters(), lr=self.cfg['fusion_lr']),
                  dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=self.cfg['text_lr'])] if online else [dict(params=model.parameters(), lr=self.cfg['fusion_lr'])]
        optim = torch.optim.AdamW(groups, weight_decay=self.cfg['weight_decay'])
        ds = TrainData(self.fit, recipe['missing'], online)
        loader = DataLoader(ds, batch_size=self.cfg['text_batch_size' if online else 'batch_size'], shuffle=True)
        counts = np.bincount(self.fit['cls'], minlength=3)
        weights = torch.tensor(np.sqrt(counts.sum()/(3*counts)), dtype=torch.float32, device=DEVICE) if recipe['weighted'] else None
        ce = torch.nn.CrossEntropyLoss(weight=weights)
        history, best, state = [], None, None
        path = OUT/'models'/name
        path.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.cfg['epochs']):
            ds.epoch = epoch
            model.train()
            losses = []
            for batch in loader:
                optim.zero_grad(set_to_none=True)
                logits, reg, cls, score = forward(model, batch)
                loss = ce(logits, cls) + recipe['reg'] * torch.nn.functional.smooth_l1_loss(reg, score)
                assert torch.isfinite(loss), 'nonfinite loss'
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optim.step()
                losses.append(float(loss.detach()))
            p, r = predict(model, self.stop, *self.stop_cases[0])
            result = metrics(p[None], r[None], self.stop['cls'], self.stop['score'])
            history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), clean=result['scenarios'][0]))
            if best is None or key(result) > key(best):
                best, best_epoch = result, epoch+1
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            write(path/'history.json', history)
            print(name, epoch+1, 'stop_acc', round(key(result)[0], 4), flush=True)
        model.load_state_dict(state)
        torch.save(dict(state_dict=state, kind=kind, seed=seed, recipe=recipe, epoch=best_epoch), path/'best.pt')
        # Only stop predictions are produced at this stage; valid remains outside recipe selection.
        p, r = panel(model, self.stop, self.stop_cases)
        np.savez_compressed(path/'stop.npz', prob=p, reg=r, sample_id=self.stop['sample_id'])
        result = metrics(p, r, self.stop['cls'], self.stop['score'])
        write(path/'result.json', dict(name=name, kind=kind, seed=seed, recipe=recipe, best_epoch=best_epoch,
              stop=result, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        self.finish(name, [path/'best.pt', path/'history.json', path/'stop.npz', path/'result.json'], start)
        del model, state, optim
        torch.cuda.empty_cache()

    def train(self):
        assert self.done('prepare')
        for recipe, cfg in self.cfg['recipes'].items():
            for seed in self.cfg['seeds']:
                self.train_one(f'{recipe}_{seed}', cfg, seed)
        if self.done('recipe'):
            return
        start = time.perf_counter()
        results = {}
        for name in self.cfg['recipes']:
            rows = [read(OUT/'models'/f'{name}_{s}'/'result.json')['stop'] for s in self.cfg['seeds']]
            results[name] = dict(scenarios=[{k: float(np.mean([r['scenarios'][i][k] for r in rows]))
                     for k in ('accuracy', 'macro_f1', 'mae', 'pearson')} for i in range(4)])
        eligible = [n for n in results if guard(results[n], results['baseline'], self.cfg)]
        selected = max(eligible, key=lambda n: (key(results[n])[0], n == 'baseline', key(results[n])[1]))
        write(OUT/'recipe.json', dict(selected=selected, results=results, eligible=eligible, selection_data='stop only'))
        self.finish('recipe', [OUT/'recipe.json'], start)
        print('RECIPE', selected, flush=True)

    def text(self):
        assert self.done('recipe')
        recipe = self.cfg['recipes'][read(OUT/'recipe.json')['selected']]
        for kind in ('frozen', 'tuned'):
            for seed in self.cfg['seeds']:
                self.train_one(f'{kind}_{seed}', recipe, seed, kind)

    def load(self, name):
        assert self.done(name)
        z = torch.load(OUT/'models'/name/'best.pt', map_location='cpu', weights_only=False)
        model = TextFusion(z['kind'] == 'tuned') if z['kind'] != 'int8' else Fusion()
        model.load_state_dict(z['state_dict'])
        return model.to(DEVICE).eval()

    def candidates(self):
        chosen = read(OUT/'recipe.json')['selected']
        result = {}
        for family in dict.fromkeys(('baseline', chosen, 'frozen', 'tuned')):
            names = [f'{family}_{s}' for s in self.cfg['seeds']]
            result[family+'_single'] = names[:1]
            result[family+'_mean'] = names
        result['mixed_mean'] = result[chosen+'_mean'] + result['tuned_mean']
        return result

    def select(self):
        if self.done('select'):
            return
        start = time.perf_counter()
        candidates = self.candidates()
        cache = {}
        for name in dict.fromkeys(n for v in candidates.values() for n in v):
            path = OUT/'valid_predictions'/f'{name}.npz'
            if not self.done('valid_'+name):
                t = time.perf_counter()
                model = self.load(name)
                p, r = panel(model, self.valid, self.valid_cases)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, prob=p, reg=r, sample_id=self.valid['sample_id'])
                self.finish('valid_'+name, [path], t)
                del model
            cache[name] = load_npz(path)
        folds = np.asarray(self.split['valid_fold'])
        y, score = self.valid['cls'], self.valid['score']
        basep = np.mean([cache[n]['prob'] for n in candidates['baseline_mean']], 0)
        baser = np.mean([cache[n]['reg'] for n in candidates['baseline_mean']], 0)
        baseline = metrics(basep, baser, y, score)
        records = {}
        artifacts = []
        for name, models in candidates.items():
            p, r = [np.mean([cache[n][k] for n in models], 0) for k in ('prob', 'reg')]
            for calibrated in (False, True):
                biases, oof = [], p.copy()
                if calibrated:
                    for fold in range(3):
                        train, held = folds != fold, folds == fold
                        bias = fit_bias(p[:, train], r[:, train], y[train], score[train], self.cfg)
                        biases.append(bias)
                        oof[:, held] = apply_bias(p[:, held], bias)
                result = metrics(oof, r, y, score)
                label = name + ('_bias' if calibrated else '')
                records[label] = dict(models=models, calibrated=calibrated, fold_bias=biases,
                     metrics=result, eligible=guard(result, baseline, self.cfg))
                path = OUT/'oof'/f'{label}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, prob=oof, reg=r, sample_id=self.valid['sample_id'], fold=folds)
                artifacts.append(path)
        eligible = [n for n, r in records.items() if r['eligible']]
        selected = sorted(eligible, key=lambda n: (-key(records[n]['metrics'])[0], len(records[n]['models']), records[n]['calibrated'], n))[0]
        chosen = records[selected]
        p, r = [np.mean([cache[n][k] for n in chosen['models']], 0) for k in ('prob', 'reg')]
        bias = fit_bias(p, r, y, score, self.cfg) if chosen['calibrated'] else [0., 0., 0.]
        write(OUT/'selection.json', dict(candidates=records, baseline=baseline, selected=selected))
        write(OUT/'lock.json', dict(selected=selected, models=chosen['models'], bias=bias,
              run_hash=self.run_hash, selection_hash=digest(OUT/'selection.json'),
              note='Internal grouped validation; reused test is post-lock reassessment, not blind. Full precision candidate is not quantized deployment.'))
        self.finish('select', [OUT/'selection.json', OUT/'lock.json', *artifacts], start)
        print('LOCKED', selected, key(chosen['metrics'])[0], bias, flush=True)

    def evaluate(self):
        assert self.done('select')
        if self.done('evaluate'):
            return
        start = time.perf_counter()
        locked = read(OUT/'lock.json')
        aligned = next((ROOT/'data').glob('*/aligned_50.pkl'))
        assert digest(aligned) == read(OLD/'prepared.json')['signature']['source']['aligned_pickle']
        with aligned.open('rb') as f:
            original = pickle.load(f)['test']
        session = session_for(OLD/'prepared/text_encoder_int8.onnx')
        text = encode_text(session, original['text_bert'], 'cpu', batch_size=1)
        data = assemble_sample(original['text_bert'], text, original['audio'], original['vision'],
                               original['classification_labels'], original['regression_labels'])
        data['sample_id'] = np.array([f'test:{i}' for i in range(len(data['cls']))])
        apply_audio_vision_scale(data, self.scale)
        refs = dict(baseline=self.candidates()['baseline_mean'], selected=locked['models'])
        models = {n: self.load(n) for n in dict.fromkeys(n for v in refs.values() for n in v)}
        reports, artifacts, texts = {}, [], {}
        for scenario in [None]+[(m,r,p) for m in MODES for r in RATES for p in POSITIONS]:
            name = 'clean' if scenario is None else case_name(*scenario)[:-4]
            path = OUT/'test'/f'{name}.json'
            predictions = path.with_suffix('.npz')
            if not self.done('test_'+name):
                t = time.perf_counter()
                masks = masks_for(data, scenario)
                token_key = masks[:, 0].tobytes()
                if token_key not in texts:
                    texts[token_key] = data['text'] if np.array_equal(masks[:, 0], data['tmask']) else _encode_with_masks(session, data, masks)
                values = {n: predict(m, data, masks, texts[token_key]) for n, m in models.items()}
                stats, arrays = {}, {}
                for ref, members in refs.items():
                    p, r = [np.mean([values[n][k] for n in members], 0) for k in (0, 1)]
                    if ref == 'selected':
                        p = apply_bias(p, locked['bias'])
                    stats[ref] = metrics(p[None], r[None], data['cls'], data['score'])['scenarios'][0]
                    arrays[ref+'_prob'], arrays[ref+'_reg'] = p, r
                write(path, stats)
                np.savez_compressed(predictions, **arrays, cls=data['cls'], score=data['score'], sample_id=data['sample_id'])
                self.finish('test_'+name, [path, predictions], t)
            reports[name] = read(path)
            artifacts.extend([path, predictions])
        write(OUT/'test_summary.json', dict(lock_hash=digest(OUT/'lock.json'), selected=locked['selected'],
              scenarios=reports, averages={ref: {k: float(np.mean([r[ref][k] for r in reports.values()]))
                    for k in ('accuracy', 'macro_f1', 'mae', 'pearson')} for ref in refs}))
        self.finish('evaluate', [OUT/'test_summary.json', *artifacts], start)

    def report(self):
        assert self.done('evaluate')
        results = read(OUT/'test_summary.json')
        selection = read(OUT/'selection.json')
        recipe = read(OUT/'recipe.json')
        lines = ['# 第三轮准确率优化实验结果', '', '状态：训练、分组验证、锁定与64条件测试复评完成。', '',
                 f"训练配方（仅 stop 选择）：{recipe['selected']}；最终候选：{selection['selected']}。", '',
                 '## 单因素实验（3种子 stop 指标均值）', '', '| 配方 | Accuracy | 宏F1 | 缺失平均宏F1 |', '|---|---:|---:|---:|']
        for name, r in recipe['results'].items():
            rows = r['scenarios']
            lines.append(f"| {name} | {rows[0]['accuracy']:.4f} | {rows[0]['macro_f1']:.4f} | {np.mean([x['macro_f1'] for x in rows[1:]]):.4f} |")
        lines += ['', '## 内部分组验证', '', '| 候选 | Accuracy | 宏F1 | 通过退化约束 |', '|---|---:|---:|---|']
        for name, r in selection['candidates'].items():
            v = r['metrics']['scenarios'][0]
            lines.append(f"| {name} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {r['eligible']} |")
        lines += ['', '## 锁定后测试复评', '', '| 方法 | 干净 Accuracy | 干净宏F1 | 64条件平均 Accuracy | 64条件平均宏F1 | 平均 MAE |', '|---|---:|---:|---:|---:|---:|']
        for name in ('baseline', 'selected'):
            c, a = results['scenarios']['clean'][name], results['averages'][name]
            lines.append(f"| {name} | {c['accuracy']:.4f} | {c['macro_f1']:.4f} | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | {a['mae']:.4f} |")
        lines += ['', '## 复现与边界', '',
                  '- 配置和入口见[实验方案](第三轮准确率优化实验.md)。训练历史、权重、逐样本概率、源文件哈希与锁定记录在 `C/outputs/q2_round3_accuracy_v1/`。',
                  '- 训练配置与 epoch 使用 stop；方法和校准使用 valid。因此验证分数属于内部选择，不能当作独立泛化证据。',
                  '- test 已在历史实验查看，本轮只作锁定后复评；不得据此回调本轮超参数。',
                  '- 微调效果必须与 frozen 配对结果比较；全精度与 int8 的差异不能全部归因于微调。',
                  '- 未自动替换提交包，未做本轮微调模型的 int8 量化部署验证。', '']
        path = ROOT/'docs/C/experiments/第三轮准确率优化实验结果.md'
        path.write_text('\n'.join(lines), encoding='utf-8')
        print('REPORT', path, flush=True)


def apply_bias(prob, bias):
    z = np.log(np.maximum(prob, 1e-8)) + np.asarray(bias)
    z = np.exp(z-z.max(-1, keepdims=True))
    return z/z.sum(-1, keepdims=True)


def fit_bias(p, r, y, scores, cfg):
    base = metrics(p, r, y, scores)
    best, bias = (key(base)[0], 0.), [0., 0., 0.]
    for a in cfg['bias_grid']:
        for b in cfg['bias_grid']:
            value = [a, b, 0.]
            result = metrics(apply_bias(p, value), r, y, scores)
            rank = key(result)[0], -(a*a+b*b)
            if guard(result, base, cfg) and rank > best:
                best, bias = rank, value
    return bias


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all','prepare','train','text','select','evaluate','report'], default='all')
    args = parser.parse_args()
    run = Run()
    for stage in (['prepare','train','text','select','evaluate','report'] if args.stage == 'all' else [args.stage]):
        getattr(run, stage)()
