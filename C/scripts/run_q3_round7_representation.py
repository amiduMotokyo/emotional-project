"""Paired round-seven training, conditional combination, locked refit and validation."""
import sys, time, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'C/outputs/q2_round3_env'))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from C.scripts import run_q3_minilm_round6 as base
from B.src.q3_round7_model import Round7Model
from C.src.class_balanced_batch import ClassBalancedBatch
from C.src.supervised_contrastive import supervised_contrastive

OUT = ROOT/'C/outputs/q3_round7_representation_v1'
CFG = ROOT/'C/configs/q3_round7_representation.json'
base.OUT = OUT; base.CFG = CFG; base.MiniLMFusion = Round7Model
read, write, digest = base.read, base.write, base.digest
MODEL, PREP, SPLIT = base.MODEL, base.PREP, base.SPLIT


class Indexed(Dataset):
    def __init__(self, d): self.dataset = base.RawTextDataset(d)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, i): return (*self.dataset[i], i)


class Run(base.Run):
    def __init__(self):
        super().__init__()
        paths = [Path(__file__), ROOT/'B/src/q3_round7_model.py', ROOT/'B/src/q3_temporal_primary_fusion.py',
                 ROOT/'C/src/class_balanced_batch.py', ROOT/'C/src/supervised_contrastive.py']
        hashes = {str(p): digest(p) for p in paths}
        p = OUT/'round7_sources.json'
        if p.exists(): assert read(p) == hashes, 'Round7 source changed: use a versioned run'
        else: write(p, hashes)
        self.hash += digest(p)
        sources = [x.split('$_$')[0] for x in self.split['source_metadata']['train']['segment_ids']]
        _, self.sources = np.unique(sources, return_inverse=True)
        self.fit_sources = self.sources[self.split['fit_indices']]
        if (OUT/'combination.json').exists():
            spec = read(OUT/'combination.json')['spec']
            if spec is not None: self.cfg['groups']['AB'] = spec

    def fit_one(self, group, s, phase):
        name = f'{phase}_{group}_{s}'
        if self.done(name): return
        t = time.perf_counter(); base.seed(s); torch.cuda.reset_peak_memory_stats()
        spec = self.cfg['groups'][group]; model = Round7Model(MODEL, **spec).to(base.DEVICE)
        data = self.fit if phase == 'inner' else self.train
        sources = self.fit_sources if phase == 'inner' else self.sources
        n = self.cfg['epochs'] if phase == 'inner' else read(OUT/'lock.json')['epochs'][group][str(s)]
        path = OUT/phase/f'{group}_{s}'; path.mkdir(parents=True, exist_ok=True)
        counts = np.bincount(data['cls'], minlength=3)
        weights = torch.tensor(np.sqrt(counts.sum()/(3*counts)), dtype=torch.float32, device=base.DEVICE)
        optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=.01)
        natural = DataLoader(Indexed(data), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(s))
        frozen = {k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad}
        best = None; best_epoch = 0; history = []
        for epoch in range(1, n+1):
            start = time.perf_counter(); model.train(); components = []; valid_total = anchor_total = 0
            sampler = ClassBalancedBatch(data['cls'], sources, s+epoch*100003) if spec['balanced'] else None
            loader = DataLoader(Indexed(data), batch_sampler=sampler) if sampler is not None else natural
            for i, batch in enumerate(loader):
                indices = batch[-1].numpy(); x,y,target = base.inputs(batch[:-1], 'train', random.Random(s+epoch*100003+i))
                optimizer.zero_grad(set_to_none=True); logits, r, _ = model(*x)
                ce = torch.nn.functional.cross_entropy(logits, y, weight=weights)
                regression = torch.nn.functional.smooth_l1_loss(r, target)
                contrast = logits.sum()*0.
                if model.projector is not None:
                    contrast, valid, total = supervised_contrastive(model.projector(model.last_features), y,
                        torch.tensor(sources[indices], device=base.DEVICE), self.cfg['temperature'])
                    valid_total += valid; anchor_total += total
                loss = ce + .8*regression + self.cfg['contrastive_weight']*contrast
                assert torch.isfinite(loss), 'nonfinite training loss'
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert torch.isfinite(grad), 'nonfinite gradient'
                optimizer.step(); components.append([float(loss.detach()), float(ce.detach()), float(regression.detach()), float(contrast.detach()), float(grad)])
            if anchor_total: assert valid_total/anchor_total >= .8, 'Insufficient valid contrastive anchors'
            row = dict(epoch=epoch, loss_components=dict(zip(['total','ce','regression','contrastive','gradient_norm'], np.mean(components, 0).tolist())),
                       valid_anchor_fraction=valid_total/anchor_total if anchor_total else None,
                       sampling=sampler.audit if sampler is not None else {'type':'natural'}, train_seconds=time.perf_counter()-start)
            save = phase == 'final'
            if phase == 'inner':
                p,r = base.predict(model, self.stop); m = base.evaluate(p,r,self.stop['cls'],self.stop['score']); row['metrics'] = m
                if best is None or base.key(m) > best: best = base.key(m); save = True
            if save:
                best_epoch = epoch
                torch.save(dict(encoder_state_dict=model.encoder.state_dict(), fusion_state_dict=model.fusion.state_dict(),
                                projector_state_dict=model.projector.state_dict() if model.projector is not None else None,
                                spec=spec, group=group, seed=s, epoch=epoch), path/'best.pt')
                if phase == 'inner': np.savez_compressed(path/'stop.npz', prob=p, reg=r, cls=self.stop['cls'], score=self.stop['score'], sample_id=self.stop['sample_id'])
            row['epoch_seconds'] = time.perf_counter()-start; history.append(row); write(path/'history.json', history)
            print(name, epoch, 'loss', round(row['loss_components']['total'],4), 'acc', round(row.get('metrics',{}).get('scenarios',[{}])[0].get('accuracy',0),4), flush=True)
        torch.save(dict(encoder_state_dict=model.encoder.state_dict(), fusion_state_dict=model.fusion.state_dict(),
                        projector_state_dict=model.projector.state_dict() if model.projector is not None else None,
                        spec=spec, group=group, seed=s, epoch=n), path/'last.pt')
        assert all(torch.equal(p.detach().cpu(), frozen[k]) for k,p in model.named_parameters() if k in frozen)
        write(path/'result.json', dict(epoch=best_epoch, epochs_run=n, parameters=sum(p.numel() for p in model.parameters()),
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), train_samples=len(data['cls']),
              frozen_unchanged=True, peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        files = [path/x for x in ['best.pt','last.pt','history.json','result.json']]
        if phase == 'inner': files.append(path/'stop.npz')
        self.finish(name, files, t); del model, optimizer, frozen; torch.cuda.empty_cache()

    def stats(self):
        per = {}; epochs = {}
        for g in self.cfg['groups']:
            if not all(self.done(f'inner_{g}_{s}') for s in self.cfg['seeds']): continue
            per[g] = []; epochs[g] = {}
            for s in self.cfg['seeds']:
                path = OUT/'inner'/f'{g}_{s}'; z = np.load(path/'stop.npz')
                assert np.array_equal(z['sample_id'], self.stop['sample_id'])
                per[g].append(base.evaluate(z['prob'],z['reg'],z['cls'],z['score']))
                epochs[g][str(s)] = read(path/'result.json')['epoch']
        return per, {g:base.mean_metrics(ms) for g,ms in per.items()}, epochs

    @staticmethod
    def eligible(g, b, per, means):
        delta = [base.key(a)[0]-base.key(bb)[0] for a,bb in zip(per[g],per[b])]
        return bool(np.mean(delta) >= .005-1e-12 and sum(x>0 for x in delta)>=2 and base.guard(means[g],means[b]))

    def rank(self, names, means):
        return sorted(names, key=lambda g:(-base.key(means[g])[0], -base.key(means[g])[1],
            read(OUT/'inner'/f'{g}_{self.cfg["seeds"][0]}'/'result.json')['parameters'], g))

    def inner(self):
        for g in ['A0','A1','A2','B0','B1','B2']:
            for s in self.cfg['seeds']: self.fit_one(g,s,'inner')
        per,means,_ = self.stats()
        aa = self.rank([g for g in ['A1','A2'] if self.eligible(g,'A0',per,means)], means)
        bb = self.rank([g for g in ['B0','B1','B2'] if self.eligible(g,'A0',per,means)], means)
        combination = dict(spec=None, parents=None)
        if aa and bb:
            spec = dict(self.cfg['groups'][bb[0]])
            spec.update({k:self.cfg['groups'][aa[0]][k] for k in ['balanced','contrastive']})
            combination = dict(spec=spec, parents=[aa[0],bb[0]])
        if (OUT/'combination.json').exists(): assert read(OUT/'combination.json') == combination
        else: write(OUT/'combination.json', combination)
        if combination['spec'] is not None:
            self.cfg['groups']['AB'] = combination['spec']
            for s in self.cfg['seeds']: self.fit_one('AB',s,'inner')

    def select(self):
        if self.done('select'): return
        t=time.perf_counter(); per,means,epochs=self.stats()
        assert all(g in per for g in self.cfg['groups'])
        eligible = ['A0']+[g for g in per if g not in ['A0','AB'] and self.eligible(g,'A0',per,means)]
        if 'AB' in per:
            stronger = self.rank(read(OUT/'combination.json')['parents'], means)[0]
            if self.eligible('AB', stronger, per, means) and self.eligible('AB','A0',per,means): eligible.append('AB')
        chosen=self.rank(eligible,means)[0]
        details={g:dict(delta_accuracy=[base.key(a)[0]-base.key(b)[0] for a,b in zip(per[g],per['A0'])],
                       eligible=g in eligible) for g in per}
        write(OUT/'selection.json',dict(per_seed=per,means=means,details=details,selected=chosen,
              contrastive_supported=self.eligible('A2','A1',per,means),
              dynamic_supported=all(self.eligible('B2',g,per,means) for g in ['B0','B1'])))
        write(OUT/'lock.json',dict(selected=chosen,refit_groups=sorted(set(['A0',chosen])),epochs=epochs,
              deployment_seed=self.cfg['deployment_seed'],selection_hash=digest(OUT/'selection.json'),valid_used_for_architecture=False))
        self.finish('select',[OUT/'selection.json',OUT/'lock.json'],t);print('LOCKED',chosen,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','inner','select','refit','validate'],default='all')
    args=parser.parse_args();run=Run()
    for stage in (['inner','select','refit','validate'] if args.stage=='all' else [args.stage]): getattr(run,stage)()
