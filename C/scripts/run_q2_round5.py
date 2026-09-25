"""Resumable independent-expert diagnosis, grouped OOF gating and locked evaluation."""
import sys
import time
import pickle
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from C.scripts import run_q2_round3 as e
from C.scripts.run_q2_optimization import read,write,digest
from C.src.q2_structural import TextOnly,build
from C.src.q2_experts import AudioVisualExpert,DecisionGate,mix
from B.src.data import fit_audio_vision_scale
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader,Subset
import numpy as np
import torch

OUT=ROOT/'C/outputs/q2_round5_experts_v1'
R4=ROOT/'C/outputs/q2_round4_structure_v1'
e.OUT=OUT
e.CFG=ROOT/'C/configs/q2_round5_experts.json'
ORDER=['wide_control','text_only','audio_visual','equal_mix','learned_gate']

def average(rows):
    return dict(scenarios=[{k:float(np.mean([r['scenarios'][s][k] for r in rows])) for k in ('accuracy','macro_f1','mae','pearson')} for s in range(len(rows[0]['scenarios']))])

def rates_for(data,masks):
    return masks.sum(-1).astype(np.float32)/np.maximum(e.masks_for(data,None).sum(-1),1).astype(np.float32)

def decision(p,r,rates,gate=None):
    with torch.inference_mode():
        args=[torch.as_tensor(v,dtype=torch.float32) for v in (p,r,rates)]
        out=mix(*args) if gate is None else gate.eval()(*args)
    return tuple(v.numpy() for v in out)

class Experiment(e.Run):
    def __init__(self):
        super().__init__()
        self.old_hashes={}
        for p in (R4/'state').glob('*.json'):self.old_hashes.update(read(p)['artifacts'])
        paths=[Path(__file__),ROOT/'C/src/q2_experts.py',ROOT/'C/src/q2_structural.py',ROOT/'C/scripts/check_q2_round5.py',ROOT/'docs/C/experiments/第五轮独立专家与决策融合实验方案.md']
        for family in ['text_only','wide_control']:
            for seed in self.cfg['seeds']:
                for name in ['best.pt','stop.npz','result.json']:
                    p=R4/'models'/f'{family}_{seed}'/name;self.checked_old(p);paths.append(p)
                p=R4/'valid_predictions'/f'{family}_{seed}.npz';self.checked_old(p);paths.append(p)
        signature=dict(base_manifest=digest(OUT/'manifest.json'),code_and_inputs={str(p):digest(p) for p in paths})
        path=OUT/'implementation.json'
        if path.exists():assert read(path)==signature,'Round-five sources changed'
        else:write(path,signature)
        self.run_hash=digest(path)

    def checked_old(self,path):
        assert digest(path)==self.old_hashes[str(path)],str(path)
        return path

    def old_model(self,family,seed):
        obj=torch.load(self.checked_old(R4/'models'/f'{family}_{seed}'/'best.pt'),map_location='cpu',weights_only=False)
        model=build(family);model.load_state_dict(obj['state_dict'])
        return model.to(e.DEVICE).eval()

    def av_model(self,seed):
        assert self.done(f'audio_visual_{seed}')
        obj=torch.load(OUT/'models'/f'audio_visual_{seed}'/'best.pt',map_location='cpu',weights_only=False)
        m=AudioVisualExpert();m.load_state_dict(obj['state_dict']);return m.to(e.DEVICE).eval()

    def train(self):
        assert self.done('prepare')
        original=e.Fusion
        try:
            e.Fusion=AudioVisualExpert
            for seed in self.cfg['seeds']:self.train_one(f'audio_visual_{seed}',dict(self.cfg['recipe'],architecture='audio_visual'),seed)
        finally:e.Fusion=original

    def diagnose(self):
        if self.done('diagnose'):return
        start=time.perf_counter();rows=[]
        for seed in self.cfg['seeds']:
            assert self.done(f'audio_visual_{seed}')
            t=e.load_npz(self.checked_old(R4/'models'/f'text_only_{seed}'/'stop.npz'))
            a=e.load_npz(OUT/'models'/f'audio_visual_{seed}'/'stop.npz')
            assert np.array_equal(t['sample_id'],a['sample_id'])
            tc=t['prob'].argmax(-1)==self.stop['cls'];ac=a['prob'].argmax(-1)==self.stop['cls']
            rows.append(dict(seed=seed,scenarios=[dict(text_accuracy=float(x.mean()),av_accuracy=float(y.mean()),
                rescued=float((~x & y).mean()),harmed=float((x & ~y).mean()),both_wrong=float((~x & ~y).mean()),oracle=float((x|y).mean()),
                disagreement=float((t['prob'][i].argmax(-1)!=a['prob'][i].argmax(-1)).mean())) for i,(x,y) in enumerate(zip(tc,ac))]))
        rescue=[r['scenarios'][0]['rescued'] for r in rows]
        go=np.mean(rescue)>=self.cfg['rescue_mean_min'] and sum(x>=self.cfg['rescue_seed_min'] for x in rescue)>=self.cfg['rescue_seed_count']
        write(OUT/'diagnosis.json',dict(rows=rows,mean_clean_rescue=float(np.mean(rescue)),route='fusion' if go else 'representation',note='Stop used for epoch selection; oracle is not deployable performance.'))
        self.finish('diagnose',[OUT/'diagnosis.json'],start)
        print('ROUTE',read(OUT/'diagnosis.json')['route'],'rescue',np.mean(rescue),flush=True)

    def folds(self):
        path=OUT/'folds.json'
        if path.exists():return np.array(read(path)['fold'])
        groups=np.array(self.split['source_metadata']['train']['group_ids'])[self.split['fit_indices']]
        fold=np.full(len(groups),-1,dtype=int)
        for k,(train,held) in enumerate(StratifiedGroupKFold(self.cfg['folds'],shuffle=True,random_state=self.cfg['fold_seed']).split(np.zeros(len(groups)),self.fit['cls'],groups)):
            assert not set(groups[train]) & set(groups[held]);fold[held]=k
        assert (fold>=0).all()
        write(path,dict(fold=fold.tolist(),groups=groups.tolist(),sample_id=self.fit['sample_id'].tolist()))
        return fold

    def fold_data(self,fold,k):
        raw=e.load_npz(e.OLD/'prepared/train.npz')
        fit=e.subset(raw,self.split['fit_indices']);stop=e.subset(raw,self.split['stop_indices'])
        scale=fit_audio_vision_scale(e.subset(fit,np.flatnonzero(fold!=k)))
        for data in (fit,stop):e.apply_audio_vision_scale(data,scale)
        return fit,stop,scale

    def train_fold(self,k,seed,family,fold):
        name=f'fold{k}_{family}_{seed}'
        if self.done(name):return
        start=time.perf_counter();e.seed_all(seed)
        data,stop,scale=self.fold_data(fold,k);indices=np.flatnonzero(fold!=k);held=np.flatnonzero(fold==k)
        ds=e.ViewDataset(data,e.R2/'fit_views');loader=DataLoader(Subset(ds,indices.tolist()),batch_size=self.cfg['batch_size'],shuffle=True)
        model=(TextOnly() if family=='text_only' else AudioVisualExpert()).to(e.DEVICE)
        optim=torch.optim.AdamW(model.parameters(),lr=self.cfg['fusion_lr'],weight_decay=self.cfg['weight_decay'])
        counts=np.bincount(data['cls'][indices],minlength=3);weights=torch.tensor(np.sqrt(counts.sum()/(3*counts)),device=e.DEVICE,dtype=torch.float32)
        ce=torch.nn.CrossEntropyLoss(weight=weights);best=None;history=[]
        path=OUT/'fold_models'/name;path.mkdir(parents=True,exist_ok=True)
        for epoch in range(self.cfg['epochs']):
            ds.epoch=epoch;model.train();losses=[]
            for batch in loader:
                optim.zero_grad(set_to_none=True);logits,reg,y,s=e.forward(model,batch)
                loss=ce(logits,y)+.8*torch.nn.functional.smooth_l1_loss(reg,s)
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(float(loss.detach()))
            p,r=e.predict(model,stop,e.masks_for(stop,None));result=e.metrics(p[None],r[None],stop['cls'],stop['score'])
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),clean=result['scenarios'][0]))
            if best is None or e.key(result)>e.key(best):
                best=result;best_epoch=epoch+1;state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        model.load_state_dict(state)
        torch.save(dict(state_dict=state,family=family,seed=seed,fold=k,epoch=best_epoch,train_indices=indices,held_indices=held),path/'best.pt')
        write(path/'history.json',history)
        np.savez_compressed(path/'normalization.npz',**{n+'_'+kind:val[j] for n,val in scale.items() for j,kind in enumerate(['mean','std'])})
        hd=e.subset(data,held);ps=[];rs=[];rates=[]
        masks_cache=np.load(e.R2/'fit_views/train_masks.npy',mmap_mode='r');text_cache=np.load(e.R2/'fit_views/train_text.npy',mmap_mode='r')
        for panel in range(11):
            masks=e.masks_for(hd,None);text=hd['text']
            if 1<=panel<=8:masks=np.array(masks_cache[panel-1,held]);text=np.asarray(text_cache[panel-1,held],dtype=np.float32)
            if panel==9:masks[:,0]=False
            if panel==10:masks[:,1:]=False
            # Fully unavailable experts are masked by the decision layer; do not encode restored text.
            if (panel==9 and family=='text_only') or (panel==10 and family=='audio_visual'):
                p=np.full((len(held),3),1/3,dtype=np.float32);r=np.zeros(len(held),dtype=np.float32)
            else:p,r=e.predict(model,hd,masks,text)
            ps.append(p);rs.append(r);rates.append(rates_for(hd,masks))
        np.savez_compressed(path/'held.npz',prob=np.stack(ps),reg=np.stack(rs),rates=np.stack(rates),indices=held,sample_id=hd['sample_id'])
        write(path/'result.json',dict(epoch=best_epoch,stop=best,train=len(indices),held=len(held),parameters=sum(p.numel() for p in model.parameters())))
        self.finish(name,list(path.glob('*')),start)
        del model,optim;torch.cuda.empty_cache()

    def oof(self):
        assert self.done('diagnose') and read(OUT/'diagnosis.json')['route']=='fusion'
        if self.done('oof'):return
        start=time.perf_counter();fold=self.folds();files=[OUT/'folds.json']
        for seed in self.cfg['seeds']:
            for k in range(self.cfg['folds']):
                for family in ['text_only','audio_visual']:self.train_fold(k,seed,family,fold)
            p=np.zeros((11,len(fold),2,3),dtype=np.float32);r=np.zeros((11,len(fold),2),dtype=np.float32);rates=np.zeros((11,len(fold),3),dtype=np.float32)
            for k in range(self.cfg['folds']):
                for j,family in enumerate(['text_only','audio_visual']):
                    z=e.load_npz(OUT/'fold_models'/f'fold{k}_{family}_{seed}'/'held.npz');idx=z['indices']
                    assert np.array_equal(z['sample_id'],self.fit['sample_id'][idx]);p[:,idx,j]=z['prob'];r[:,idx,j]=z['reg'];rates[:,idx]=z['rates']
            path=OUT/'oof'/f'{seed}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,prob=p,reg=r,rates=rates,fold=fold,sample_id=self.fit['sample_id']);files.append(path)
        self.finish('oof',files,start)

    def gate(self):
        assert self.done('oof')
        for seed in self.cfg['seeds']:
            name=f'gate_{seed}'
            if self.done(name):continue
            start=time.perf_counter();e.seed_all(seed);z=e.load_npz(OUT/'oof'/f'{seed}.npz')
            p,r,rates=[torch.tensor(z[k],device=e.DEVICE) for k in ['prob','reg','rates']]
            y=torch.tensor(self.fit['cls'],device=e.DEVICE);score=torch.tensor(self.fit['score'],device=e.DEVICE)
            counts=np.bincount(self.fit['cls'],minlength=3);w=torch.tensor(np.sqrt(counts.sum()/(3*counts)),device=e.DEVICE,dtype=torch.float32)[y]
            panel_w=torch.tensor(self.cfg['gate_panel_weights'],device=e.DEVICE)
            gate=DecisionGate().to(e.DEVICE);optim=torch.optim.AdamW(gate.parameters(),lr=self.cfg['gate_lr'],weight_decay=self.cfg['gate_weight_decay']);losses=[]
            for step in range(self.cfg['gate_steps']):
                optim.zero_grad(set_to_none=True);pred,reg,_=gate(p,r,rates)
                nll=-pred.clamp_min(1e-8).log().gather(-1,y[None,:,None].expand(11,-1,1)).squeeze(-1)
                loss=((((nll*w).sum(1)/w.sum())+.8*torch.nn.functional.smooth_l1_loss(reg,score[None].expand_as(reg),reduction='none').mean(1))*panel_w).sum()
                assert torch.isfinite(loss);loss.backward();optim.step();losses.append(float(loss.detach()))
            path=OUT/'gates'/str(seed);path.mkdir(parents=True,exist_ok=True)
            torch.save(gate.cpu().state_dict(),path/'gate.pt');write(path/'loss.json',losses)
            self.finish(name,[path/'gate.pt',path/'loss.json'],start)

    def load_gate(self,seed):
        assert self.done(f'gate_{seed}');gate=DecisionGate();gate.load_state_dict(torch.load(OUT/'gates'/str(seed)/'gate.pt',weights_only=True));return gate.eval()

    def select(self):
        if self.done('select'):return
        start=time.perf_counter();records={n:[] for n in ORDER};files=[]
        rates=np.stack([rates_for(self.valid,m) for m,t in self.valid_cases])
        for seed in self.cfg['seeds']:
            task=f'valid_{seed}';path=OUT/'valid_predictions'/f'{seed}.npz'
            if not self.done(task):
                t0=time.perf_counter();t=e.load_npz(self.checked_old(R4/'valid_predictions'/f'text_only_{seed}.npz'))
                b=e.load_npz(self.checked_old(R4/'valid_predictions'/f'wide_control_{seed}.npz'))
                av=self.av_model(seed);ap,ar=e.panel(av,self.valid,self.valid_cases);del av
                p=np.stack([t['prob'],ap],-2);r=np.stack([t['reg'],ar],-1)
                ep,er,_=decision(p,r,rates);gp,gr,gw=decision(p,r,rates,self.load_gate(seed))
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,wide_control_prob=b['prob'],wide_control_reg=b['reg'],text_only_prob=t['prob'],text_only_reg=t['reg'],audio_visual_prob=ap,audio_visual_reg=ar,equal_mix_prob=ep,equal_mix_reg=er,learned_gate_prob=gp,learned_gate_reg=gr,gate_weights=gw,sample_id=self.valid['sample_id'])
                self.finish(task,[path],t0)
            z=e.load_npz(path);files.append(path)
            for n in ORDER:records[n].append(e.metrics(z[n+'_prob'],z[n+'_reg'],self.valid['cls'],self.valid['score']))
        means={n:average(rows) for n,rows in records.items()}
        eligible=[n for n in ORDER if e.guard(means[n],means['wide_control'],self.cfg)]
        selected=sorted(eligible,key=lambda n:(-e.key(means[n])[0],ORDER.index(n)))[0]
        write(OUT/'selection.json',dict(selected=selected,candidates={n:dict(mean=means[n],seeds=records[n],eligible=n in eligible) for n in ORDER}))
        write(OUT/'lock.json',dict(selected=selected,seeds=self.cfg['seeds'],run_hash=self.run_hash,selection_hash=digest(OUT/'selection.json'),note='Independent per-seed pipelines, not cross-seed ensemble; reused test only after lock.'))
        self.finish('select',[OUT/'selection.json',OUT/'lock.json',*files],start)
        print('LOCKED',selected,e.key(means[selected])[0],flush=True)

    def evaluate(self):
        assert self.done('select')
        if self.done('evaluate'):return
        start=time.perf_counter();locked=read(OUT/'lock.json');selected=locked['selected']
        names=list(dict.fromkeys(['wide_control',selected]));models={};gates={}
        if selected!='wide_control':
            for seed in self.cfg['seeds']:
                models[seed]=(self.old_model('text_only',seed),self.av_model(seed));gates[seed]=self.load_gate(seed)
            aligned=next((ROOT/'data').glob('*/aligned_50.pkl'))
            assert digest(aligned)==read(e.OLD/'prepared.json')['signature']['source']['aligned_pickle']
            with aligned.open('rb') as f:raw=pickle.load(f)['test']
            session=e.session_for(e.OLD/'prepared/text_encoder_int8.onnx')
            text=e.encode_text(session,raw['text_bert'],'cpu',batch_size=1)
            data=e.assemble_sample(raw['text_bert'],text,raw['audio'],raw['vision'],raw['classification_labels'],raw['regression_labels'])
            e.apply_audio_vision_scale(data,self.scale)
        reports={};files=[];texts={}
        for scenario in [None]+[(m,r,p) for m in e.MODES for r in e.RATES for p in e.POSITIONS]:
            name='clean' if scenario is None else e.case_name(*scenario)[:-4];path=OUT/'test'/f'{name}.npz';stat=path.with_suffix('.json')
            if not self.done('test_'+name):
                t0=time.perf_counter();old=e.load_npz(self.checked_old(R4/'test'/f'{name}.npz'));arrays={k:old[k] for k in ['cls','score','sample_id']};rows={n:[] for n in names}
                if selected!='wide_control':
                    assert np.array_equal(data['cls'],old['cls']) and np.allclose(data['score'],old['score'])
                    masks=e.masks_for(data,scenario);key=masks[:,0].tobytes()
                    if key not in texts:texts[key]=data['text'] if np.array_equal(masks[:,0],data['tmask']) else e._encode_with_masks(session,data,masks)
                    rates=rates_for(data,masks)
                for seed in self.cfg['seeds']:
                    bp,br=old[f'wide_control_{seed}_prob'],old[f'wide_control_{seed}_reg']
                    pairs={'wide_control':(bp,br)}
                    if selected!='wide_control':
                        tp,tr=e.predict(models[seed][0],data,masks,texts[key]);ap,ar=e.predict(models[seed][1],data,masks,texts[key])
                        if selected=='text_only':sp,sr=tp,tr
                        elif selected=='audio_visual':sp,sr=ap,ar
                        else:sp,sr,weights=decision(np.stack([tp,ap],-2),np.stack([tr,ar],-1),rates,gates[seed] if selected=='learned_gate' else None)
                        pairs[selected]=(sp,sr)
                    for n,(p,r) in pairs.items():
                        arrays[f'{n}_{seed}_prob']=p;arrays[f'{n}_{seed}_reg']=r;rows[n].append(e.metrics(p[None],r[None],old['cls'],old['score']))
                write(stat,{n:dict(mean=average(v)['scenarios'][0],seeds=[x['scenarios'][0] for x in v]) for n,v in rows.items()})
                np.savez_compressed(path,**arrays);self.finish('test_'+name,[path,stat],t0)
            reports[name]=read(stat);files.extend([path,stat])
        write(OUT/'test_summary.json',dict(selected=selected,lock_hash=digest(OUT/'lock.json'),scenarios=reports,averages={n:{k:float(np.mean([v[n]['mean'][k] for v in reports.values()])) for k in ['accuracy','macro_f1','mae','pearson']} for n in names}))
        self.finish('evaluate',[OUT/'test_summary.json',*files],start)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','prepare','train','diagnose','oof','gate','select','evaluate']);args=parser.parse_args()
    run=Experiment()
    for stage in (['prepare','train','diagnose','oof','gate','select','evaluate'] if args.stage=='all' else [args.stage]):
        if stage in ['oof','gate','select','evaluate'] and read(OUT/'diagnosis.json')['route']!='fusion':
            print('Representation branch required; fusion skipped.',flush=True);break
        getattr(run,stage)()
