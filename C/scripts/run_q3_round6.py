"""Q3 large models: grouped inner selection, full-train refit, locked valid evaluation."""
import os
import sys
import time
import random
import pickle
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from scipy.optimize import differential_evolution
from B.src.q3_large_fusion import build
from B.src.q3_feature_data import prepare,subset,FeatureDataset,fit_audio_vision_scale,apply_audio_vision_scale
from B.scripts.train_q3_explanation import deletion_effects,explanation_loss
from C.src.missingness import sample_training_masks,corrupt_masks
from C.src.q2_protocol import coherent_score
from C.src.ensemble_search import metrics
from C.scripts.run_q2_optimization import read,write,digest

OUT=ROOT/'C/outputs/q3_round6_large_v1'
CFG=ROOT/'C/configs/q3_round6_large.json'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
CASES=['clean','text','audio','vision']

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def mask_panel(d,case):
    m=[torch.from_numpy(d[k].copy()) for k in ['tmask','amask','vmask']]
    return m if case=='clean' else corrupt_masks(m,case,.3,'middle')

def outputs(model,d):
    result=[];model.eval()
    with torch.inference_mode():
        for case in CASES:
            masks=mask_panel(d,case);ps=[];rs=[]
            for start in range(0,len(d['cls']),128):
                tensors=[torch.from_numpy(d[k][start:start+128]).to(DEVICE) for k in ['text','audio','vision']]+[m[start:start+128].to(DEVICE) for m in masks]
                logits,reg,_=model(*tensors);ps.append(logits.softmax(-1).cpu().numpy());rs.append(reg.cpu().numpy())
            result.append((np.concatenate(ps),np.concatenate(rs)))
    return np.stack([x[0] for x in result]),np.stack([x[1] for x in result])

def evaluate(p,r,y,s):return metrics(p,coherent_score(p.argmax(-1),r),y,s)

def stop_score(m):
    s=m['scenarios'];return .7*s[0]['accuracy']+.3*s[0]['macro_f1']-.05*s[0]['mae']+.05*np.mean([x['macro_f1'] for x in s[1:]])

def decode(p,r,weights,bias):
    prob=np.tensordot(np.asarray(weights),p,axes=(0,0));reg=np.tensordot(np.asarray(weights),r,axes=(0,0))
    logits=np.log(np.maximum(prob,1e-9))+np.asarray(bias);out=np.exp(logits-logits.max(-1,keepdims=True));out/=out.sum(-1,keepdims=True)
    return out,reg

def eligible(m,baseline,cfg):
    a,b=m['scenarios'],baseline['scenarios']
    return bool(a[0]['macro_f1']>=b[0]['macro_f1']-cfg['guard_macro'] and a[0]['per_class_f1'][1]>=b[0]['per_class_f1'][1]-cfg['guard_neutral'] and np.mean([x['macro_f1'] for x in a[1:]])>=np.mean([x['macro_f1'] for x in b[1:]])-cfg['guard_macro'] and np.mean([x['mae'] for x in a])<=np.mean([x['mae'] for x in b])+cfg['guard_mae'])

class Experiment:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True);self.cfg=read(CFG);torch.set_num_threads(2)
        self.source=next((ROOT/'data').glob('*/aligned_50.pkl'))
        paths=[Path(__file__),CFG,ROOT/'B/src/q3_large_fusion.py',ROOT/'B/src/q3_feature_data.py',ROOT/'B/src/data.py',ROOT/'B/src/fusion.py',ROOT/'B/scripts/train_q3_explanation.py',ROOT/'C/src/missingness.py',ROOT/'C/src/q2_protocol.py',ROOT/'C/src/ensemble_search.py',ROOT/'docs/C/experiments/第六轮第三问大容量模型训练方案.md']
        signature=dict(files={str(p):digest(p) for p in paths},aligned_source=digest(self.source))
        path=OUT/'manifest.json'
        if path.exists():assert read(path)['signature']==signature,'Sources changed; use a fresh run directory'
        else:write(path,dict(signature=signature,config=self.cfg,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=DEVICE,gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None)))
        self.run_hash=digest(path)
        with self.source.open('rb') as f:raw=pickle.load(f)
        self.train=prepare(raw['train']);del raw

    def done(self,name):
        path=OUT/'state'/f'{name}.json'
        if not path.exists():return False
        z=read(path);assert z['run_hash']==self.run_hash
        for p,h in z['artifacts'].items():assert digest(p)==h,p
        return True

    def finish(self,name,files,start):
        write(OUT/'state'/f'{name}.json',dict(run_hash=self.run_hash,seconds=time.perf_counter()-start,artifacts={str(p):digest(p) for p in files}))
        print('COMPLETE',name,round(time.perf_counter()-start,2),flush=True)

    def prepare(self):
        if self.done('prepare'):return
        start=time.perf_counter();groups=np.array([x.split('$_$')[0] for x in self.train['sample_id']])
        tr,dev=next(GroupShuffleSplit(n_splits=1,test_size=self.cfg['dev_fraction'],random_state=self.cfg['split_seed']).split(groups,groups=groups))
        assert not set(groups[tr]) & set(groups[dev])
        write(OUT/'split.json',dict(train=tr.tolist(),dev=dev.tolist(),sample_id=self.train['sample_id'].tolist(),groups=groups.tolist()))
        for kind,indices in [('inner',tr),('full',np.arange(len(groups)))]:
            scale=fit_audio_vision_scale(subset(self.train,indices))
            np.savez_compressed(OUT/f'{kind}_normalization.npz',**{n+'_'+suffix:v[j] for n,v in scale.items() for j,suffix in enumerate(['mean','std'])})
        write(OUT/'preparation.json',dict(train=len(tr),dev=len(dev),full=len(groups),train_groups=len(set(groups[tr])),dev_groups=len(set(groups[dev])),test_evaluated=False))
        self.finish('prepare',[OUT/'split.json',OUT/'inner_normalization.npz',OUT/'full_normalization.npz',OUT/'preparation.json'],start)

    def scaled(self,kind):
        d={k:v.copy() for k,v in self.train.items()};z=np.load(OUT/f'{kind}_normalization.npz')
        apply_audio_vision_scale(d,{n:(z[n+'_mean'],z[n+'_std']) for n in ['audio','vision']})
        return d

    def fit_one(self,spec,phase):
        task=phase+'_'+spec['name']
        if self.done(task):return
        start=time.perf_counter();seed_all(spec['seed']);split=read(OUT/'split.json');all_data=self.scaled('inner' if phase=='inner' else 'full')
        train=subset(all_data,split['train']) if phase=='inner' else all_data
        dev=subset(all_data,split['dev']) if phase=='inner' else None
        epochs=self.cfg['epochs'] if phase=='inner' else read(OUT/'inner'/spec['name']/'result.json')['best_epoch']
        model=build(spec).to(DEVICE);optim=torch.optim.AdamW(model.parameters(),lr=self.cfg['lr'],weight_decay=self.cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs,eta_min=self.cfg['lr']*.1)
        counts=np.bincount(train['cls'],minlength=3);weight=torch.tensor(np.sqrt(counts.sum()/(3*counts)),dtype=torch.float32,device=DEVICE)
        ce=torch.nn.CrossEntropyLoss(weight=weight);rng=random.Random(spec['seed'])
        loader=DataLoader(FeatureDataset(train),batch_size=self.cfg['batch_size'],shuffle=True,num_workers=0)
        path=OUT/phase/spec['name'];path.mkdir(parents=True,exist_ok=True)
        history=[];best=-float('inf');best_epoch=0
        for epoch in range(epochs):
            model.train();losses=[]
            for step,batch in enumerate(loader):
                raw=[v.to(DEVICE) for v in batch];inputs=raw[:6];y,s=raw[6:];masks=sample_training_masks([m.cpu() for m in inputs[3:]],spec['missing'],rng)
                corrupted=inputs[:3]+[m.to(DEVICE) for m in masks]
                optim.zero_grad(set_to_none=True);logits,reg,gates,auxp,auxr=model.forward_all(*corrupted)
                loss=ce(logits,y)+self.cfg['reg_weight']*torch.nn.functional.smooth_l1_loss(reg,s)
                present=torch.stack([m.any(1) for m in corrupted[3:]],1)
                auxce=torch.nn.functional.cross_entropy(auxp.reshape(-1,3),y[:,None].expand(-1,3).reshape(-1),weight=weight,reduction='none').reshape(-1,3)
                auxreg=torch.nn.functional.smooth_l1_loss(auxr,s[:,None].expand_as(auxr),reduction='none')
                loss=loss+self.cfg['aux_weight']*((auxce+.8*auxreg)*present).sum()/present.sum().clamp_min(1)
                p=logits.softmax(-1);loss=loss+self.cfg['consistency_weight']*(reg/3-(p[:,2]-p[:,0])).square().mean()
                if step%self.cfg['explanation_every']==0:
                    small=tuple(x[:self.cfg['explanation_batch']] for x in inputs)
                    # cuDNN's eval-mode GRU has no backward reserve; native GRU supports this attribution gradient.
                    with torch.backends.cudnn.flags(enabled=False):
                        eg,ef=deletion_effects(model,small,y[:len(small[0])])
                    loss=loss+self.cfg['explanation_weight']*explanation_loss(eg,ef,small[3:])
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(float(loss.detach()))
            scheduler.step();row=dict(epoch=epoch+1,loss=float(np.mean(losses)))
            if phase=='inner':
                p,r=outputs(model,dev);m=evaluate(p,r,dev['cls'],dev['score']);score=float(stop_score(m));row.update(metrics=m,selection_score=score)
                if score>best+1e-4:
                    best,best_epoch=score,epoch+1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                print(task,epoch+1,'dev acc',round(m['scenarios'][0]['accuracy'],4),'F1',round(m['scenarios'][0]['macro_f1'],4),flush=True)
            else:best_epoch=epoch+1;print(task,epoch+1,'loss',round(row['loss'],4),flush=True)
            history.append(row);write(path/'history.json',history)
            if phase=='inner' and epoch+1-best_epoch>=self.cfg['patience']:break
        if phase=='inner':model.load_state_dict(state)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(dict(state_dict=state,spec=spec,epoch=best_epoch),path/'model.pt')
        files=[path/'model.pt',path/'history.json',path/'result.json']
        if phase=='inner':
            p,r=outputs(model,dev);np.savez_compressed(path/'dev.npz',prob=p,reg=r,sample_id=dev['sample_id']);files.append(path/'dev.npz')
        write(path/'result.json',dict(best_epoch=best_epoch,epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()),train_samples=len(train['cls']),kind=spec['kind']))
        self.finish(task,files,start);del model,optim,all_data,train,dev;torch.cuda.empty_cache()

    def inner(self):
        assert self.done('prepare')
        for spec in self.cfg['models']:self.fit_one(spec,'inner')

    def select(self):
        if self.done('select'):return
        start=time.perf_counter();split=read(OUT/'split.json');dev=subset(self.train,split['dev']);arrays=[]
        for spec in self.cfg['models']:
            assert self.done('inner_'+spec['name']);z=np.load(OUT/'inner'/spec['name']/'dev.npz');assert np.array_equal(z['sample_id'],dev['sample_id']);arrays.append((z['prob'],z['reg']))
        p=np.stack([x[0] for x in arrays]);r=np.stack([x[1] for x in arrays]);n=len(arrays);y,s=dev['cls'],dev['score']
        equal=np.ones(n)/n;base=evaluate(*decode(p,r,equal,[0,0,0]),y,s)
        def unpack(x):
            w=np.exp(x[:n]-max(x[:n]));return w/w.sum(),[float(x[n]),float(x[n+1]),0.]
        def objective(x):
            w,b=unpack(x);m=evaluate(*decode(p,r,w,b),y,s)
            if not eligible(m,base,self.cfg):return 2.
            c=m['scenarios'][0];return -c['accuracy']-.001*c['macro_f1']+.0001*c['mae']
        search=differential_evolution(objective,[(-3,3)]*n+[(-.4,.4)]*2,seed=self.cfg['search_seed'],maxiter=self.cfg['search_iterations'],popsize=self.cfg['search_popsize'],polish=False,workers=1,tol=1e-8,x0=np.zeros(n+2))
        sw,sb=unpack(search.x);candidates={'equal':dict(weights=equal.tolist(),bias=[0.,0.,0.],metrics=base)}
        for i,spec in enumerate(self.cfg['models']):
            w=np.eye(n)[i];candidates[spec['name']]=dict(weights=w.tolist(),bias=[0.,0.,0.],metrics=evaluate(*decode(p,r,w,[0,0,0]),y,s))
        candidates['optimized']=dict(weights=sw.tolist(),bias=sb,metrics=evaluate(*decode(p,r,sw,sb),y,s))
        for v in candidates.values():v['eligible']=eligible(v['metrics'],base,self.cfg)
        ok=[k for k,v in candidates.items() if v['eligible']]
        chosen=sorted(ok,key=lambda k:(-candidates[k]['metrics']['scenarios'][0]['accuracy'],k!='equal',-candidates[k]['metrics']['scenarios'][0]['macro_f1'],candidates[k]['metrics']['scenarios'][0]['mae'],k))[0]
        write(OUT/'selection.json',dict(selected=chosen,candidates=candidates,search_evaluations=search.nfev))
        write(OUT/'lock.json',dict(selected=chosen,weights=candidates[chosen]['weights'],bias=candidates[chosen]['bias'],model_specs=self.cfg['models'],epochs={spec['name']:read(OUT/'inner'/spec['name']/'result.json')['best_epoch'] for spec in self.cfg['models']},selection_hash=digest(OUT/'selection.json'),run_hash=self.run_hash,valid_used_for_selection=False))
        self.finish('select',[OUT/'selection.json',OUT/'lock.json'],start);print('LOCKED',chosen,flush=True)

    def refit(self):
        assert self.done('select')
        for spec in self.cfg['models']:self.fit_one(spec,'final')

    def validate(self):
        assert self.done('select')
        if self.done('validate'):return
        start=time.perf_counter();lock=read(OUT/'lock.json')
        with self.source.open('rb') as f:valid=prepare(pickle.load(f)['valid'])
        z=np.load(OUT/'full_normalization.npz');apply_audio_vision_scale(valid,{n:(z[n+'_mean'],z[n+'_std']) for n in ['audio','vision']})
        arrays=[];files=[];single={}
        for spec in self.cfg['models']:
            name=spec['name'];assert self.done('final_'+name);path=OUT/'valid'/f'{name}.npz'
            if not self.done('valid_'+name):
                t=time.perf_counter();model=build(spec).to(DEVICE);obj=torch.load(OUT/'final'/name/'model.pt',map_location=DEVICE,weights_only=True);model.load_state_dict(obj['state_dict']);p,r=outputs(model,valid)
                path.parent.mkdir(exist_ok=True);np.savez_compressed(path,prob=p,reg=r,sample_id=valid['sample_id']);self.finish('valid_'+name,[path],t);del model
            z=np.load(path);arrays.append((z['prob'],z['reg']));files.append(path);single[name]=evaluate(z['prob'],z['reg'],valid['cls'],valid['score'])
        p=np.stack([x[0] for x in arrays]);r=np.stack([x[1] for x in arrays]);summaries={};arr=dict(cls=valid['cls'],score=valid['score'],sample_id=valid['sample_id'])
        for name,w,b in [('selected',lock['weights'],lock['bias']),('equal',[1/len(arrays)]*len(arrays),[0,0,0])]:
            pp,rr=decode(p,r,w,b);summaries[name]=evaluate(pp,rr,valid['cls'],valid['score']);arr[name+'_prob']=pp;arr[name+'_raw']=rr;arr[name+'_reg']=coherent_score(pp.argmax(-1),rr)
        np.savez_compressed(OUT/'validation_predictions.npz',**arr)
        write(OUT/'validation.json',dict(selected=lock['selected'],lock_hash=digest(OUT/'lock.json'),results=summaries,single_models=single,n=len(valid['cls']),note='Official valid post-lock, historically reused; feature-level missingness. No test evaluation.'))
        self.finish('validate',[OUT/'validation.json',OUT/'validation_predictions.npz',*files],start)
        print('VALID',summaries['selected']['scenarios'][0],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','inner','select','refit','validate'],default='all');args=p.parse_args();run=Experiment()
    for stage in (['prepare','inner','select','refit','validate'] if args.stage=='all' else [args.stage]):getattr(run,stage)()
