"""Round-six v2: paired online learning, locked full-train refit, FP32 validation."""
import sys, os, time, random, argparse, json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'C/outputs/q2_round3_env'))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from torch.utils.data import DataLoader
from B.src.data import load_npz, fit_audio_vision_scale, apply_audio_vision_scale
from B.src.q3_minilm_fusion import MiniLMFusion
from C.scripts.finetune_q2_minilm import RawTextDataset
from C.src.missingness import sample_training_masks, corrupt_masks
from C.src.ensemble_search import metrics
from C.src.q2_protocol import coherent_score

OUT=ROOT/'C/outputs/q3_round6_minilm_v2'
PREP=ROOT/'C/outputs/q2_optimization_local_myenv_v1/prepared'
SPLIT=ROOT/'C/outputs/q2_round2_ensemble_v1/splits.json'
MODEL=Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CFG=ROOT/'C/configs/q3_round6_minilm.json'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def subset(d,indices):return {k:v[indices].copy() for k,v in d.items()}
def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def evaluate(p,r,y,s):return metrics(p,coherent_score(p.argmax(-1),r),y,s)
def key(m):
    c=m['scenarios'][0];return c['accuracy'],c['macro_f1'],-c['mae']
def guard(a,b):
    a,b=a['scenarios'],b['scenarios']
    return bool(a[0]['macro_f1']>=b[0]['macro_f1']-.005-1e-12 and a[0]['per_class_f1'][1]>=b[0]['per_class_f1'][1]-.02-1e-12 and np.mean([x['macro_f1'] for x in a[1:]])>=np.mean([x['macro_f1'] for x in b[1:]])-.01-1e-12 and np.mean([x['mae'] for x in a])<=np.mean([x['mae'] for x in b])+.02+1e-12)
def mean_metrics(values):
    return {'scenarios':[{k:np.mean([m['scenarios'][i][k] for m in values],axis=0).tolist() for k in ['accuracy','macro_f1','mae','pearson','per_class_f1']} for i in range(4)]}
def inputs(batch,mode=None,rng=None):
    ids,att,tm,a,v,am,vm,y,s=batch;m=[tm,am,vm]
    if mode=='train':m=sample_training_masks(m,probability=.3,rng=rng)
    elif mode is not None:m=corrupt_masks(m,mode,.3,'middle')
    ids=ids.clone();ids[tm & ~m[0]]=100
    return [x.to(DEVICE) for x in [ids,att,a*m[1][...,None],v*m[2][...,None],*m]],y.to(DEVICE),s.to(DEVICE)
def predict(model,d):
    model.eval();pp=[];rr=[]
    with torch.inference_mode():
        for mode in [None,'text','audio','vision']:
            ps=[];rs=[]
            for batch in DataLoader(RawTextDataset(d),batch_size=64):
                x,_,_=inputs(batch,mode);p,r,_=model(*x);ps.append(p.softmax(-1).cpu().numpy());rs.append(r.cpu().numpy())
            pp.append(np.concatenate(ps));rr.append(np.concatenate(rs))
    return np.stack(pp),np.stack(rr)

class Run:
    def __init__(self):
        torch.set_num_threads(2);self.cfg=read(CFG);OUT.mkdir(parents=True,exist_ok=True)
        paths=[Path(__file__),CFG,ROOT/'B/src/q3_minilm_fusion.py',ROOT/'B/src/fusion.py',ROOT/'B/src/data.py',ROOT/'C/scripts/finetune_q2_minilm.py',ROOT/'C/src/missingness.py',ROOT/'C/src/q2_protocol.py',ROOT/'C/src/ensemble_search.py',SPLIT,PREP/'train.npz',PREP/'valid.npz',MODEL/'model.safetensors',MODEL/'config.json']
        sig={str(p):digest(p) for p in paths};p=OUT/'manifest.json'
        import transformers
        if p.exists():assert read(p)['source_hashes']==sig,'Sources changed; start an explicitly versioned run'
        else:write(p,dict(source_hashes=sig,config=self.cfg,python=sys.version,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),started=time.strftime('%Y-%m-%d %H:%M:%S')))
        self.hash=digest(p);self.split=read(SPLIT);self.train=load_npz(PREP/'train.npz')
        assert self.train['sample_id'][self.split['fit_indices']].tolist()==self.split['fit_ids']
        assert self.train['sample_id'][self.split['stop_indices']].tolist()==self.split['stop_ids']
        g=np.array(self.split['source_metadata']['train']['segment_ids']);g=np.array([x.split('$_$')[0] for x in g]);assert not set(g[self.split['fit_indices']])&set(g[self.split['stop_indices']])
        self.fit=subset(self.train,self.split['fit_indices']);self.stop=subset(self.train,self.split['stop_indices'])
        for name,d in [('fit',self.fit),('full',self.train)]:
            scale=fit_audio_vision_scale(d);p=OUT/f'{name}_normalization.npz'
            if not p.exists():np.savez_compressed(p,**{k+'_'+n:v[j] for k,v in scale.items() for j,n in enumerate(['mean','std'])})
            else:
                z=np.load(p);assert all(np.array_equal(z[k+'_'+n],v[j]) for k,v in scale.items() for j,n in enumerate(['mean','std']))
            apply_audio_vision_scale(d,scale)
            if name=='fit':apply_audio_vision_scale(self.stop,scale)
    def done(self,name):
        p=OUT/'state'/f'{name}.json'
        if not p.exists():return False
        d=read(p);assert d['run_hash']==self.hash
        for n,h in d['artifacts'].items():assert digest(OUT/n)==h,n
        return True
    def finish(self,name,files,t):
        write(OUT/'state'/f'{name}.json',dict(run_hash=self.hash,seconds=time.perf_counter()-t,artifacts={p.relative_to(OUT).as_posix():digest(p) for p in files}));print('COMPLETE',name,round(time.perf_counter()-t,2),flush=True)
    def fit_one(self,group,s,phase):
        name=f'{phase}_{group}_{s}'
        if self.done(name):return
        t=time.perf_counter();seed(s);spec=self.cfg['groups'][group];model=MiniLMFusion(MODEL,**spec).to(DEVICE)
        count=sum(p.numel() for p in model.parameters());active=sum(p.numel() for p in model.parameters() if p.requires_grad)
        train=self.fit if phase=='inner' else self.train;n=self.cfg['epochs'] if phase=='inner' else read(OUT/'lock.json')['epochs'][group][str(s)]
        path=OUT/phase/f'{group}_{s}';path.mkdir(parents=True,exist_ok=True)
        loader=DataLoader(RawTextDataset(train),batch_size=self.cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(s))
        optim=torch.optim.AdamW(model.optimizer_groups(),weight_decay=.01);counts=np.bincount(train['cls'],minlength=3);w=torch.tensor(np.sqrt(counts.sum()/(3*counts)),dtype=torch.float32,device=DEVICE)
        best=None;best_epoch=0;history=[]
        frozen={k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad}
        for e in range(1,n+1):
            model.train();losses=[]
            for i,batch in enumerate(loader):
                x,y,target=inputs(batch,'train',random.Random(s+e*100003+i));optim.zero_grad(set_to_none=True);logits,r,_=model(*x)
                loss=torch.nn.functional.cross_entropy(logits,y,weight=w)+.8*torch.nn.functional.smooth_l1_loss(r,target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(loss.item())
            row=dict(epoch=e,loss=float(np.mean(losses)))
            save=phase=='final'
            if phase=='inner':
                p,r=predict(model,self.stop);m=evaluate(p,r,self.stop['cls'],self.stop['score']);row['metrics']=m
                if best is None or key(m)>best:best=key(m);save=True
            if save:
                best_epoch=e;torch.save(dict(encoder_state_dict=model.encoder.state_dict(),fusion_state_dict=model.fusion.state_dict(),seed=s,epoch=e,group=group,spec=spec),path/'best.pt')
                if phase=='inner':np.savez_compressed(path/'stop.npz',prob=p,reg=r,cls=self.stop['cls'],score=self.stop['score'],sample_id=self.stop['sample_id'])
            history.append(row);write(path/'history.json',history)
            print(name,e,'loss',round(row['loss'],4),'stop_acc',round(row.get('metrics',{}).get('scenarios',[{}])[0].get('accuracy',0),4),flush=True)
        torch.save(dict(encoder_state_dict=model.encoder.state_dict(),fusion_state_dict=model.fusion.state_dict(),seed=s,epoch=n,group=group,spec=spec),path/'last.pt')
        assert all(torch.equal(p.detach().cpu(),frozen[k]) for k,p in model.named_parameters() if k in frozen)
        write(path/'result.json',dict(epoch=best_epoch,epochs_run=n,parameters=count,trainable_parameters=active,train_samples=len(train['cls']),frozen_unchanged=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        files=[path/x for x in ['best.pt','last.pt','history.json','result.json']]
        if phase=='inner':files.append(path/'stop.npz')
        self.finish(name,files,t);del model,optim,frozen;torch.cuda.empty_cache()
    def inner(self):
        for g in self.cfg['groups']:
            for s in self.cfg['seeds']:self.fit_one(g,s,'inner')
    def select(self):
        if self.done('select'):return
        t=time.perf_counter();per={};epochs={}
        for g in self.cfg['groups']:
            per[g]=[];epochs[g]={}
            for s in self.cfg['seeds']:
                assert self.done(f'inner_{g}_{s}');path=OUT/'inner'/f'{g}_{s}';z=np.load(path/'stop.npz');per[g].append(evaluate(z['prob'],z['reg'],z['cls'],z['score']));epochs[g][str(s)]=read(path/'result.json')['epoch']
        means={g:mean_metrics(v) for g,v in per.items()};eligible=['R0'];details={}
        for g in ['R1','R2','R3']:
            delta=[key(a)[0]-key(b)[0] for a,b in zip(per[g],per['R0'])];ok=guard(means[g],means['R0']) and np.mean(delta)>=.005-1e-12 and sum(d>0 for d in delta)>=2
            details[g]=dict(delta_accuracy=delta,guard=guard(means[g],means['R0']),improvement_eligible=bool(ok))
            if ok:eligible.append(g)
        chosen=sorted(eligible,key=lambda g:(-key(means[g])[0],g!='R0',read(OUT/'inner'/f'{g}_{self.cfg["seeds"][0]}'/'result.json')['trainable_parameters'],g))[0]
        write(OUT/'selection.json',dict(per_seed=per,means=means,details=details,selected=chosen))
        write(OUT/'lock.json',dict(selected=chosen,refit_groups=sorted(set(['R0',chosen])),epochs=epochs,deployment_seed=self.cfg['deployment_seed'],selection_hash=digest(OUT/'selection.json'),valid_used_for_architecture=False))
        self.finish('select',[OUT/'selection.json',OUT/'lock.json'],t);print('LOCKED',chosen,flush=True)
    def refit(self):
        assert self.done('select')
        for g in read(OUT/'lock.json')['refit_groups']:
            for s in self.cfg['seeds']:self.fit_one(g,s,'final')
    def validate(self):
        assert self.done('select');valid=load_npz(PREP/'valid.npz');z=np.load(OUT/'full_normalization.npz');apply_audio_vision_scale(valid,{k:(z[k+'_mean'],z[k+'_std']) for k in ['audio','vision']})
        for g in read(OUT/'lock.json')['refit_groups']:
            for s in self.cfg['seeds']:
                name=f'fp32_{g}_{s}'
                if self.done(name):continue
                assert self.done(f'final_{g}_{s}');t=time.perf_counter();model=MiniLMFusion(MODEL,**self.cfg['groups'][g]).to(DEVICE);obj=torch.load(OUT/'final'/f'{g}_{s}'/'best.pt',map_location=DEVICE,weights_only=True);model.encoder.load_state_dict(obj['encoder_state_dict']);model.fusion.load_state_dict(obj['fusion_state_dict']);p,r=predict(model,valid)
                path=OUT/'valid'/name;path.parent.mkdir(exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),prob=p,reg=r,cls=valid['cls'],score=valid['score'],sample_id=valid['sample_id']);write(path.with_suffix('.json'),evaluate(p,r,valid['cls'],valid['score']));self.finish(name,[path.with_suffix('.npz'),path.with_suffix('.json')],t);del model,obj;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','inner','select','refit','validate'],default='all');args=p.parse_args();run=Run()
    for stage in (['inner','select','refit','validate'] if args.stage=='all' else [args.stage]):getattr(run,stage)()
