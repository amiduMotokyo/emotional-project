"""Fixed-epoch paired three-layer ablations, train-only fit/dev, no test reads."""
import sys,os,time,random,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from torch.utils.data import DataLoader
from C.scripts import run_q3_minilm_round6 as base
from C.src.q2_ablation_model import AblationModel
from C.src.missingness import _corrupt_row,MODALITY_SETS,corrupt_masks
from B.src.data import load_npz,fit_audio_vision_scale,apply_audio_vision_scale
from C.scripts.finetune_q2_minilm import RawTextDataset

OUT=ROOT/'C/outputs/q2_three_layer_ablation_v2'
CFG=ROOT/'C/configs/q2_three_layer_ablation.json'

def make_config():
    groups={'full':{}}
    for p in [0,.1,.5]:groups[f'prob_{p:g}']={'probability':p}
    for r in [.1,.3,.5]:groups[f'length_{r:g}']={'length':r}
    groups.update(unweighted={'weighted':False},ce_only={'objective':'ce'},reg_only={'objective':'reg'},frozen={'scope':'frozen'})
    for k in ['no_mask','mean_pool','mean_fusion','interaction','experts']:groups[k]={'kind':k}
    for name,mods in [('T',[0]),('A',[1]),('V',[2]),('TA',[0,1]),('TV',[0,2]),('AV',[1,2])]:groups['input_'+name]={'modalities':mods}
    return dict(seeds=[20267001,20267002,20267003],epochs=10,batch_size=32,checkpoint='fixed_final_epoch_no_selection',precision='FP32',groups=groups)

def inputs(batch,spec,rng=None,case=None):
    ids,att,tm,a,v,am,vm,y,target=batch;original=[tm,am,vm]
    masks=[m.clone() for m in original]
    if rng is not None:
        for row in range(len(ids)):
            selected=rng.choice(MODALITY_SETS);rate=spec.get('length',rng.choice([.1,.2,.3,.5]));apply=rng.random()<spec.get('probability',.3)
            temp=[m.clone() for m in original]
            _corrupt_row(temp,original,row,selected,rate,'random',rng)
            if apply:
                for j in range(3):masks[j][row]=temp[j][row]
    elif case is not None:masks=corrupt_masks(original,case,.3,'middle')
    ids=ids.clone();ids[tm & ~masks[0]]=100
    a=a*masks[1][...,None];v=v*masks[2][...,None]
    if spec.get('kind')=='no_mask':
        # Keep base support/padding semantics, hide only newly introduced missingness.
        masks=[m.clone() for m in original]
    for j in set(range(3))-set(spec.get('modalities',[0,1,2])):
        masks[j].zero_()
        if j==0:ids.fill_(100)
        elif j==1:a.zero_()
        else:v.zero_()
    return [x.to(base.DEVICE) for x in [ids,att,a,v,*masks]],y.to(base.DEVICE),target.to(base.DEVICE)

def predict(model,data,spec):
    model.eval();pp=[];rr=[]
    with torch.inference_mode():
        for mode in [None,'text','audio','vision']:
            ps=[];rs=[]
            for batch in DataLoader(RawTextDataset(data),batch_size=64):
                x,_,_=inputs(batch,spec,case=mode);p,r,_=model(*x);ps.append(p.softmax(-1).cpu().numpy());rs.append(r.cpu().numpy())
            pp.append(np.concatenate(ps));rr.append(np.concatenate(rs))
    return np.stack(pp),np.stack(rr)

def run(smoke=False):
    assert torch.cuda.is_available(),'GPU required for scheduled budget'
    torch.set_num_threads(2);cfg=base.read(CFG);OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'C/src/q2_ablation_model.py',CFG,base.SPLIT,base.PREP/'train.npz',base.MODEL/'model.safetensors',ROOT/'B/src/fusion.py',ROOT/'B/src/q3_minilm_fusion.py',ROOT/'C/src/missingness.py']
    sig={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):base.digest(p) for p in sources}
    if (OUT/'manifest.json').exists():assert base.read(OUT/'manifest.json')['sources']==sig,'Source changed; use new version'
    else:base.write(OUT/'manifest.json',dict(sources=sig,config=cfg,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,started=time.strftime('%Y-%m-%d %H:%M:%S'),valid_read=False,test_read=False))
    split=base.read(base.SPLIT);train=load_npz(base.PREP/'train.npz');fit=base.subset(train,split['fit_indices']);dev=base.subset(train,split['stop_indices'])
    groups=[x.split('$_$')[0] for x in split['source_metadata']['train']['segment_ids']]
    assert not {groups[i] for i in split['fit_indices']}&{groups[i] for i in split['stop_indices']}
    assert len(fit['cls'])==3004 and len(dev['cls'])==391
    scale=fit_audio_vision_scale(fit);apply_audio_vision_scale(fit,scale);apply_audio_vision_scale(dev,scale)
    np.savez_compressed(OUT/'fit_normalization.npz',**{k+'_'+n:v[j] for k,v in scale.items() for j,n in enumerate(['mean','std'])})
    w=torch.tensor(np.sqrt(len(fit['cls'])/(3*np.bincount(fit['cls'],minlength=3))),dtype=torch.float32,device=base.DEVICE)
    for group,spec in cfg['groups'].items():
        for seed in (cfg['seeds'][:1] if smoke else cfg['seeds']):
            path=OUT/('smoke' if smoke else 'runs')/f'{group}_{seed}';path.mkdir(parents=True,exist_ok=True)
            if (path/'result.json').exists():continue
            start=time.perf_counter();base.seed(seed);model=AblationModel(base.MODEL,spec).to(base.DEVICE)
            optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=.01)
            loader=DataLoader(RawTextDataset(fit),batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            history=[];objective=spec.get('objective','joint')
            for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
                model.train();losses=[]
                for i,batch in enumerate(loader):
                    # Isolate stochastic masks from model dropout and modality-specific RNG consumption.
                    torch.manual_seed(seed+epoch*100003+i)
                    x,y,target=inputs(batch,spec,random.Random(seed+epoch*100003+i));optimizer.zero_grad(set_to_none=True)
                    logits,r,_=model(*x)
                    ce=torch.nn.functional.cross_entropy(logits,y,weight=w if spec.get('weighted',True) else None)
                    reg=torch.nn.functional.smooth_l1_loss(r,target)
                    loss=ce if objective=='ce' else reg if objective=='reg' else ce+.8*reg
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
                    if smoke:break
                history.append(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-start));base.write(path/'history.json',history)
                print(group,seed,'epoch',epoch,'loss',round(history[-1]['loss'],4),'seconds',round(time.perf_counter()-start,1),flush=True)
            p,r=predict(model,base.subset(dev,np.arange(8)) if smoke else dev,spec)
            labels=dev['cls'][:8] if smoke else dev['cls'];scores=dev['score'][:8] if smoke else dev['score']
            result=base.evaluate(p,r,labels,scores)
            raw=base.metrics(p,r,labels,scores)
            if objective=='ce':
                for m in result['scenarios']:
                    for k in ['mae','pearson']:m[k]=None
            if objective=='reg':
                result=raw
                for m in result['scenarios']:
                    for k in ['accuracy','macro_f1','per_class_f1']:m[k]=None
            if objective!='joint':result['selection_score']=None
            result.update(group=group,seed=seed,spec=spec,epoch=history[-1]['epoch'],seconds=time.perf_counter()-start,raw_regression_mae=[m['mae'] for m in raw['scenarios']] if objective!='ce' else None,raw_regression_pearson=[m['pearson'] for m in raw['scenarios']] if objective!='ce' else None)
            assert np.isfinite(p).all() and np.isfinite(r).all()
            if not smoke:
                np.savez_compressed(path/'dev_predictions.npz',prob=p,reg=r,cls=labels,score=scores,sample_id=dev['sample_id'])
                active={n for n,t in model.named_parameters() if t.requires_grad}
                state={n:t.cpu() for n,t in model.state_dict().items() if n in active or n.startswith('fusion.')}
                torch.save(dict(state_dict=state,spec=spec,seed=seed,epoch=cfg['epochs'],encoder_source=str(base.MODEL)),path/'checkpoint.pt')
            base.write(path/'result.json',result);del model,optimizer;torch.cuda.empty_cache()
            print('COMPLETE',group,seed,round(result['seconds'],1),flush=True)
    results=[base.read(p) for p in (OUT/('smoke' if smoke else 'runs')).glob('*/result.json')]
    base.write(OUT/('smoke_summary.json' if smoke else 'summary.json'),dict(complete=len(results)==len(cfg['groups'])*(1 if smoke else len(cfg['seeds'])),runs=results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--init',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.init:base.write(CFG,make_config());print('CONFIG_CREATED')
    else:run(args.smoke)
