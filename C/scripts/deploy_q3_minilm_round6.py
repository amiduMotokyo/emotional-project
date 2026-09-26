"""Export, evaluate batch-one int8 inference, and fit grouped deployment calibration."""
import sys, itertools, shutil, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'));sys.path.insert(0,str(ROOT/'C/outputs/q3_round6_export_env'))
import numpy as np
import torch
import onnxruntime as ort
from sklearn.model_selection import StratifiedGroupKFold
from C.scripts.run_q3_minilm_round6 import OUT,MODEL,PREP,CFG,SPLIT,Run,read,write,digest,evaluate,guard,mean_metrics
from C.scripts.export_q2_minilm_onnx import export
from B.src.data import load_npz,apply_audio_vision_scale
from B.src.fusion import Fusion
from C.src.missingness import corrupt_masks
from C.src.q2_protocol import MODES,RATES,POSITIONS

def bias_prob(p,b):
    z=np.log(np.maximum(p,1e-12))+b;p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)

def int8_panel(encoder,model,d,conditions):
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(encoder),sess_options=opts,providers=['CPUExecutionProvider'])
    cache={};allp=[];allr=[];model.eval();base=[torch.from_numpy(d[k].copy()) for k in ['tmask','amask','vmask']]
    for mode,rate,pos in conditions:
        masks=base if mode is None else corrupt_masks(base,mode,rate,pos)
        key=(rate,pos) if mode and 'text' in mode else ('clean',)
        if key not in cache:
            ids=d['token_ids'].copy().astype(np.int64);ids[d['tmask']&~masks[0].numpy()]=100
            values=[]
            for i in range(len(ids)):
                values.append(session.run(None,dict(input_ids=ids[i:i+1],attention_mask=d['attention'][i:i+1].astype(np.int64),token_type_ids=np.zeros((1,50),dtype=np.int64)))[0][0])
            cache[key]=np.stack(values).astype(np.float32)
        ps=[];rs=[]
        with torch.inference_mode():
            for start in range(0,len(d['cls']),64):
                end=start+64;t=torch.from_numpy(cache[key][start:end]);a=torch.from_numpy(d['audio'][start:end]);v=torch.from_numpy(d['vision'][start:end]);m=[x[start:end] for x in masks]
                p,r,_=model(t,a*m[1][...,None],v*m[2][...,None],*m);ps.append(p.softmax(-1).numpy());rs.append(r.numpy())
        allp.append(np.concatenate(ps));allr.append(np.concatenate(rs))
    return np.stack(allp),np.stack(allr)

def main(context=None, package=None, model_factory=None):
    # Optional context/factory lets later paired experiments reuse this exact evaluator.
    context = context or sys.modules[__name__]
    OUT, CFG = context.OUT, context.CFG
    run=context.Run();cfg=run.cfg;lock=read(OUT/'lock.json');torch.set_num_threads(2)
    def make_model(group):
        if model_factory is not None: return model_factory(cfg['groups'][group])
        kwargs={'dim':cfg['groups'][group]['dim']}
        return Fusion(**kwargs), 'fusion', kwargs
    d=load_npz(PREP/'valid.npz');norm=np.load(OUT/'full_normalization.npz');apply_audio_vision_scale(d,{k:(norm[k+'_mean'],norm[k+'_std']) for k in ['audio','vision']})
    conditions=[(None,0,'middle')]+[(m,.3,'middle') for m in ['text','audio','vision']]
    reports={}
    groups=np.array([x.split('$_$')[0] for x in read(SPLIT)['source_metadata']['valid']['segment_ids']])
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=cfg['calibration_seed']).split(groups,d['cls'],groups))
    assert all(not set(groups[a])&set(groups[b]) for a,b in folds)
    for group in lock['refit_groups']:
        arrays=[]
        for seed in cfg['seeds']:
            name=f'int8_{group}_{seed}';path=OUT/'deploy'/f'{group}_{seed}';path.mkdir(parents=True,exist_ok=True)
            assert run.done(f'fp32_{group}_{seed}')
            if not run.done(name):
                t=time.perf_counter();cp=OUT/'final'/f'{group}_{seed}'/'best.pt';obj=torch.load(cp,map_location='cpu',weights_only=True)
                report=export(MODEL,cp,path);write(path/'export.json',report)
                model,architecture,kwargs=make_model(group);model.load_state_dict(obj['fusion_state_dict']);torch.save(dict(architecture=architecture,model_kwargs=kwargs,state_dict=model.state_dict()),path/'model.pt')
                p,r=int8_panel(path/'text_encoder_finetuned_int8.onnx',model,d,conditions)
                np.savez_compressed(path/'valid.npz',prob=p,reg=r,cls=d['cls'],score=d['score'],sample_id=d['sample_id']);write(path/'valid.json',evaluate(p,r,d['cls'],d['score']))
                run.finish(name,[path/x for x in ['text_encoder_finetuned_int8.onnx','model.pt','export.json','valid.npz','valid.json']],t)
                del obj,model
            z=np.load(path/'valid.npz');assert np.array_equal(z['sample_id'],d['sample_id']);arrays.append((z['prob'],z['reg']))
        task='calibrate_'+group
        if not run.done(task):
            t=time.perf_counter();candidates=[(a,b,0.) for a,b in itertools.product(cfg['bias_levels'],repeat=2)]
            def summary(indices,bias):
                return mean_metrics([evaluate(bias_prob(p[:,indices],bias),r[:,indices],d['cls'][indices],d['score'][indices]) for p,r in arrays])
            def choose(indices):
                baseline=summary(indices,(0,0,0));values=[]
                for b in candidates:
                    result=summary(indices,b)
                    if guard(result,baseline):values.append((b,result['scenarios'][0]['accuracy']))
                return sorted(values,key=lambda x:(-x[1],sum(abs(v) for v in x[0]),x[0]))[0][0]
            oof=[np.zeros_like(p,dtype=np.float64) for p,r in arrays];fold_report=[]
            for fit,held in folds:
                b=choose(fit);fold_report.append(dict(fit=fit.tolist(),held=held.tolist(),bias=list(b)))
                for out,(p,r) in zip(oof,arrays):out[:,held]=bias_prob(p[:,held],b)
            base=summary(np.arange(len(groups)),(0,0,0));oof_per=[evaluate(p,r,d['cls'],d['score']) for p,(_,r) in zip(oof,arrays)];oof_summary=mean_metrics(oof_per)
            use=guard(oof_summary,base) and oof_summary['scenarios'][0]['accuracy']>base['scenarios'][0]['accuracy']+1e-12
            b=choose(np.arange(len(groups))) if use else (0.,0.,0.)
            report=dict(class_bias=list(b),used=bool(use),folds=fold_report,baseline=base,oof=oof_summary,oof_per_seed=oof_per,full_fit=summary(np.arange(len(groups)),b),note='Grouped calibration only; historical validation remains internal. Architecture selected on train-internal stop.')
            write(OUT/'calibration'/f'{group}.json',report)
            np.savez_compressed(OUT/'calibration'/f'{group}_oof.npz',prob=np.stack(oof),sample_id=d['sample_id'])
            run.finish(task,[OUT/'calibration'/f'{group}.json',OUT/'calibration'/f'{group}_oof.npz'],t)
        reports[group]=read(OUT/'calibration'/f'{group}.json')
    # Deployment seed was fixed before any validation predictions.
    group=lock['selected'];seed=cfg['deployment_seed'];src=OUT/'deploy'/f'{group}_{seed}';package=package or ROOT/'C/outputs/submission/q3_round6_minilm_v2'
    package.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src/'text_encoder_finetuned_int8.onnx',package/'text_encoder_int8.onnx');shutil.copy2(src/'model.pt',package/'model.pt');shutil.copy2(OUT/'full_normalization.npz',package/'audio_vision_normalization.npz')
    write(package/'class_bias.json',dict(class_bias=reports[group]['class_bias']))
    write(package/'metadata.json',dict(group=group,seed=seed,epoch=lock['epochs'][group][str(seed)],model_kwargs=cfg['groups'][group],architecture_selection='train internal stop',calibration='official valid grouped five folds',test_used=False))
    tok=package/'tokenizer';tok.mkdir(exist_ok=True)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']:shutil.copy2(MODEL/name,tok/name)
    task='final_grid'
    if not run.done(task):
        t=time.perf_counter();obj=torch.load(package/'model.pt',map_location='cpu',weights_only=True);model,_,_=make_model(group);model.load_state_dict(obj['state_dict']);conditions=[(None,0,'middle')]+list(itertools.product(MODES,RATES,POSITIONS));p,r=int8_panel(package/'text_encoder_int8.onnx',model,d,conditions);pp=bias_prob(p,reports[group]['class_bias'])
        np.savez_compressed(OUT/'final_grid.npz',prob=pp,raw_prob=p,reg=r,cls=d['cls'],score=d['score'],sample_id=d['sample_id']);write(OUT/'final_grid.json',dict(conditions=conditions,metrics=evaluate(pp,r,d['cls'],d['score']),group=group,seed=seed));run.finish(task,[OUT/'final_grid.npz',OUT/'final_grid.json'],t)
    print('DEPLOYMENT_READY',group,seed,read(OUT/'final_grid.json')['metrics']['scenarios'][0],flush=True)

if __name__=='__main__':main()
