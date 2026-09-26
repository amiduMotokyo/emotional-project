"""Round-six model mechanism and saved training-result checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from B.src.q3_large_fusion import build
from C.scripts.run_q2_optimization import read,write,digest
OUT=ROOT/'C/outputs/q3_round6_large_v1'

def preflight():
    torch.set_num_threads(2);torch.manual_seed(60);cfg=read(ROOT/'C/configs/q3_round6_large.json');counts={}
    x=[torch.randn(3,50,n) for n in (768,74,35)];m=[torch.rand(3,50)>.3 for _ in range(3)]
    for spec in cfg['models']:
        model=build(spec).eval();counts[spec['name']]=sum(p.numel() for p in model.parameters())
        for combo in range(8):
            ms=[a & bool(combo&(1<<i)) for i,a in enumerate(m)];a=model(*x,*ms)
            changed=[v.masked_fill(~mask[...,None],9999.) for v,mask in zip(x,ms)];b=model(*changed,*ms)
            assert all(torch.isfinite(v).all() for v in a)
            assert torch.allclose(a[0],b[0],atol=1e-6) and torch.allclose(a[1],b[1],atol=1e-6)
            loss=a[0].square().mean()+a[1].square().mean();model.zero_grad();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert min(counts.values())>500000
    return dict(passed=True,parameters=counts,total_parameters=sum(counts.values()),availability_combinations=8)

def audit():
    from C.scripts.run_q3_round6 import Experiment,evaluate,decode,eligible,stop_score
    run=Experiment();assert run.done('validate')
    for p in (OUT/'state').glob('*.json'):assert run.done(p.stem)
    split=read(OUT/'split.json');g=np.array(split['groups']);assert not set(g[split['train']])&set(g[split['dev']])
    lock=read(OUT/'lock.json');selection=read(OUT/'selection.json');assert lock['selection_hash']==digest(OUT/'selection.json')
    for spec in run.cfg['models']:
        n=spec['name'];inner=read(OUT/'inner'/n/'result.json');final=read(OUT/'final'/n/'result.json');history=read(OUT/'inner'/n/'history.json')
        # The documented improvement tolerance determines checkpoint selection.
        best=-float('inf');epoch=0
        for row in history:
            if row['selection_score']>best+1e-4:best=row['selection_score'];epoch=row['epoch']
        assert inner['best_epoch']==epoch==lock['epochs'][n]==final['epochs_run']==final['best_epoch']
        assert final['train_samples']==3395
    with np.load(OUT/'validation_predictions.npz') as z:
        result=read(OUT/'validation.json');assert result['lock_hash']==digest(OUT/'lock.json')
        for name in ['selected','equal']:
            computed=evaluate(z[name+'_prob'],z[name+'_raw'],z['cls'],z['score']);assert computed==result['results'][name]
    return dict(passed=True,inner_models=7,full_train_models=7,full_train_samples=3395,valid_samples=728,test_evaluated=False,selected=lock['selected'])

if __name__=='__main__':
    result=audit() if '--audit' in sys.argv else preflight();write(OUT/('audit.json' if '--audit' in sys.argv else 'preflight.json'),result);print(result)
