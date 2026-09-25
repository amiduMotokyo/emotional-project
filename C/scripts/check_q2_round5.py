"""Mechanism tests and saved-result audit for round-five expert fusion."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from C.src.q2_experts import AudioVisualExpert,DecisionGate,mix
from C.scripts.run_q2_optimization import read,write,digest
OUT=ROOT/'C/outputs/q2_round5_experts_v1'

def preflight():
    torch.manual_seed(15);torch.set_num_threads(1)
    model=AudioVisualExpert().eval();inputs=[torch.randn(3,50,n) for n in [384,74,35]]
    masks=[torch.rand(3,50)>.3 for _ in range(3)]
    for combination in range(8):
        ms=[m & bool(combination&(1<<j)) for j,m in enumerate(masks)]
        z=model(*inputs,*ms)
        changed=[x.masked_fill(~m[:,:,None],9999) for x,m in zip(inputs,ms)]
        other=model(*changed,*ms)
        assert torch.allclose(z[0],other[0],atol=1e-6)
        assert torch.equal(z[0],model(inputs[0]*100,inputs[1],inputs[2],~ms[0],ms[1],ms[2])[0])
        loss=z[0].square().mean()+z[1].square().mean();model.zero_grad();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    p=torch.rand(8,2,3).softmax(-1);r=torch.randn(8,2)
    rates=torch.tensor([[bool(i&(1<<j)) for j in range(3)] for i in range(8)],dtype=torch.float32)
    gate=DecisionGate()
    a=gate(p,r,rates);b=mix(p,r,rates)
    assert all(torch.allclose(x,y) for x,y in zip(a,b))
    assert torch.allclose(a[0].sum(-1),torch.ones(8))
    assert torch.allclose(a[0][0],torch.ones(3)/3) and a[1][0]==0
    assert a[2][1].tolist()==[1.,0.] and a[2][2].tolist()==[0.,1.]
    altered=p.clone();altered[2,0]=torch.tensor([1.,0.,0.]);assert torch.equal(gate(altered,r,rates)[0][2],a[0][2])
    (a[0].square().sum()+a[1].square().sum()).backward()
    assert all(torch.isfinite(x.grad).all() for x in gate.parameters())
    return dict(passed=True,av_parameters=sum(p.numel() for p in model.parameters()),gate_parameters=sum(p.numel() for p in gate.parameters()),availability_combinations=8)

def audit():
    from C.scripts.run_q2_round5 import Experiment,e,average,decision,ORDER
    run=Experiment()
    for p in (OUT/'state').glob('*.json'):assert run.done(p.stem)
    fold=run.folds();saved=read(OUT/'folds.json');groups=np.array(saved['groups'])
    from sklearn.model_selection import StratifiedGroupKFold
    expected=np.full(len(fold),-1)
    for k,(tr,held) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=run.cfg['fold_seed']).split(np.zeros(len(fold)),run.fit['cls'],groups)):expected[held]=k
    assert np.array_equal(fold,expected)
    model_count=0
    for seed in run.cfg['seeds']:
        oof=e.load_npz(OUT/'oof'/f'{seed}.npz');counts=np.zeros(len(fold),dtype=int)
        assert np.array_equal(oof['sample_id'],run.fit['sample_id']) and np.array_equal(oof['fold'],fold)
        for k in range(3):
            assert not set(groups[fold==k]) & set(groups[fold!=k]);counts[fold==k]+=1
            _,_,scale=run.fold_data(fold,k)
            for j,family in enumerate(['text_only','audio_visual']):
                path=OUT/'fold_models'/f'fold{k}_{family}_{seed}';z=e.load_npz(path/'held.npz');norm=e.load_npz(path/'normalization.npz')
                obj=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
                assert np.array_equal(obj['train_indices'],np.flatnonzero(fold!=k)) and np.array_equal(obj['held_indices'],np.flatnonzero(fold==k))
                for n,v in scale.items():
                    for q,kind in enumerate(['mean','std']):assert np.array_equal(norm[n+'_'+kind],v[q])
                assert np.array_equal(oof['prob'][:,fold==k,j],z['prob']) and np.array_equal(oof['reg'][:,fold==k,j],z['reg'])
                history=read(path/'history.json');assert len(history)==18
                best=max(history,key=lambda v:(v['clean']['accuracy'],v['clean']['macro_f1'],-v['clean']['mae']))
                assert best['epoch']==obj['epoch'];model_count+=1
        assert (counts==1).all()
    sel=read(OUT/'selection.json');means={}
    for n in ORDER:
        rows=[]
        for seed in run.cfg['seeds']:
            z=e.load_npz(OUT/'valid_predictions'/f'{seed}.npz');assert np.array_equal(z['sample_id'],run.valid['sample_id'])
            rows.append(e.metrics(z[n+'_prob'],z[n+'_reg'],run.valid['cls'],run.valid['score']))
        assert rows==sel['candidates'][n]['seeds'];means[n]=average(rows);assert means[n]==sel['candidates'][n]['mean']
    eligible=[n for n in ORDER if e.guard(means[n],means['wide_control'],run.cfg)]
    assert all(sel['candidates'][n]['eligible']==(n in eligible) for n in ORDER)
    chosen=sorted(eligible,key=lambda n:(-e.key(means[n])[0],ORDER.index(n)))[0]
    assert chosen==sel['selected']
    lock=read(OUT/'lock.json');summary=read(OUT/'test_summary.json')
    assert lock['selected']==chosen and lock['selection_hash']==digest(OUT/'selection.json') and summary['lock_hash']==digest(OUT/'lock.json')
    assert len(summary['scenarios'])==64
    for name,stats in summary['scenarios'].items():
        z=e.load_npz(OUT/'test'/f'{name}.npz')
        for n,v in stats.items():
            rows=[e.metrics(z[f'{n}_{s}_prob'][None],z[f'{n}_{s}_reg'][None],z['cls'],z['score']) for s in run.cfg['seeds']]
            assert average(rows)['scenarios'][0]==v['mean'] and [r['scenarios'][0] for r in rows]==v['seeds']
    return dict(passed=True,new_full_experts=3,fold_experts=model_count,new_expert_epochs=(model_count+3)*18,gates=3,gate_steps_each=200,scenarios=64,selected=chosen)

if __name__=='__main__':
    result=audit() if '--audit' in sys.argv else preflight()
    write(OUT/('audit.json' if '--audit' in sys.argv else 'preflight.json'),result);print(result)
