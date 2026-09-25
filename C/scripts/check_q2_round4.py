"""Synthetic mechanism checks and immutable result audit, no new fitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from C.src.q2_structural import build,FactorizedHead
from C.scripts.run_q2_optimization import read,write,digest
OUT=ROOT/'C/outputs/q2_round4_structure_v1'


def preflight():
    torch.set_num_threads(1);torch.manual_seed(44)
    cfg=read(ROOT/'C/configs/q2_round4_structure.json')
    inputs=[torch.randn(3,50,d) for d in (384,74,35)]
    masks=[torch.rand(3,50)>.3 for _ in range(3)]
    counts={};deltas={}
    for name in cfg['families']:
        model=build(name).eval();counts[name]=sum(p.numel() for p in model.parameters())
        for combination in range(8):
            ms=[m & bool(combination & (1<<i)) for i,m in enumerate(masks)]
            logits,reg,_=model(*inputs,*ms)
            assert torch.isfinite(logits).all() and torch.isfinite(reg).all()
            assert torch.allclose(logits.softmax(-1).sum(-1),torch.ones(3),atol=1e-6)
            changed=[x.masked_fill(~m[:,:,None],9999.) for x,m in zip(inputs,ms)]
            other=model(*changed,*ms)
            assert torch.allclose(logits,other[0],atol=1e-6) and torch.allclose(reg,other[1],atol=1e-6)
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2]))+reg.square().mean()
            model.zero_grad();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if name=='text_only':
            a=model(*inputs,*masks)[0];b=model(inputs[0],inputs[1]*100,inputs[2]*100,masks[0],~masks[1],~masks[2])[0]
            assert torch.equal(a,b)
        if name in ('baseline','interaction'):
            full=[torch.ones_like(m) for m in masks];perm=torch.randperm(50)
            a=model(*inputs,*full)[0];b=model(inputs[0],inputs[1][:,perm],inputs[2],*full)[0]
            deltas[name]=float((a-b).abs().max())
        if name in ('factorized','combined'):
            # Verify the exact factorization on the head, independent of forward network.
            head=model.head;x=torch.randn(3,head.features[0].in_features)
            q,r,_=head.output(head.features(x)).unbind(-1)
            expected=torch.stack([torch.sigmoid(-q)*torch.sigmoid(-r),torch.sigmoid(q),torch.sigmoid(-q)*torch.sigmoid(r)],-1)
            assert torch.allclose(head(x)[:,:3].exp(),expected,atol=1e-6)
    assert deltas['baseline']<1e-6 and deltas['interaction']>1e-6,deltas
    error=abs(counts['wide_control']/counts['interaction']-1)
    assert error<.03
    return dict(passed=True,parameters=counts,parameter_matching_relative_error=error,permutation_delta=deltas)


def audit():
    from C.scripts.run_q2_round4 import Experiment,average_metrics,engine
    run=Experiment();states=list((OUT/'state').glob('*.json'))
    for p in states:assert run.done(p.stem)
    manifest=read(OUT/'manifest.json');implementation=read(OUT/'implementation.json')
    for p,h in {**manifest['source_hashes'],**implementation['code']}.items():assert digest(p)==h
    split=run.split;groups=np.asarray(split['source_metadata']['train']['group_ids'])
    assert not set(groups[split['fit_indices']]) & set(groups[split['stop_indices']])
    rows=list((OUT/'models').glob('*/result.json'));assert len(rows)==18
    for p in rows:
        result=read(p);history=read(p.parent/'history.json');assert len(history)==18
        best=max(history,key=lambda h:(h['clean']['accuracy'],h['clean']['macro_f1'],-h['clean']['mae']))
        assert result['best_epoch']==best['epoch']
        z=engine.load_npz(p.parent/'stop.npz')
        assert np.array_equal(z['sample_id'],run.stop['sample_id'])
        assert engine.metrics(z['prob'],z['reg'],run.stop['cls'],run.stop['score'])==result['stop']
    selection=read(OUT/'selection.json')
    for family,r in selection['families'].items():
        computed=[]
        for seed in run.cfg['seeds']:
            z=engine.load_npz(OUT/'valid_predictions'/f'{family}_{seed}.npz')
            assert np.array_equal(z['sample_id'],run.valid['sample_id'])
            computed.append(engine.metrics(z['prob'],z['reg'],run.valid['cls'],run.valid['score']))
        assert computed==r['seeds'] and average_metrics(computed)==r['mean']
        assert r['eligible']==bool(engine.guard(r['mean'],selection['families']['baseline']['mean'],run.cfg))
    eligible=[n for n,r in selection['families'].items() if r['eligible']]
    expected=sorted(eligible,key=lambda n:(-engine.key(selection['families'][n]['mean'])[0],n!='baseline',selection['families'][n]['parameters'],n))[0]
    assert selection['selected']==expected
    lock=read(OUT/'lock.json');summary=read(OUT/'test_summary.json')
    assert lock['selected']==expected and lock['selection_hash']==digest(OUT/'selection.json')
    assert summary['lock_hash']==digest(OUT/'lock.json') and len(summary['scenarios'])==64
    for name,stats in summary['scenarios'].items():
        z=engine.load_npz(OUT/'test'/f'{name}.npz')
        for family,r in stats.items():
            computed=[engine.metrics(z[f'{family}_{s}_prob'][None],z[f'{family}_{s}_reg'][None],z['cls'],z['score']) for s in run.cfg['seeds']]
            assert average_metrics(computed)['scenarios'][0]==r['mean']
            assert [c['scenarios'][0] for c in computed]==r['seeds']
    return dict(passed=True,states=len(states),models=18,epochs=324,scenarios=64,selected=expected)


if __name__=='__main__':
    result=audit() if '--audit' in sys.argv else preflight()
    write(OUT/('audit.json' if '--audit' in sys.argv else 'preflight.json'),result)
    print(result)
