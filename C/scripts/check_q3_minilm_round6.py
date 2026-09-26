"""Audit training provenance, deployed predictions, and explanation arithmetic."""
import sys,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
import numpy as np
import torch
from C.scripts.run_q3_minilm_round6 import Run,OUT,PREP,MODEL,read,write,evaluate,key
from B.src.q3_minilm_explain import MiniLMExplanation
from B.src.data import load_npz
from B.src.q3_explain import MODALITIES,MASK_NAMES

PACKAGE=ROOT/'C/outputs/submission/q3_round6_minilm_v2'

def main(context=None, package=None):
    context=context or sys.modules[__name__]
    OUT=context.OUT; PACKAGE=package or globals()['PACKAGE']
    run=context.Run();lock=read(OUT/'lock.json');torch.set_num_threads(2)
    for path in (OUT/'state').glob('*.json'):assert run.done(path.stem)
    for g in run.cfg['groups']:
        for s in run.cfg['seeds']:
            path=OUT/'inner'/f'{g}_{s}';history=read(path/'history.json');best=max(history,key=lambda x:key(x['metrics']))
            assert best['epoch']==read(path/'result.json')['epoch']==lock['epochs'][g][str(s)]
    for g in lock['refit_groups']:
        for s in run.cfg['seeds']:
            path=OUT/'final'/f'{g}_{s}';result=read(path/'result.json');assert result['train_samples']==3395 and result['epochs_run']==lock['epochs'][g][str(s)]
            obj=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            assert obj['epoch']==result['epoch']
    saved=np.load(OUT/'final_grid.npz');report=read(OUT/'final_grid.json');assert evaluate(saved['prob'],saved['reg'],saved['cls'],saved['score'])==report['metrics']
    predictor=MiniLMExplanation(PACKAGE);valid=load_npz(PREP/'valid.npz');indices=np.concatenate([np.flatnonzero(valid['cls']==c)[:4] for c in range(3)]);checks=[]
    for index in indices:
        sample=predictor.prepare(valid['token_ids'][index],valid['attention'][index],valid['audio'][index],valid['vision'][index]);base=predictor.predict(sample)
        assert base['predicted_class']==saved['prob'][0,index].argmax()
        assert np.allclose(base['probabilities'],saved['prob'][0,index],atol=2e-6)
        explanation=predictor.explain(sample);target=base['predicted_class']
        assert abs(explanation['shapley']['additivity_error'])<1e-8
        for j,m in enumerate(MODALITIES):
            local=explanation['local'][m]['top_window']
            if local is None:continue
            width=len(local['aligned_positions_zero_based']);positions=np.flatnonzero(sample[MASK_NAMES[j]]).tolist()
            candidates=[tuple(positions[k:k+width]) for k in range(len(positions)-width+1) if positions[k+width-1]-positions[k]+1==width]
            if not candidates:continue
            windows=random.Random(20267000+int(index)*3+j).choices(candidates,k=5);effects=[];flips=[]
            for window in windows:
                pred=predictor.predict(sample,{m:window});effects.append(base['biased_logits'][target]-pred['biased_logits'][target]);flips.append(pred['predicted_class']!=target)
            checks.append(dict(row=int(index),modality=m,width=width,top_effect=local['joint_delta_logit'],top_direction=local['direction'],random_mean_effect=float(np.mean(effects)),random_flip_rate=float(np.mean(flips))))
    records=read(PACKAGE/'results/attachment4_explanations.json');assert len(records)==20 and len({r['sample_id'] for r in records})==20
    previews={'audio':0,'vision':0};statuses={}
    for r in records:
        p=r['prediction'];probs=np.array(p['probabilities']);assert np.isclose(probs.sum(),1) and p['predicted_class']==probs.argmax() and -3<=p['predicted_intensity']<=3
        if p['predicted_class']==1:assert p['predicted_intensity']==0
        assert abs(r['shapley']['additivity_error'])<1e-8
        for m in MODALITIES:
            status=r['localized_evidence'][m]['mapping_status'];statuses[m+':'+status]=statuses.get(m+':'+status,0)+1
        for m,keyname,folder in [('audio','audio_preview','audio'),('vision','visual_preview','frames')]:
            if r[keyname]:assert (PACKAGE/'results'/folder/r[keyname]).exists();previews[m]+=1
        assert (PACKAGE/'results/cards'/f"{r['sample_id']}.md").exists()
    write(OUT/'audit.json',dict(passed=True,inner_models=len(run.cfg['groups'])*len(run.cfg['seeds']),refit_models=len(lock['refit_groups'])*len(run.cfg['seeds']),valid_deployment_rows_checked=len(indices),attachment4_rows=20,previews=previews,mapping_statuses=statuses,test_evaluated=False,explanation_checks=checks,explanation_check_scope='First four valid examples of each class; descriptive sensitivity, not annotated explanation accuracy.'))
    print('AUDIT_PASS',report['group'],previews,statuses,flush=True)

if __name__=='__main__':main()
