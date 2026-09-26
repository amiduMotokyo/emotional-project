"""Fixed single-model prediction and evaluation; no model selection from test."""
from common import *
from model import load_model,predict
from calibration import apply_bias
import pandas as pd
import argparse

def export(d,y,p,bias,path):
    q=apply_bias(p,bias)
    out={'sample_id':d['ids'],'predicted_polarity':[NAMES[i] for i in q.argmax(1)],'predicted_intensity':y,
         'raw_head_polarity':[NAMES[i] for i in p.argmax(1)]}
    for j,n in enumerate(NAMES):
        out['raw_probability_'+n.lower()]=p[:,j]; out['biased_probability_'+n.lower()]=q[:,j]
    if 'y' in d: out.update(true_class=d['c'],true_intensity=d['y'])
    pd.DataFrame(out).to_csv(path,index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluate',action='store_true'); ap.add_argument('--output',default='attachment3_predictions.csv'); args=ap.parse_args()
    torch.set_num_threads(4); dev=device(); sel=json.loads((ROOT/'results/selection_frozen.json').read_text())
    model,_=load_model(ROOT/sel['paths'][0],dev); bias=sel['bias']
    d=load_split('special3'); y,p=predict(model,d,dev); export(d,y,p,bias,ROOT/args.output)
    print('ATTACHMENT3 COMPLETE',len(y),flush=True)
    if not args.evaluate: return
    final={}; summary=[]
    for split in ['valid','test']:
        d=load_split(split); y,p=predict(model,d,dev)
        final[split]={'raw':metrics(d['y'],d['c'],y,p),'calibrated':metrics(d['y'],d['c'],y,apply_bias(p,bias))}
        final[split]['sign_conflicts']=int((((apply_bias(p,bias).argmax(1)==0)&(y>0))|((apply_bias(p,bias).argmax(1)==2)&(y<0))).sum())
        export(d,y,p,bias,ROOT/'results'/f'{split}_predictions.csv')
        for rule in ['raw','calibrated']: summary.append({'split':split,'rule':rule,**{k:v for k,v in final[split][rule].items() if k!='confusion'}})
        print('FULL',split,final[split],flush=True)
    dump(final,ROOT/'results/final_metrics.json'); pd.DataFrame(summary).to_csv(ROOT/'results/full_metrics.csv',index=False)
    cfg=json.loads((ROOT/'protocol.json').read_text()); valid=load_split('valid'); rows=[]
    cases=[(m,rate,'random',ms,'rate') for m in ['T','A','V','TAV'] for rate in cfg['rates'] for ms in cfg['evaluation_corruption_seeds']]
    cases += [(m,.3,pos,ms,'position') for m in cfg['modalities'] for pos in cfg['positions'] for ms in cfg['evaluation_corruption_seeds']]
    for i,(mods,rate,pos,ms,kind) in enumerate(cases):
        O=make_mask(valid['P'],valid['O'],np.random.default_rng(ms),mods,rate,pos); y,p=predict(model,valid,dev,O)
        rows.append({'kind':kind,'modalities':mods,'rate':rate,'position':pos,'mask_seed':ms,**{k:v for k,v in metrics(valid['y'],valid['c'],y,apply_bias(p,bias)).items() if k!='confusion'}})
        if i%20==0: print('MISSING CASE',i,'/',len(cases),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/validation_missingness.csv',index=False)
    d=load_split('test'); rows=[]
    for mods in ['T','A','V','TAV']:
        O=make_mask(d['P'],d['O'],np.random.default_rng(901),mods,.3); y,p=predict(model,d,dev,O)
        rows.append({'modalities':mods,'rate':.3,'mask_seed':901,**{k:v for k,v in metrics(d['y'],d['c'],y,apply_bias(p,bias)).items() if k!='confusion'}})
    pd.DataFrame(rows).to_csv(ROOT/'results/test_missing30.csv',index=False)
    dump({'complete':True,'single_model':True,'fixed_bias':bias},ROOT/'results/evaluation_complete.json')
    print('EVALUATION COMPLETE',flush=True)
if __name__=='__main__': main()
