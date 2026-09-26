"""Finite, predeclared class-bias search on complete validation labels."""
from common import *
import pandas as pd

def bias_grid(negative_only=False):
    cfg=json.loads((ROOT/'protocol.json').read_text())['bias_search']
    grid=np.round(np.arange(cfg['grid_min'],cfg['grid_max']+cfg['step']/2,cfg['step']),8)
    return np.array([[a,b,0.] for a in grid for b in ([0.] if negative_only else grid)],dtype=np.float64)

def apply_bias(p,bias):
    z=np.log(np.clip(p,1e-12,1)).astype(np.float64)+np.asarray(bias)
    z-=z.max(1,keepdims=True); q=np.exp(z); return q/q.sum(1,keepdims=True)

def search_bias(p,c,negative_only=False,return_grid=False):
    biases=bias_grid(negative_only); logp=np.log(np.clip(p,1e-12,1)).astype(np.float64)
    actual=np.bincount(c,minlength=3); rows=[]; best=None
    for start in range(0,len(biases),256):
        bs=biases[start:start+256]; pred=(logp[None]+bs[:,None,:]).argmax(-1)
        correct=(pred==c[None]).sum(1); f1=np.zeros(len(bs))
        for j in range(3):
            tp=((pred==j)&(c[None]==j)).sum(1); denom=(pred==j).sum(1)+actual[j]
            f1+=np.divide(2*tp,denom,out=np.zeros(len(bs),float),where=denom>0)/3
        norm=(bs**2).sum(1)
        order=np.lexsort((norm,-f1,-correct)); j=int(order[0])
        candidate=(int(correct[j]),float(f1[j]),-float(norm[j]))
        if best is None or candidate>best[0]:
            best=(candidate,bs[j].tolist())
        if return_grid:
            rows.extend({'negative_bias':float(b[0]),'neutral_bias':float(b[1]),'correct':int(n),'accuracy':float(n/len(c)),'macro_f1':float(f)} for b,n,f in zip(bs,correct,f1))
    result={'bias':best[1],'correct':best[0][0],'accuracy':best[0][0]/len(c),'macro_f1':best[0][1],
            'squared_bias_norm':-best[0][2],'grid_candidates':len(biases)}
    return (result,pd.DataFrame(rows)) if return_grid else result

def selection_key(cal,mae):
    return (cal['correct'],cal['macro_f1'],-cal['squared_bias_norm'],-mae)
