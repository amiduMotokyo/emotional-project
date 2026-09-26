from pathlib import Path
import os, json, random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
CACHE = WORK / 'cache'
DATA = Path(os.environ.get('MOSEI_DATA_ROOT', str(ROOT / 'data')))
MODS = ('T', 'A', 'V')
NAMES = ('Negative', 'Neutral', 'Positive')

def dump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def device():
    return torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

def load_split(split):
    z = np.load(CACHE / f'{split}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}

def batch(data, ix, dev):
    return ([torch.as_tensor(data[m][ix], device=dev) for m in MODS],
            torch.as_tensor(data['P'][ix], device=dev),
            torch.as_tensor(data['O'][ix], device=dev))

def metrics(y, c, pred, probs):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
    labels = np.argmax(probs, axis=1)
    corr = np.corrcoef(y, pred)[0,1] if np.std(pred)>1e-10 and np.std(y)>1e-10 else 0.
    return {'accuracy':float(accuracy_score(c, labels)),
            'macro_f1':float(f1_score(c, labels, labels=[0,1,2], average='macro', zero_division=0)),
            'weighted_f1':float(f1_score(c, labels, labels=[0,1,2], average='weighted', zero_division=0)),
            'mae':float(mean_absolute_error(y,pred)), 'pearson':float(corr),
            'confusion':confusion_matrix(c,labels,labels=[0,1,2]).tolist()}

def make_mask(P, base, rng, mods='TAV', rate=.3, position='random', pattern='single', sync=True):
    """Drop contiguous positions inside valid token support, leaving source arrays intact."""
    out=base.copy()
    chosen=[MODS.index(m) for m in mods]
    for i in range(len(P)):
        positions=np.flatnonzero(P[i]); L=len(positions)
        if L==0: continue
        k=min(L,max(1,int(round(rate*L)))) if rate>0 else 0
        if not k: continue
        def sample():
            if pattern=='fragmented':
                # Two disjoint intervals, exact total requested positions.
                k1=(k+1)//2; k2=k//2
                if k2==0: return positions[:k1]
                gap=(L-k)//2
                return np.r_[positions[:k1],positions[k1+gap:k1+gap+k2]]
            start={'front':0,'middle':(L-k)//2,'back':L-k}.get(position)
            if start is None: start=int(rng.integers(0,L-k+1))
            return positions[start:start+k]
        ids=sample()
        for m in chosen:
            if not sync: ids=sample()
            out[i,ids,m]=False
    return out

def training_mask(P, base, rng, coalition=False):
    out=base.copy()
    for i in range(len(P)):
        if rng.random()<.2: continue
        if coalition and rng.random()<.25:
            code=int(rng.integers(0,8))
            for m in range(3):
                if not code & (1<<m): out[i,:,m]=False
            continue
        num=int(rng.choice([1,2,3],p=[.5,.35,.15]))
        ms=''.join(MODS[j] for j in rng.choice(3,num,replace=False))
        out[i:i+1]=make_mask(P[i:i+1],out[i:i+1],rng,ms,float(rng.uniform(.1,.6)),sync=bool(rng.integers(2)))
    return out

# Attachment3/4 were serialized with NumPy2's private module name; retain NumPy1 ABI for Torch2.3.
def load_pickle(path):
    import pickle
    class CompatibleUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module.startswith('numpy._core'): module=module.replace('numpy._core','numpy.core',1)
            return super().find_class(module,name)
    with open(path,'rb') as f: return CompatibleUnpickler(f).load()
