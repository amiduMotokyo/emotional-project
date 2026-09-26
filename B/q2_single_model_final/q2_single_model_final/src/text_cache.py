"""Re-encode missing words BEFORE BERT; cache frozen views, not labels."""
from common import *
import hashlib
_MODEL=None

def bert_model(dev):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel
        _MODEL=AutoModel.from_pretrained(str(ROOT/'models/bert-base-uncased'),local_files_only=True).to(dev).eval()
        for p in _MODEL.parameters(): p.requires_grad_(False)
    return _MODEL

@torch.inference_mode()
def text_view(data,O,dev):
    changed=data['O'][:,:,0] & ~O[:,:,0]
    if not changed.any(): return data['T']
    h=hashlib.sha256(data['bert'].tobytes()+changed.tobytes()).hexdigest()
    folder=CACHE/'text_views'; folder.mkdir(parents=True,exist_ok=True)
    f=folder/(h+'.npy')
    if f.exists(): return np.load(f,mmap_mode='r')
    result=data['T'].copy(); result[~O[:,:,0]]=0
    rows=np.flatnonzero(changed.any(1) & O[:,:,0].any(1)); model=bert_model(dev)
    stats=np.load(ROOT/'checkpoints/normalization.npz')
    for st in range(0,len(rows),64):
        ix=rows[st:st+64]; b=data['bert'][ix].copy()
        b[:,0][changed[ix]]=100
        out=model(input_ids=torch.tensor(b[:,0],device=dev),attention_mask=torch.tensor(b[:,1],device=dev),token_type_ids=torch.tensor(b[:,2],device=dev)).last_hidden_state.float().cpu().numpy()
        out=np.clip((out-stats['T_mean'])/stats['T_std'],-10,10)
        out[~O[ix,:,0]]=0
        result[ix]=out
    np.save(f,result.astype(np.float16))
    return np.load(f,mmap_mode='r')

def build_training_bank(data,dev,count=6):
    manifest=[]
    for k in range(count):
        f=CACHE/f'train_mask_bank{k}.npy'
        if f.exists(): O=np.load(f)
        else:
            O=training_mask(data['P'],data['O'],np.random.default_rng(91000+k),False)
            np.save(f,O)
        T=text_view(data,O,dev); manifest.append((O,T))
        print('TRAINING VIEW BANK',k,'ready',flush=True)
    return manifest
