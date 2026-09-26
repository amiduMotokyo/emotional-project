"""Read supplied aligned PKLs; freeze a single pretrained BERT for every split."""
import os
from common import *
os.environ['HF_HOME']=str(WORK/'hf_cache')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import pickle, hashlib, time, argparse
from transformers import AutoModel, AutoTokenizer

def read_special(tag):
    fs=sorted(p for p in DATA.rglob('*.pkl') if tag in str(p) and '未对齐' not in str(p))
    samples=[]
    for f in fs:
        s=load_pickle(f); s=s.get('test',s)
        bert=np.asarray(s['text_bert']).reshape(-1,3,50)
        for j in range(len(bert)):
            raw=s.get('raw_text','')
            raw=str(raw if np.ndim(raw)==0 else raw[j])
            samples.append({'text_bert':bert[j], 'audio':np.asarray(s['audio']).reshape(-1,50,74)[j],
                'vision':np.asarray(s['vision']).reshape(-1,50,35)[j], 'id':f.stem if len(bert)==1 else f'{f.stem}_{j}',
                'raw_text':raw, 'source_file':str(f.relative_to(DATA))})
    return {k:np.asarray([s[k] for s in samples]) for k in samples[0]}

def encode(s, model, dev, name):
    bert=np.asarray(s['text_bert']).astype(np.int64)
    n=len(bert); ids=bert[:,0]; att=bert[:,1].astype(bool)
    assert np.isin(bert[:,1],[0,1]).all()
    P=att & ~np.isin(ids,[0,101,102])
    O=np.stack([P & (ids!=100), P & np.any(s['audio']!=0,-1), P & np.any(s['vision']!=0,-1)],-1)
    chunks=[]; t0=time.time()
    for start in range(0,n,32):
        b=bert[start:start+32]
        with torch.inference_mode():
            out=model(input_ids=torch.tensor(b[:,0],device=dev), attention_mask=torch.tensor(b[:,1],device=dev),
                      token_type_ids=torch.tensor(b[:,2],device=dev)).last_hidden_state.float().cpu().numpy()
        chunks.append(out)
        if start%320==0: print(f'ENCODE {name} {start}/{n} elapsed={time.time()-t0:.1f}s',flush=True)
    out={'T':np.concatenate(chunks),'A':np.asarray(s['audio'],dtype=np.float32),
         'V':np.asarray(s['vision'],dtype=np.float32),'P':P,'O':O,
         'ids':np.asarray(s['id'],dtype=str),'raw_text':np.asarray(s.get('raw_text',['']*n),dtype=str),'bert':bert}
    if 'regression_labels' in s:
        out['y']=np.asarray(s['regression_labels'],dtype=np.float32)
        out['c']=np.asarray(s['classification_labels'],dtype=np.int64)
        assert np.array_equal(out['c'],np.where(out['y']<0,0,np.where(out['y']>0,2,1)))
    if 'source_file' in s: out['source_file']=s['source_file']
    return out

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); dev=device(); print('BERT device',dev,flush=True)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/bert-base-uncased'),local_files_only=True)
    model=AutoModel.from_pretrained(str(ROOT/'models/bert-base-uncased'),local_files_only=True).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    tokenizer_dir=ROOT/'tokenizer'
    if tokenizer_dir.is_symlink(): tokenizer_dir.unlink()
    tok.save_pretrained(tokenizer_dir)
    dump({'model':'bert-base-uncased','revision':getattr(model.config,'_commit_hash',None),
          'frozen':True,'device':str(dev),'torch':torch.__version__},ROOT/'results/encoder.json')
    ap=argparse.ArgumentParser()
    ap.add_argument('--inference-only',action='store_true')
    args=ap.parse_args()
    if args.inference_only:
        s=encode(read_special('附件3'),model,dev,'special3')
        stats=np.load(ROOT/'checkpoints/normalization.npz')
        for j,m in enumerate(MODS):
            s[m]=np.clip((s[m]-stats[m+'_mean'])/stats[m+'_std'],-10,10)
            s[m][~s['O'][:,:,j]]=0
        target=CACHE/'special3.npz'
        if target.is_symlink(): target.unlink()
        np.savez(target,**s)
        print('INFERENCE PREPARATION COMPLETE',flush=True)
        return
    raw=load_pickle(DATA/'附件2-数据集特征文件/aligned_50.pkl')
    for split in ['train','valid','test']:
        f=CACHE/f'{split}_raw.npz'
        if not f.exists(): np.savez(f,**encode(raw[split],model,dev,split))
    del raw
    for tag,split in [('附件3','special3')]:
        f=CACHE/f'{split}_raw.npz'
        if not f.exists(): np.savez(f,**encode(read_special(tag),model,dev,split))
    s=dict(np.load(CACHE/'train_raw.npz',allow_pickle=False)); stats={}
    for j,m in enumerate(MODS):
        obs=s[m][s['O'][:,:,j]]
        stats[m+'_mean']=obs.mean(0,dtype=np.float64).astype(np.float32)
        stats[m+'_std']=np.maximum(obs.std(0,dtype=np.float64),1e-5).astype(np.float32)
    np.savez(ROOT/'checkpoints/normalization.npz',**stats)
    audit={}
    for split in ['train','valid','test','special3']:
        s=dict(np.load(CACHE/f'{split}_raw.npz',allow_pickle=False))
        for j,m in enumerate(MODS):
            s[m]=np.clip((s[m]-stats[m+'_mean'])/stats[m+'_std'],-10,10)
            s[m][~s['O'][:,:,j]]=0
            assert np.isfinite(s[m]).all()
        target=CACHE/f'{split}.npz'
        if target.is_symlink(): target.unlink()
        np.savez(target,**s)
        audit[split]={'n':len(s['P']),'valid_positions':int(s['P'].sum()),'observed':[int(s['O'][:,:,j].sum()) for j in range(3)],
                      'min_length':int(s['P'].sum(1).min()),'max_length':int(s['P'].sum(1).max())}
    dump(audit,ROOT/'results/preprocessing.json')
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__': main()
