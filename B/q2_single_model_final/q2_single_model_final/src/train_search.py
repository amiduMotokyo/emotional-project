"""A4 configuration search. Train labels update weights; complete valid labels select."""
from common import *
from model import Predictor,predict
from calibration import search_bias,selection_key
from text_cache import build_training_bank
from torch.nn import functional as F
import copy,time

def train_one(setting,seed,config,train,valid,dev,bank):
    name=setting['name']; target=ROOT/'checkpoints'/f'{name}_seed{seed}_cal.pt'
    complete=ROOT/'results'/f'{name}_seed{seed}_complete.json'
    if complete.exists(): print('SKIP',name,seed,flush=True); return
    cfg=copy.deepcopy(config)
    cfg.update({k:v for k,v in setting.items() if k not in ['name','pooled_skip']})
    spec=copy.deepcopy(config['variants']['A4_consistency']); spec['pooled_skip']=setting.get('pooled_skip',False)
    seed_all(seed); rng=np.random.default_rng(seed)
    model=Predictor(spec,cfg['hidden_dim'],cfg['dropout']).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['max_epochs'],eta_min=8e-5)
    weights=(len(train['c'])/(3*np.bincount(train['c'],minlength=3)))**.5
    weights=torch.as_tensor(weights/weights.mean(),dtype=torch.float32,device=dev)
    def supervised(y,log,ty,tc):
        return F.huber_loss(y,ty,delta=1.)+cfg['classification_weight']*F.cross_entropy(log,tc,weight=weights)
    best=None; best_raw=None; history=[]; started=time.time()
    for epoch in range(1,cfg['max_epochs']+1):
        model.train(); total=0
        for start in range(0,len(train['P']),cfg['batch_size']):
            if start==0: order=rng.permutation(len(train['P']))
            ix=order[start:start+cfg['batch_size']]; xs,p,o=batch(train,ix,dev)
            ty=torch.as_tensor(train['y'][ix],device=dev); tc=torch.as_tensor(train['c'][ix],device=dev)
            bids=rng.integers(0,len(bank),len(ix)); om=np.stack([bank[b][0][i] for b,i in zip(bids,ix)])
            tx=np.stack([bank[b][1][i] for b,i in zip(bids,ix)]).astype(np.float32)
            om=torch.as_tensor(om,device=dev); xm=[torch.as_tensor(tx,device=dev),xs[1],xs[2]]
            opt.zero_grad(set_to_none=True)
            ys,ls=model([torch.cat([x,xx]) for x,xx in zip(xs,xm)],torch.cat([p,p]),torch.cat([o,om]))
            yf,ym=ys.chunk(2); lf,lm=ls.chunk(2)
            loss=.5*(supervised(yf,lf,ty,tc)+supervised(ym,lm,ty,tc))
            retained=(om.sum((1,2))/o.sum((1,2)).clamp_min(1)).detach()
            cons=(ym-yf.detach()).abs()+.5*F.kl_div(lm.log_softmax(-1),lf.detach().softmax(-1),reduction='none').sum(-1)
            loss+=cfg['consistency_weight']*(retained*cons).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']); opt.step()
            total+=float(loss.detach())*len(ix)
        scheduler.step(); model.eval(); y,p=predict(model,valid,dev)
        raw=metrics(valid['y'],valid['c'],y,p); cal=search_bias(p,valid['c'])
        row={'epoch':epoch,'loss':total/len(train['P']),'raw':raw,'calibration':cal,'elapsed_seconds':time.time()-started}
        history.append(row)
        np.savez_compressed(ROOT/'results/epoch_predictions'/f'{name}_{seed}_{epoch:02d}.npz',y=y,p=p)
        def save(path,rule):
            torch.save({'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'spec':spec,'config':cfg,
                        'variant':name,'seed':seed,'best_epoch':epoch,'selection_rule':rule,'calibration':cal,'valid_raw':raw},path)
        key=selection_key(cal,raw['mae'])
        if best is None or key>best:
            best=key; save(target,'calibrated complete validation accuracy')
        rawkey=(raw['accuracy'],raw['macro_f1'],-raw['mae'])
        if best_raw is None or rawkey>best_raw:
            best_raw=rawkey; save(target.with_name(f'{name}_seed{seed}_raw.pt'),'raw complete validation accuracy')
        dump(history,ROOT/'results'/f'training_{name}_seed{seed}.json')
        print(name,seed,epoch,'raw',round(raw['accuracy'],4),'cal',round(cal['accuracy'],4),'bias',cal['bias'],'seconds',round(time.time()-started,1),flush=True)
    dump({'complete':True,'name':name,'seed':seed,'epochs':cfg['max_epochs']},complete)

def main():
    (ROOT/'results/epoch_predictions').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); cfg=json.loads((ROOT/'protocol.json').read_text()); dev=device()
    train=load_split('train'); valid=load_split('valid'); bank=build_training_bank(train,dev)
    for setting in cfg['search_configs']:
        for seed in cfg['seeds']: train_one(setting,seed,cfg,train,valid,dev,bank)
    dump({'complete':True,'configurations':len(cfg['search_configs']),'seeds':cfg['seeds']},ROOT/'results/training_complete.json')
    print('SEARCH TRAINING COMPLETE',flush=True)
if __name__=='__main__': main()
