from common import *
from torch import nn
import math

class Predictor(nn.Module):
    def __init__(self, spec, d=48, dropout=.2):
        super().__init__(); self.spec=spec; self.d=d
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(k+int(spec['mask_aware']),d),nn.LayerNorm(d),nn.GELU()) for k in [768,74,35]])
        self.encoders=nn.ModuleList([nn.TransformerEncoderLayer(d,4,2*d,dropout,batch_first=True,activation='gelu',norm_first=True) for _ in range(3)])
        self.missing=nn.Parameter(torch.zeros(3,d))
        self.all_missing=nn.Parameter(torch.zeros(d))
        pos=torch.arange(50)[:,None]; div=torch.exp(torch.arange(0,d,2)*(-math.log(10000.)/d))
        pe=torch.zeros(50,d); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pos',pe*.1)
        self.gate=nn.Sequential(nn.Linear(3*d+3,d),nn.GELU(),nn.Linear(d,3))
        self.concat=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.LayerNorm(d))
        self.attn=nn.Linear(d,1)
        self.output=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(dropout))
        self.reg=nn.Linear(d,1); self.cls=nn.Linear(d,3)
        if spec.get("pooled_skip",False): self.skip=nn.Sequential(nn.Linear(4*d+3,d),nn.GELU(),nn.LayerNorm(d))

    def forward(self, xs, P, O, return_attention=False):
        hs=[]
        for m,x in enumerate(xs):
            obs=O[:,:,m]
            x=x*obs[:,:,None]
            if self.spec['mask_aware']: x=torch.cat([x,obs[:,:,None].float()],-1)
            u=self.projections[m](x)
            if self.spec['mask_aware']: u=torch.where(obs[:,:,None],u,self.missing[m])
            u=(u+self.pos)*P[:,:,None]
            # P has at least one content token in every supplied sample.
            h=self.encoders[m](u,src_key_padding_mask=~P)
            hs.append(h*P[:,:,None])
        h=torch.stack(hs,2)
        if self.spec['dynamic']:
            e=self.gate(torch.cat(hs+[O.float()],-1)).masked_fill(~O,-1e4)
            a=e.softmax(-1)*O.float()
            a=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
            z=(a[:,:,:,None]*h).sum(2)
            z=torch.where(O.any(-1)[:,:,None],z,self.all_missing+self.pos)
        else:
            z=self.concat(torch.cat(hs,-1)); a=torch.zeros_like(O,dtype=z.dtype)
        b=self.attn(z).squeeze(-1).masked_fill(~P,-1e4).softmax(-1)*P
        b=b/b.sum(-1,keepdim=True).clamp_min(1e-8)
        fused=(z*b[:,:,None]).sum(1)
        if self.spec.get('pooled_skip',False):
            pieces=[]
            for m in range(3):
                weight=b*O[:,:,m]
                weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-8)
                pieces.append((h[:,:,m]*weight[:,:,None]).sum(1))
            rates=O.sum(1)/P.sum(1,keepdim=True).clamp_min(1)
            fused=self.skip(torch.cat(pieces+[fused,rates],-1))
        g=self.output(fused)
        y=3*self.reg(g).squeeze(-1).tanh(); logits=self.cls(g)
        return (y,logits,a,b) if return_attention else (y,logits)

def load_model(path, dev):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=Predictor(saved['spec'],saved['config']['hidden_dim'],saved['config']['dropout'])
    model.load_state_dict(saved['state']); return model.to(dev).eval(),saved

@torch.inference_mode()
def predict(model,data,dev,O=None,batch_size=256):
    ys=[]; ps=[]
    text=None
    if O is not None:
        from text_cache import text_view
        text=text_view(data,O,dev)
    for start in range(0,len(data['P']),batch_size):
        ix=slice(start,start+batch_size); xs,p,o=batch(data,ix,dev)
        if O is not None:
            o=torch.as_tensor(O[ix],device=dev)
            xs[0]=torch.as_tensor(np.array(text[ix],dtype=np.float32),device=dev)
        y,log=model(xs,p,o); ys.append(y.cpu().numpy()); ps.append(log.softmax(-1).cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)
