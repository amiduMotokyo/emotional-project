"""Larger 768/74/35 sequence fusion models shared with C's Q3 training."""
import torch
from torch import nn
from B.src.fusion import MaskedAttentionPool


class SequenceFusion(nn.Module):
    def __init__(self,kind='gru',dim=128,layers=2,heads=4,dropout=.2):
        super().__init__();self.kind=kind
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(n,dim),nn.LayerNorm(dim),nn.GELU()) for n in (768,74,35)])
        if kind=='gru':
            self.encoders=nn.ModuleList([nn.GRU(dim,dim,batch_first=True,bidirectional=True) for _ in range(3)]);width=2*dim
        elif kind=='transformer':
            self.position=nn.Parameter(torch.randn(1,50,dim)*.02)
            self.modality=nn.Parameter(torch.randn(3,dim)*.02)
            layer=nn.TransformerEncoderLayer(dim,heads,dim*2,dropout,batch_first=True,activation='gelu',norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);width=dim
        else:raise ValueError(kind)
        self.pools=nn.ModuleList([MaskedAttentionPool(width) for _ in range(3)])
        self.gate=nn.Sequential(nn.Linear(width*3+3,dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,3))
        self.head=nn.Sequential(nn.Linear(width*4+3,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,128),nn.GELU(),nn.Linear(128,4))
        self.aux=nn.ModuleList([nn.Linear(width,4) for _ in range(3)])

    def forward_all(self,t,a,v,tm,am,vm):
        masks=[m.bool() for m in (tm,am,vm)]
        h=[p(x.float().masked_fill(~m[...,None],0))*m[...,None] for p,x,m in zip(self.projections,(t,a,v),masks)]
        if self.kind=='gru':h=[enc(x)[0]*m[...,None] for enc,x,m in zip(self.encoders,h,masks)]
        else:
            h=[(x+self.position+self.modality[j])*m[...,None] for j,(x,m) in enumerate(zip(h,masks))]
            seq=torch.cat(h,1);valid=torch.cat(masks,1)
            seq=torch.cat([seq,torch.zeros_like(seq[:,:1])],1)
            valid=torch.cat([valid,torch.ones_like(valid[:,:1])],1)
            seq=self.encoder(seq,src_key_padding_mask=~valid)[:,:150]
            h=[seq[:,j*50:(j+1)*50]*m[...,None] for j,m in enumerate(masks)]
        pooled=[p(x,m) for p,x,m in zip(self.pools,h,masks)]
        rates=torch.stack([m.float().mean(1) for m in masks],1);present=torch.stack([m.any(1) for m in masks],1)
        joined=torch.cat([*pooled,rates],1);gates=self.gate(joined).masked_fill(~present,-1e4).softmax(-1)*present
        gates=gates/gates.sum(1,keepdim=True).clamp_min(1e-8)
        fused=sum(gates[:,j:j+1]*x for j,x in enumerate(pooled))
        out=self.head(torch.cat([*pooled,fused,rates],1));aux=torch.stack([p(x) for p,x in zip(self.aux,pooled)],1)
        return out[:,:3],3*out[:,3].tanh(),gates,aux[:,:,:3],3*aux[:,:,3].tanh()

    def forward(self,*args):return self.forward_all(*args)[:3]


def build(spec):
    return SequenceFusion(**{k:spec[k] for k in ['kind','dim','layers','heads','dropout'] if k in spec})
