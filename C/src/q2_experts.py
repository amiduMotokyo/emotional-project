"""Independent audio/visual expert and small availability-aware decision gate."""
import torch
from torch import nn
from B.src.fusion import MaskedAttentionPool


class AudioVisualExpert(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj=nn.ModuleList([nn.Sequential(nn.Linear(n,dim),nn.LayerNorm(dim),nn.GELU()) for n in (74,35)])
        self.pools=nn.ModuleList([MaskedAttentionPool(dim) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(dim*2+2,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,64),nn.GELU(),nn.Linear(64,4))

    def forward(self,t,a,v,tm,am,vm):
        rates=torch.stack([am.float().mean(1),vm.float().mean(1)],1)
        pooled=[pool(proj(x),mask) for pool,proj,x,mask in zip(self.pools,self.proj,(a,v),(am,vm))]
        out=self.head(torch.cat([*pooled,rates],1))
        return out[:,:3],3*out[:,3].tanh(),torch.cat([torch.zeros_like(rates[:,:1]),rates],1)


def available(rates):
    return torch.stack([rates[...,0]>0,rates[...,1:].sum(-1)>0],-1)


def sanitized(prob,reg,rates):
    present=available(rates)
    return torch.where(present[...,None],prob,torch.full_like(prob,1/3)),torch.where(present,reg,torch.zeros_like(reg))


def mix(prob,reg,rates,logits=None):
    prob,reg=sanitized(prob,reg,rates)
    present=available(rates)
    if logits is None:logits=torch.zeros_like(reg)
    weights=logits.masked_fill(~present,-1e4).softmax(-1)*present
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    p=(weights[...,None]*prob).sum(-2)
    p=torch.where(present.any(-1,keepdim=True),p,torch.full_like(p,1/3))
    return p,(weights*reg).sum(-1),weights


class DecisionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(13,8),nn.Tanh(),nn.Linear(8,2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,prob,reg,rates):
        prob,reg=sanitized(prob,reg,rates)
        entropy=-(prob*prob.clamp_min(1e-8).log()).sum(-1)
        features=torch.cat([prob.flatten(-2),reg/3,entropy,rates],-1)
        return mix(prob,reg,rates,self.net(features))
