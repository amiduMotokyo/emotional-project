"""Small sequence-interaction and factorized-polarity Q2 candidates."""
import torch
from torch import nn
from torch.nn import functional as F
from B.src.fusion import Fusion, MaskedAttentionPool


class FactorizedHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.features = head[:-1]
        self.output = nn.Linear(head[-1].in_features, 3)

    def forward(self, x):
        q, r, score = self.output(self.features(x)).unbind(-1)
        logp = torch.stack([F.logsigmoid(-q)+F.logsigmoid(-r),
                            F.logsigmoid(q), F.logsigmoid(-q)+F.logsigmoid(r)], -1)
        return torch.cat([logp, score[:, None]], -1)


class TextOnly(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(384,dim),nn.LayerNorm(dim),nn.GELU())
        self.pool=MaskedAttentionPool(dim)
        self.head=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,64),nn.GELU(),nn.Linear(64,4))

    def forward(self,t,a,v,tm,am,vm):
        o=self.head(self.pool(self.project(t),tm))
        gates=torch.zeros(t.shape[0],3,device=t.device);gates[:,0]=tm.any(1).float()
        return o[:,:3],3*o[:,3].tanh(),gates


class Interaction(nn.Module):
    """Pool after cross-attention; direct text path and independent AV fallback."""
    def __init__(self, dim=64, factorized=False, cross=True):
        super().__init__()
        self.cross=cross
        self.proj=nn.ModuleList([nn.Sequential(nn.Linear(n,dim),nn.LayerNorm(dim),nn.GELU()) for n in (384,74,35)])
        self.position=nn.Parameter(torch.randn(50,dim)*.02)
        self.attention=nn.MultiheadAttention(dim,4,dropout=.1,batch_first=True)
        self.norm=nn.LayerNorm(dim)
        self.pools=nn.ModuleList([MaskedAttentionPool(dim) for _ in range(4)])
        self.gate=nn.Linear(dim*2+3,dim)
        self.head=nn.Sequential(nn.Linear(dim*4+3,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,64),nn.GELU(),nn.Linear(64,4))
        if factorized:self.head=FactorizedHead(self.head)

    def forward(self,t,a,v,tm,am,vm):
        masks=[tm.bool(),am.bool(),vm.bool()]
        hidden=[(p(x)+self.position[None,:x.shape[1]])*m[:,:,None] for p,x,m in zip(self.proj,(t,a,v),masks)]
        query=hidden[0]
        memory=torch.cat(hidden[1:],1); observed=torch.cat(masks[1:],1)
        # Valid zero sentinel prevents all-masked softmax; output suppressed when AV absent.
        memory=torch.cat([memory,torch.zeros_like(memory[:,:1])],1)
        valid=torch.cat([observed,torch.ones_like(observed[:,:1])],1)
        update,_=self.attention(query,memory,memory,key_padding_mask=~valid,need_weights=False)
        update=update*observed.any(1)[:,None,None]
        local=self.norm(query+update)*masks[0][:,:,None]
        summaries=[p(x,m) for p,x,m in zip(self.pools,hidden,masks)]
        interaction=self.pools[3](local,masks[0])
        rates=torch.stack([m.float().mean(1) for m in masks],1)
        gate=torch.sigmoid(self.gate(torch.cat([summaries[0],interaction,rates],1)))
        residual=summaries[0]+gate*(interaction-summaries[0])
        out=self.head(torch.cat([residual,interaction,summaries[1],summaries[2],rates],1))
        return out[:,:3],3*out[:,3].tanh(),rates/rates.sum(1,keepdim=True).clamp_min(1e-8)


def build(name):
    if name=='baseline':return Fusion()
    if name=='text_only':return TextOnly()
    if name=='factorized':
        m=Fusion();m.head=FactorizedHead(m.head);return m
    if name in ('interaction','combined'):return Interaction(factorized=name=='combined')
    if name=='wide_control':
        target=sum(p.numel() for p in Interaction().parameters())
        # Select dimension solely by parameter count, before any training or metrics.
        dim=min(range(64,161),key=lambda d:abs(sum(p.numel() for p in Fusion(dim=d).parameters())-target))
        return Fusion(dim=dim)
    raise ValueError(name)
