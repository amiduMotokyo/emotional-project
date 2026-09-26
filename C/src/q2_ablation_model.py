"""Controlled ablation adapters; reuse the shared encoder, projections and heads."""
import torch
from torch import nn
from B.src.fusion import Fusion
from B.src.q3_minilm_fusion import MiniLMFusion

class AblationFusion(Fusion):
    def __init__(self, kind='base'):
        super().__init__(dim=64)
        self.kind=kind
        if kind=='interaction':
            self.cross=nn.MultiheadAttention(64,4,dropout=.25,batch_first=True)
            self.norm=nn.LayerNorm(64)
        if kind=='experts':
            self.text_expert=nn.Linear(64,4)
            self.av_expert=nn.Linear(128,4)

    def forward(self,text,audio,vision,tm,am,vm):
        masks=[tm.bool(),am.bool(),vm.bool()]
        xs=[p(x.float()) for p,x in zip([self.tproj,self.aproj,self.vproj],[text,audio,vision])]
        if self.kind=='interaction':
            updated=[]
            for i in range(3):
                kv=torch.cat([xs[j] for j in range(3) if j!=i],1)
                mask=torch.cat([masks[j] for j in range(3) if j!=i],1)
                absent=~mask.any(1);safe=mask.clone();safe[absent,0]=True
                delta,_=self.cross(xs[i],kv*mask[...,None],kv*mask[...,None],key_padding_mask=~safe,need_weights=False)
                updated.append(self.norm(xs[i]+delta.masked_fill(absent[:,None,None],0)))
            xs=updated
        if self.kind=='mean_pool':
            hs=[(x*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1) for x,m in zip(xs,masks)]
        else:hs=[p(x,m) for p,x,m in zip(self.pools,xs,masks)]
        rates=torch.stack([m.float().mean(1) for m in masks],1)
        present=torch.stack([m.any(1) for m in masks],1)
        if self.kind=='mean_fusion':g=present.float()
        else:g=self.gate(torch.cat([*hs,rates],1)).masked_fill(~present,-1e4).softmax(1)*present
        g=g/g.sum(1,keepdim=True).clamp_min(1e-8)
        fused=sum(g[:,i:i+1]*hs[i] for i in range(3))
        out=self.head(torch.cat([*hs,fused,rates],1))
        if self.kind=='experts':
            ep=torch.stack([present[:,0],present[:,1:].any(1)],1).float()
            ep=ep/ep.sum(1,keepdim=True).clamp_min(1)
            experts=ep[:,:1]*self.text_expert(hs[0])+ep[:,1:]*self.av_expert(torch.cat(hs[1:],1))
            out=.5*(out+experts)
        return out[:,:3],3*out[:,3].tanh(),g

class AblationModel(MiniLMFusion):
    def __init__(self,encoder,spec):
        super().__init__(encoder,dim=64,scope=spec.get('scope','top2'))
        original=self.fusion.state_dict()
        self.fusion=AblationFusion(spec.get('kind','base'))
        self.fusion.load_state_dict(original,strict=False)
        self.use_text=0 in spec.get('modalities',[0,1,2])
        if not self.use_text:self.encoder.requires_grad_(False)

    def forward(self,ids,attention,audio,vision,tm,am,vm):
        if self.use_text:
            text=self.encoder(input_ids=ids.long(),attention_mask=attention.long(),token_type_ids=torch.zeros_like(ids).long()).last_hidden_state
        else:text=audio.new_zeros((*ids.shape,384))
        return self.fusion(text,audio,vision,tm,am,vm)
