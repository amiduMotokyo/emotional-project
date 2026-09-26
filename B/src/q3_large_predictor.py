"""Deploy the complete large ensemble through B's existing Q3 explanation contract."""
import json
from pathlib import Path
import numpy as np
import torch
from B.src.q3_explain import Q3Predictor,MODALITIES,MASK_NAMES
from B.src.q3_feature_data import prepare,apply_audio_vision_scale
from B.src.q3_large_fusion import build
from B.src.data import LABELS
from C.src.q2_protocol import coherent_score


class Q3LargePredictor(Q3Predictor):
    """Supplied contextual feature deletion, NOT raw-token re-encoding."""
    def __init__(self,package:Path,device='cpu'):
        self.package=Path(package);self.device=device;self.architecture='large_768_ensemble'
        self.lock=json.loads((self.package/'lock.json').read_text(encoding='utf-8'))
        self.weights=np.asarray(self.lock['weights'],dtype=np.float64);self.class_bias=np.asarray(self.lock['bias'],dtype=np.float64)
        self.models=[]
        for spec in self.lock['model_specs']:
            obj=torch.load(self.package/'models'/f"{spec['name']}.pt",map_location='cpu',weights_only=True)
            model=build(spec);model.load_state_dict(obj['state_dict']);self.models.append(model.to(device).eval())
        z=np.load(self.package/'normalization.npz');self.scale={n:(z[n+'_mean'],z[n+'_std']) for n in ['audio','vision']}
        self._sample=None;self._cache={}

    def prepare_aligned(self,aligned):
        d=prepare(aligned);apply_audio_vision_scale(d,self.scale)
        return {**{k:d[k][0] for k in ['token_ids','attention','audio','vision','tmask','amask','vmask']},'text_embedding':d['text'][0]}

    def prepare(self,*args,**kwargs):
        raise ValueError('Q3LargePredictor requires prepare_aligned with the supplied 768-dimensional text field')

    def _key(self,removed):return tuple(tuple(sorted(removed.get(m,()))) for m in MODALITIES)

    def _predict_many(self,sample,removed_list):
        results=[]
        for start in range(0,len(removed_list),64):
            part=removed_list[start:start+64];n=len(part)
            x=[np.repeat(sample[k][None],n,axis=0).copy() for k in ['text_embedding','audio','vision',*MASK_NAMES]]
            for row,removed in enumerate(part):
                for j,modality in enumerate(MODALITIES):
                    positions=list(removed.get(modality,()))
                    if any(p<0 or p>=50 or not sample[MASK_NAMES[j]][p] for p in positions):raise ValueError('Invalid evidence deletion')
                    x[j][row,positions]=0;x[j+3][row,positions]=False
            tensors=[torch.from_numpy(v).to(self.device) for v in x]
            probs=np.zeros((n,3),dtype=np.float64);raw=np.zeros(n,dtype=np.float64);gates=np.zeros((n,3),dtype=np.float64)
            with torch.inference_mode():
                for weight,model in zip(self.weights,self.models):
                    if weight==0:continue
                    logits,r,g=model(*tensors);probs+=weight*logits.softmax(-1).cpu().numpy();raw+=weight*r.cpu().numpy();gates+=weight*g.cpu().numpy()
            logits=np.log(np.maximum(probs,1e-9));biased=logits+self.class_bias;out=np.exp(biased-biased.max(-1,keepdims=True));out/=out.sum(-1,keepdims=True)
            pred=out.argmax(-1);score=coherent_score(pred,raw)
            for i in range(n):results.append(dict(predicted_class=int(pred[i]),predicted_polarity=LABELS[pred[i]],predicted_intensity=float(score[i]),raw_intensity=float(raw[i]),raw_logits=logits[i].tolist(),biased_logits=biased[i].tolist(),probabilities=out[i].tolist(),gate_weights=gates[i].tolist()))
        return results

    def predict(self,sample,removed=None):
        if self._sample is not sample:self._sample=sample;self._cache={}
        removed=removed or {};key=self._key(removed)
        if key not in self._cache:self._cache[key]=self._predict_many(sample,[removed])[0]
        return self._cache[key]

    def explain(self,sample,include_local=True):
        self._sample=sample;self._cache={};requests={self._key({}):{}}
        for m,mask in zip(MODALITIES,MASK_NAMES):
            positions=np.flatnonzero(sample[mask]).tolist()
            if positions:
                remove={m:tuple(positions)};requests[self._key(remove)]=remove
            if include_local:
                for width in [1,2,3]:
                    for start in positions:
                        window=tuple(p for p in range(start,min(50,start+width)) if sample[mask][p]);remove={m:window};requests[self._key(remove)]=remove
        for key,pred in zip(requests,self._predict_many(sample,list(requests.values()))):self._cache[key]=pred
        result=super().explain(sample,include_local)
        result['explanation_scope']='Deletion of supplied contextual feature positions, not raw-text causal deletion; score is log ensemble probability plus fixed bias.'
        return result
