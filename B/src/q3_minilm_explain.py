"""Final int8 model sensitivity: input UNK re-encoding and three-modal Shapley."""
import itertools
import math
import numpy as np
from B.src.q3_explain import Q3Predictor, MODALITIES, MASK_NAMES


class MiniLMExplanation(Q3Predictor):
    def __init__(self, package, device='cpu'):
        super().__init__(package,device,ort_threads=1)
        self._sample=None;self._predictions={}

    def predict(self,sample,removed=None):
        if self._sample is not sample:self._sample=sample;self._predictions={}
        removed=removed or {};key=tuple(tuple(sorted(removed.get(m,()))) for m in MODALITIES)
        if key not in self._predictions:
            result=super().predict(sample,removed)
            raw=np.asarray(result['raw_logits'],dtype=np.float64);raw=np.exp(raw-raw.max());raw/=raw.sum()
            result['raw_probabilities']=raw.tolist()
            result['calibrated_probabilities']=result['probabilities']
            result['model_biased_logits']=result['biased_logits']
            # Shared B deletion code reads this field; its explicit meaning here is log probability.
            result['biased_logits']=np.log(np.maximum(result['probabilities'],1e-12)).tolist()
            self._predictions[key]=result
        return self._predictions[key]

    def modality_effects(self,sample,base=None):
        result=super().modality_effects(sample,base)
        if result['main_basis']!='largest_positive_logit_drop':
            result['main_modality']=None;result['main_basis']='no_single_positive_support'
        else:result['main_basis']='largest_positive_log_probability_drop'
        return result

    def explain(self,sample,include_local=True):
        result=super().explain(sample,include_local);target=result['prediction']['predicted_class'];values={}
        for mask in range(8):
            removed={m:tuple(np.flatnonzero(sample[k]).tolist()) for j,(m,k) in enumerate(zip(MODALITIES,MASK_NAMES)) if not mask&(1<<j)}
            values[mask]=self.predict(sample,removed)['biased_logits'][target]
        phi={}
        for j,m in enumerate(MODALITIES):
            phi[m]=sum(math.factorial(mask.bit_count())*math.factorial(2-mask.bit_count())/6*(values[mask|(1<<j)]-values[mask]) for mask in range(8) if not mask&(1<<j))
        error=sum(phi.values())-(values[7]-values[0]);assert abs(error)<1e-8
        denominator=sum(abs(x) for x in phi.values())
        result['shapley']=dict(signed=phi,abs_share={k:abs(v)/denominator if denominator else 0. for k,v in phi.items()},subset_log_probabilities={str(k):v for k,v in values.items()},additivity_error=error)
        result['explanation_note']='本模型对输入词元作UNK替换后重新编码；音视频置零并同步掩码。表中logit字段实际为固定原预测类别的校准后log概率下降，门控权重不是贡献。Shapley使用全部8个模态子集；空集输出包含模型先验。局部证据为输入干预敏感性，不是人类情感因果标注。'
        return result
