"""Check baseline equivalence and information removal without altering training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
import torch
from C.scripts import run_q2_three_layer_ablation as run
from B.src.q3_minilm_fusion import MiniLMFusion
from C.src.q2_ablation_model import AblationModel
from torch.utils.data import DataLoader

torch.set_num_threads(2)
run.base.seed(20267001);reference=MiniLMFusion(run.base.MODEL,dim=64,scope='top2').eval()
run.base.seed(20267001);model=AblationModel(run.base.MODEL,{}).eval()
assert all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items())
data=run.load_npz(run.base.PREP/'train.npz');batch=next(iter(DataLoader(run.RawTextDataset(data),batch_size=3)))
run.base.DEVICE='cpu';x,_,_=run.inputs(batch,{})
with torch.inference_mode():
    a=reference(*x);b=model(*x)
    assert all(torch.allclose(v,w,atol=1e-7,rtol=1e-6) for v,w in zip(a,b))
for spec in run.make_config()['groups'].values():
    inputs,_,_=run.inputs(batch,spec)
    for j in set(range(3))-set(spec.get('modalities',[0,1,2])):
        assert not inputs[4+j].any()
        if j>0:assert not inputs[1+j].any()
run.base.write(run.OUT/'interface_check.json',dict(baseline_state_identical=True,baseline_forward_equal=True,excluded_modalities_have_zero_masks=True,checked_variants=22))
print('CHECK_OK baseline state/forward equality and 22 input contracts')
