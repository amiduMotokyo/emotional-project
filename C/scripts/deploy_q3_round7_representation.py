"""Reuse round-six deployment/calibration with round-seven model metadata."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'));sys.path.insert(0,str(ROOT/'C/outputs/q3_round6_export_env'))
from C.scripts import run_q3_round7_representation as context
from C.scripts.deploy_q3_minilm_round6 import main
from B.src.fusion import Fusion
from B.src.q3_temporal_primary_fusion import TemporalPrimaryFusion

PACKAGE=ROOT/'C/outputs/submission/q3_round7_representation_v1'


def factory(spec):
    if spec['fusion']=='original': return Fusion(dim=spec['dim']), 'fusion', {'dim':spec['dim']}
    kwargs=dict(mode=spec['fusion'],dim=spec['dim'])
    return TemporalPrimaryFusion(**kwargs), 'temporal_primary', kwargs


if __name__=='__main__':
    sources={str(p):context.digest(p) for p in [Path(__file__), ROOT/'C/scripts/deploy_q3_minilm_round6.py',ROOT/'C/scripts/export_q2_minilm_onnx.py']}
    p=context.OUT/'deployment_sources.json'
    if p.exists(): assert context.read(p)==sources
    else: context.write(p,sources)
    main(context=context,package=PACKAGE,model_factory=factory)
