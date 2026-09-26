"""Round-seven provenance and deployed explanation checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
from C.scripts import run_q3_round7_representation as context
from C.scripts.check_q3_minilm_round6 import main

if __name__=='__main__':
    main(context=context,package=ROOT/'C/outputs/submission/q3_round7_representation_v1')
