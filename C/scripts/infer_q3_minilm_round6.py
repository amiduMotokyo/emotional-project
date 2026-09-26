"""Portable inference entry point for the round-six MiniLM Q3 package."""
import sys,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
if (ROOT/'C/outputs/q2_round3_env').exists():sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
import torch
from transformers import AutoTokenizer
from B.src.q3_minilm_explain import MiniLMExplanation
from B.scripts.run_q3 import run_attachment4

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=ROOT);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--ffmpeg',type=Path,default=shutil.which('ffmpeg'));p.add_argument('--ffprobe',type=Path,default=shutil.which('ffprobe'));args=p.parse_args()
    torch.set_num_threads(2);out=args.output or args.package/'results';out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(str(args.package/'tokenizer'),local_files_only=True,use_fast=True)
    model=MiniLMExplanation(args.package,'cpu');run_attachment4(model,args.data_root,tokenizer,out,args.ffprobe,args.ffmpeg)

if __name__=='__main__':main()
