"""One-command Attachment-4 inference and evidence export from a packaged ensemble."""
import argparse
import sys
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from B.src.q3_large_predictor import Q3LargePredictor
from B.scripts.run_q3 import run_attachment4

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--package',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ffmpeg',type=Path,default=shutil.which('ffmpeg'));p.add_argument('--ffprobe',type=Path,default=shutil.which('ffprobe'));a=p.parse_args()
    torch.set_num_threads(2);out=a.output or a.package/'results';out.mkdir(parents=True,exist_ok=True)
    predictor=Q3LargePredictor(a.package,a.device)
    tokenizer=AutoTokenizer.from_pretrained(str(a.package/'tokenizer'),local_files_only=True,use_fast=True)
    run_attachment4(predictor,a.data_root,tokenizer,out,a.ffprobe,a.ffmpeg)

if __name__=='__main__':main()
