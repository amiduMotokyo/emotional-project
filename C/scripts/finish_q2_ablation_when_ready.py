"""Wait for all 66 results, then regenerate and audit paper; report failures explicitly."""
from pathlib import Path
import time,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'C/outputs/q2_three_layer_ablation_v2'
deadline=time.monotonic()+3*3600
while time.monotonic()<deadline:
    p=OUT/'summary.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8')).get('complete'):
        try:
            subprocess.run([sys.executable,'C/scripts/build_q2_integrated_paper.py'],cwd=ROOT,check=True)
            subprocess.run(['C:/Users/ken/miniconda3/envs/myenv/python.exe','C/scripts/build_q2_integrated_paper.py','--check'],cwd=ROOT,check=True)
            (OUT/'completion.json').write_text(json.dumps(dict(training_complete=True,paper_updated=True,word_layout_checked=True,finished=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2),encoding='utf-8')
        except Exception as exc:
            (OUT/'completion_error.json').write_text(json.dumps(dict(error=str(exc))),encoding='utf-8');raise
        break
    time.sleep(20)
else:raise TimeoutError('Full training summary not available within three hours; no paper results written')
