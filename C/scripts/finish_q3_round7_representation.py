"""Generate Q3 evidence, audit, archive, and rerun the isolated archive."""
import sys, os, subprocess, zipfile, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
import torch
import numpy as np
from C.scripts import run_q3_round7_representation as context
from C.scripts.infer_q3_minilm_round6 import main as infer
from C.scripts.check_q3_minilm_round6 import main as check
from C.scripts.report_q3_round7_representation import main as report
PACKAGE=ROOT/'C/outputs/submission/q3_round7_representation_v1'


def quantization_audit():
    from B.src.data import load_npz, apply_audio_vision_scale
    from C.scripts.deploy_q3_minilm_round6 import int8_panel
    from C.scripts.deploy_q3_round7_representation import factory
    run=context.Run();lock=context.read(context.OUT/'lock.json')
    valid=load_npz(context.PREP/'valid.npz');indices=np.concatenate([np.flatnonzero(valid['cls']==c)[:4] for c in range(3)])
    d=context.base.subset(valid,indices);norm=np.load(context.OUT/'full_normalization.npz')
    apply_audio_vision_scale(d,{k:(norm[k+'_mean'],norm[k+'_std']) for k in ['audio','vision']})
    conditions=[(None,0,'middle')]+[(m,.3,'middle') for m in ['text','audio','vision']]
    reports=[]
    for group in lock['refit_groups']:
        for seed in run.cfg['seeds']:
            path=context.OUT/'deploy'/f'{group}_{seed}';model,_,_=factory(run.cfg['groups'][group])
            model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
            prob,reg=int8_panel(path/'text_encoder_finetuned_fp32.onnx',model,d,conditions)
            expected=np.load(context.OUT/'valid'/f'fp32_{group}_{seed}.npz')
            err=float(np.max(np.abs(prob-expected['prob'][:,indices])))
            regerr=float(np.max(np.abs(reg-expected['reg'][:,indices])))
            assert err<1e-4 and regerr<1e-4,(group,seed,err,regerr)
            fp=context.read(context.OUT/'valid'/f'fp32_{group}_{seed}.json')['scenarios'][0]['accuracy']
            quant=context.read(path/'valid.json')['scenarios'][0]['accuracy']
            reports.append(dict(group=group,seed=seed,fp32_onnx_probability_max_error=err,regression_max_error=regerr,
                                fp32_minus_int8_accuracy=fp-quant))
    context.write(context.OUT/'quantization_audit.json',dict(passed=True,rows_per_model=len(indices),
        scope='Torch FP32 versus batch-one ONNX FP32, clean plus three missing conditions; quantization accuracy reported separately',models=reports))


def main():
    # Local conda FFprobe needs its matching package-cache DLL directories.
    conda=Path(sys.executable).parents[2]
    ffprobe=conda/'pkgs/ffmpeg-7.1.0-gpl_h2585aa8_705/Library/bin/ffprobe.exe'
    ffmpeg=Path(sys.executable).parent/'Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    assert ffmpeg.exists() and ffprobe.exists()
    os.environ['PATH']=';'.join(str(p) for p in (conda/'pkgs').glob('*/Library/bin'))+';'+os.environ['PATH']
    subprocess.run([str(ffprobe),'-version'],check=True,stdout=subprocess.DEVNULL)
    args=['--package',str(PACKAGE),'--data-root',str(ROOT/'data'),'--ffmpeg',str(ffmpeg),'--ffprobe',str(ffprobe)]
    torch.set_num_threads(2);quantization_audit()
    sys.argv=['infer',*args];infer()
    check(context=context,package=PACKAGE);report()
    isolated=Path(tempfile.mkdtemp(prefix='q3_round7_',dir=ROOT/'C/outputs'))
    with zipfile.ZipFile(PACKAGE.with_suffix('.zip')) as archive: archive.extractall(isolated)
    manifest=context.read(isolated/'manifest.json')
    assert all(context.digest(isolated/k)==v['sha256'] for k,v in manifest.items())
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'C/outputs/q2_round3_env')
    cmd=[sys.executable,str(isolated/'C/scripts/infer_q3_minilm_round6.py'),'--package',str(isolated),
         '--data-root',str(ROOT/'data'),'--output',str(isolated/'rerun_results'),'--ffmpeg',str(ffmpeg),'--ffprobe',str(ffprobe)]
    result=subprocess.run(cmd,cwd=isolated,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (context.OUT/'package_smoke.log').write_text(result.stdout,encoding='utf-8');assert result.returncode==0,result.stdout[-2000:]
    actual=context.read(isolated/'rerun_results/attachment4_explanations.json')
    expected=context.read(PACKAGE/'results/attachment4_explanations.json')
    assert len(actual)==20 and actual==expected,'isolated prediction/explanation mismatch'
    context.write(context.OUT/'package_smoke.json',dict(passed=True,rows=20,exact_prediction_explanation_and_mapping_match=True,
        isolated_unzip=True,isolated_directory=str(isolated),dependencies='Pinned dependency directory only; implementation loaded from isolated archive'))
    report();print('ROUND7_FINISHED',flush=True)


if __name__=='__main__':main()
