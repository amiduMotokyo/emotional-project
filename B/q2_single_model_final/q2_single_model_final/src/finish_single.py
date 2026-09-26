from common import *
import pandas as pd,zipfile,hashlib

def main():
    sel=json.loads((ROOT/'results/selection_frozen.json').read_text()); final=json.loads((ROOT/'results/final_metrics.json').read_text())
    a=pd.read_csv(ROOT/'attachment3_predictions.csv',dtype={'sample_id':str})
    b=pd.read_csv(ROOT/'results/attachment3_reproduced.csv',dtype={'sample_id':str})
    assert len(a)==30 and a.sample_id.equals(b.sample_id) and a.predicted_polarity.equals(b.predicted_polarity)
    cols=['predicted_intensity']+['biased_probability_'+n.lower() for n in NAMES]
    error=float(np.max(abs(a[cols].to_numpy()-b[cols].to_numpy())))
    assert error<1e-6
    assert len(sel['paths'])==1 and sel['bias']==[.375,.275,0.]
    assert abs(final['valid']['calibrated']['accuracy']-477/728)<1e-12
    verify={'passed':True,'single_model':True,'special3_count':30,'raw_attachment3_preprocessing_reproduced':True,
            'max_prediction_difference':error,'class_predictions_identical':True,'weight_bytes':(ROOT/sel['paths'][0]).stat().st_size}
    dump(verify,ROOT/'results/verification.json')
    rows=['| 数据 | Acc | Macro-F1 | MAE | Pearson |','|---|---:|---:|---:|---:|']
    for split in ['valid','test']:
        m=final[split]['calibrated']; rows.append(f"| {split} | {m['accuracy']:.4%} | {m['macro_f1']:.4f} | {m['mae']:.4f} | {m['pearson']:.4f} |")
    (ROOT/'实验报告.md').write_text('# 问题二最终单模型结果\n\n采用32维A4，seed43，第3轮。分类偏置为[0.375,0.275,0]，只有一个预测模型。\n\n'+'\n'.join(rows)+'\n\n选择基于先前完整验证集搜索，不根据本次测试集结果换模型。单模型验证正确477/728，原三模型集成为478/728；本版本按用户指定采用单模型。附件三30条预测已从原始PKL重新编码并核验一致。\n\n模型参数文件278746字节。通用冻结BERT、环境与原始赛题数据不打入结果包；必须按README准备依赖，本包不是完全离线环境。\n\n完整缺失场景结果见results/validation_missingness.csv和results/test_missing30.csv。分类偏置不改变回归强度。模型与偏置选自验证集，所以验证指标不能作为独立泛化估计。\n',encoding='utf-8')
    files=[]
    for folder in ['src','checkpoints','results']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in ['package_verification.json','manifest.json','finish_single.py']:
                files.append(p)
    for name in ['README.md','实验报告.md','protocol.json','requirements.txt','requirements.server.lock.txt','attachment3_predictions.csv']: files.append(ROOT/name)
    hits=[]
    for p in files:
        if p.suffix in ['.md','.txt','.py','.json','.csv','.sh']:
            t=p.read_text()
            if any(x in t for x in ['inainai','huxiangyu','/mnt/disk/data/','/Users/']): hits.append(str(p.relative_to(ROOT)))
    assert not hits,hits
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    dump(manifest,ROOT/'results/manifest.json'); files.append(ROOT/'results/manifest.json')
    out=ROOT/'问题二单模型材料.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(out) as z: assert z.testzip() is None
    dump({'passed':True,'bytes':out.stat().st_size,'files':len(files),'identity_path_scan_hits':hits},ROOT/'results/package_verification.json')
    print('VERIFIED',verify,'PACKAGE_BYTES',out.stat().st_size,flush=True)

if __name__=='__main__': main()
