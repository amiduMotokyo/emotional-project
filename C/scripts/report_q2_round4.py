"""Archive audited round-four results and add paired, per-seed diagnostics."""
import sys
import json
import pickle
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from C.scripts.run_q2_round4 import Experiment,OUT,engine
from C.scripts.run_q2_optimization import read,write,digest
import numpy as np


def main():
    run=Experiment();assert read(OUT/'audit.json')['passed'];run.report()
    selection=read(OUT/'selection.json');test=read(OUT/'test_summary.json')
    selected=selection['selected'];baseline=selection['families']['baseline']
    curves={};seed_rows=[];classes={}
    for family,r in selection['families'].items():
        curves[family]=[]
        classes[family]=np.mean([s['scenarios'][0]['per_class_f1'] for s in r['seeds']],0).tolist()
        for i,seed in enumerate(run.cfg['seeds']):
            mid=f'{family}_{seed}';result=read(OUT/'models'/mid/'result.json');h=read(OUT/'models'/mid/'history.json')
            curves[family].append(dict(seed=seed,epoch=result['best_epoch'],best_stop=h[result['best_epoch']-1]['clean'],last_stop=h[-1]['clean']))
            seed_rows.append(dict(family=family,seed=seed,epoch=result['best_epoch'],valid_accuracy=r['seeds'][i]['scenarios'][0]['accuracy'],
                                  delta=r['per_seed_accuracy_delta'][i]))
    interval=None
    if selected!='baseline':
        z=engine.load_npz(OUT/'test/clean.npz')
        differences=np.stack([(z[f'{selected}_{s}_prob'].argmax(-1)==z['cls']).astype(float)-
                              (z[f'baseline_{s}_prob'].argmax(-1)==z['cls']).astype(float) for s in run.cfg['seeds']])
        aligned=next((ROOT/'data').glob('*/aligned_50.pkl'))
        assert digest(aligned)==read(engine.OLD/'prepared.json')['signature']['source']['aligned_pickle']
        with aligned.open('rb') as f:ids=pickle.load(f)['test']['id']
        groups=np.array([str(x).split('$_$')[0] for x in ids]);unique=np.unique(groups)
        # Average seed-specific correctness differences, not probabilities or majority votes.
        delta=differences.mean(0)
        sums=np.array([delta[groups==g].sum() for g in unique]);sizes=np.array([sum(groups==g) for g in unique])
        rng=np.random.default_rng(20263010);draws=[]
        for _ in range(2000):
            idx=rng.integers(0,len(unique),len(unique));draws.append(float(sums[idx].sum()/sizes[idx].sum()))
        interval=dict(delta=float(delta.mean()),seed_delta=differences.mean(1).tolist(),videos=len(unique),
                      percentile_95=np.quantile(draws,[.025,.975]).tolist(),replicates=2000,
                      note='Conditional on fixed fitted models and seeds; not full training/selection uncertainty, reused test.')
    contrasts={}
    for a,b in [('interaction','baseline'),('interaction','wide_control'),('factorized','baseline'),('combined','interaction'),('text_only','baseline')]:
        left,right=selection['families'][a],selection['families'][b]
        contrasts[a+'_minus_'+b]=dict(accuracy=left['mean']['scenarios'][0]['accuracy']-right['mean']['scenarios'][0]['accuracy'],
            macro_f1=left['mean']['scenarios'][0]['macro_f1']-right['mean']['scenarios'][0]['macro_f1'],neutral_f1=classes[a][1]-classes[b][1])
    test_classes={n:np.mean([s['per_class_f1'] for s in r['seeds']],0).tolist() for n,r in test['scenarios']['clean'].items()}
    analysis=dict(per_seed=seed_rows,valid_per_class_f1=classes,test_per_class_f1=test_classes,learning_curves=curves,paired_test=interval,contrasts=contrasts,
                  parameter_matching=read(OUT/'preflight.json')['parameter_matching_relative_error'])
    write(OUT/'analysis.json',analysis)
    lines=['','## 逐种子与类别诊断','','| 结构 | 种子 | stop选中epoch | valid Accuracy | 对同种子基线差值 |','|---|---:|---:|---:|---:|']
    for r in seed_rows:
        lines.append(f"| {r['family']} | {r['seed']} | {r['epoch']} | {r['valid_accuracy']:.4f} | {r['delta']*100:+.2f}个百分点 |")
    lines+=['','| 结构 | 负向F1 | 中性F1 | 正向F1 |','|---|---:|---:|---:|']
    for n,v in classes.items():lines.append(f'| {n} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |')
    lines+=['','锁定后干净test逐类F1（三种子均值）：','','| 结构 | 负向F1 | 中性F1 | 正向F1 |','|---|---:|---:|---:|']
    for n,v in test_classes.items():lines.append(f'| {n} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |')
    lines+=['','## 结构差异（valid，三个独立模型均值之差）','','| 对比 | Accuracy差值 | 宏F1差值 | 中性F1差值 |','|---|---:|---:|---:|']
    for name,r in contrasts.items():
        lines.append(f"| {name} | {r['accuracy']*100:+.2f}个百分点 | {r['macro_f1']:+.4f} | {r['neutral_f1']:+.4f} |")
    lines+=['','## 解释范围','',
            f"参数量匹配误差为{analysis['parameter_matching']:.2%}。预检查确认交互结构对音频帧置换有响应，但这种机制检查不等于准确率提升证据。",'',
            '本轮所有结构使用相同训练配方；结果检验的是固定配方下的结构差异，不保证每种结构均达到各自最优超参数。分解头保留独立强度回归，没有硬级联。',
            '容量匹配基线不包含新增位置编码，因此本轮不能单独区分位置编码、跨注意力、残差路径等组件的贡献；交互模型若胜出，首先支持的是这一整体结构。',
            '三个种子的SD用于描述波动，不是置信区间；同种子不同结构也不代表相同参数初始化，只共享数据、视图和种子编号。','']
    if interval:
        lo,hi=interval['percentile_95']
        lines += [f"锁定结构相对基线的干净test准确率均值变化为{interval['delta']*100:+.2f}个百分点。按视频分组配对bootstrap 2000次的95%区间为[{lo*100:+.2f}, {hi*100:+.2f}]个百分点；三种子均在每次抽样中保留，区间仅刻画固定模型的测试抽样波动，不包含训练与选型不确定性。",'']
        if lo<=0<=hi:lines += ['该区间包含0，不能据此声称测试提升具有稳定统计证据。','']
    else:
        lines += ['本轮最终保留基线，未获得满足预设约束的结构改进；不将基线与自身的零差异当作效果证据。','']
    if selected=='wide_control':
        lines += ['## 本轮结论','',
                  '验证规则选中的是容量对照，而非提出的序列交互结构。容量对照将原Fusion隐藏维度从64增加到91，参数从74963增至102665；交互结构参数为102856。三种子验证准确率均高于对应原基线，但尚不能据此断言更大的模型普遍更好。',
                  '文本单模态的内部验证均值也高于本轮基线，说明值得继续检查音视频分支是否有效利用了信息；这不等于证明音视频信息天然无用。序列交互在该训练配方下未超过原基线或容量对照，未支持“池化前交互即可突破瓶颈”的假设。',
                  '容量对照与文本单模态的验证准确率均值仅相差约0.046个百分点（跨三种子的正确预测总数相差1），因此本轮不能断言容量对照稳定优于文本单模态；选择只遵循预先固定的最高均值规则。文本单模态未进入本轮锁定测试面板。',
                  '这些成绩是三个独立单模型的指标均值，与第三轮三模型等权集成的61.54%验证准确率不是同一口径，不作直接进步幅度比较。','']
    path=ROOT/'docs/C/experiments/第四轮结构优化实验结果.md'
    path.write_text(path.read_text(encoding='utf-8')+'\n'.join(lines),encoding='utf-8')
    archive=ROOT/'docs/C/experiments/assets/round4';archive.mkdir(parents=True,exist_ok=True)
    names=['manifest.json','implementation.json','preflight.json','selection.json','lock.json','test_summary.json','analysis.json','audit.json']
    for name in names:shutil.copy2(OUT/name,archive/name)
    write(archive/'source_manifest.json',{n:digest(OUT/n) for n in names})
    print('Report archived; selected:',selected,'paired interval:',interval)


if __name__=='__main__':main()
