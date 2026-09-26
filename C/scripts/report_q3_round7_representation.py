"""Summarize actual round-seven results and archive one audited inference candidate."""
import sys, shutil, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from C.scripts import run_q3_round7_representation as context
from C.scripts.run_q3_minilm_round6 import evaluate,guard,mean_metrics,key
from C.scripts.deploy_q3_minilm_round6 import bias_prob
from C.scripts.report_q3_minilm_round6 import closure
OUT=context.OUT;PACKAGE=ROOT/'C/outputs/submission/q3_round7_representation_v1'
ASSETS=ROOT/'docs/C/experiments/assets/round7_representation'
read,write,digest=context.read,context.write,context.digest


def main():
    run=context.Run();cfg=run.cfg;sel=read(OUT/'selection.json');lock=read(OUT/'lock.json');audit=read(OUT/'audit.json')
    assert audit['passed'];grid=read(OUT/'final_grid.json');z=np.load(OUT/'final_grid.npz');g=lock['selected']
    per={};fp={};raw={}
    for group in lock['refit_groups']:
        per[group]=[];fp[group]=[];raw[group]=[];cal=read(OUT/'calibration'/f'{group}.json')
        for seed in cfg['seeds']:
            a=np.load(OUT/'deploy'/f'{group}_{seed}'/'valid.npz')
            per[group].append(evaluate(bias_prob(a['prob'],cal['class_bias']),a['reg'],a['cls'],a['score']))
            raw[group].append(read(OUT/'deploy'/f'{group}_{seed}'/'valid.json'))
            fp[group].append(read(OUT/'valid'/f'fp32_{group}_{seed}.json'))
    means={k:mean_metrics(v) for k,v in per.items()}
    eligible=g!='A0' and run.eligible(g,'A0',per,means)
    fixed_gain=key(per[g][0])[0]-key(per['A0'][0])[0]
    decision=dict(selected=g,paired_calibrated_delta=[key(a)[0]-key(b)[0] for a,b in zip(per[g],per['A0'])],
                  fixed_seed_delta=fixed_gain,mean_guard=guard(means[g],means['A0']),
                  prespecified_mean_gate_passed=bool(eligible),
                  replacement_eligible=bool(eligible and fixed_gain>=0),old_packages_overwritten=False,
                  delivery_policy='Mean gate is prespecified; conservatively retain old single-model delivery when the fixed seed loses accuracy. Do not select another seed using validation.',
                  interpretation='Internal validation with historically reused data; no test evaluation.')
    write(OUT/'deployment_decision.json',decision)
    ASSETS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    groups=list(sel['means']);acc=[[key(m)[0]*100 for m in sel['per_seed'][k]] for k in groups]
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(groups,[np.mean(v) for v in acc],yerr=[np.std(v,ddof=1) for v in acc],capsize=4,color='#4775a7')
    ax.set_ylabel('stop准确率（%）');ax.set_title('第七轮：三种子均值与标准差');fig.tight_layout();fig.savefig(ASSETS/'01_ablation.png',dpi=200);plt.close(fig)
    cm=confusion_matrix(z['cls'],z['prob'][0].argmax(-1),labels=[0,1,2]);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm/cm.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]/cm[i].sum()>.5 else 'black')
    ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set_xlabel('预测');ax.set_ylabel('真实');ax.set_title('固定种子部署模型：内部验证');fig.tight_layout();fig.savefig(ASSETS/'02_confusion.png',dpi=200);plt.close(fig)
    rows=[]
    for name in groups:
        m=sel['means'][name]['scenarios'][0]
        rows.append(f"| {name} | {m['accuracy']*100:.4f}% | {m['macro_f1']:.4f} | {m['per_class_f1'][1]:.4f} |")
    comparisons=[]
    for group in lock['refit_groups']:
        cal=read(OUT/'calibration'/f'{group}.json')
        comparisons.append(f"| {group} | {np.mean([key(m)[0] for m in fp[group]])*100:.4f}% | {np.mean([key(m)[0] for m in raw[group]])*100:.4f}% | {key(cal['oof'])[0]*100:.4f}% | {key(means[group])[0]*100:.4f}% |")
    states=[(p.stem,read(p)) for p in (OUT/'state').glob('*.json')]
    training_names={f'{phase}_{group}_{seed}' for phase in ['inner','final'] for group in cfg['groups'] for seed in cfg['seeds']}
    seconds=sum(v['seconds'] for name,v in states if name in training_names)
    epochs=sum(read(p)['epochs_run'] for phase in ['inner','final'] for p in (OUT/phase).glob('*/result.json'))
    clean=grid['metrics']['scenarios'][0]
    pr,re,f1,support=precision_recall_fscore_support(z['cls'],z['prob'][0].argmax(-1),labels=[0,1,2],zero_division=0)
    classes='\n'.join(f'| {name} | {p:.4f} | {r:.4f} | {f:.4f} | {n} |' for name,p,r,f,n in zip(['负向','中性','正向'],pr,re,f1,support))
    if decision['replacement_eligible']:
        outcome='达到本轮预设的三种子内部验证门槛，固定交付种子亦未下降，仍非独立盲测结论。'
    elif eligible:
        outcome='达到预设三种子平均指标门槛，但预固定交付种子的准确率下降。基于实际单模型交付表现保守保留旧包，不改挑其他种子；本轮包作为可复核候选保留。这一区分不是否定已通过的平均指标门槛。'
    else:
        outcome='未达到本轮预设的三种子部署指标门槛，保留旧可用包；本轮包仅为可复核候选。'
    smoke=read(OUT/'package_smoke.json') if (OUT/'package_smoke.json').exists() else None
    text=f'''# 第七轮类别对比学习与时序动态融合：实验结果

日期：2026-09-26。负责人：C。[预设方案](第七轮实验方案：类别对比学习与时序动态融合.md)。本轮已完成{audit['inner_models']}个开发模型、{audit['refit_models']}个完整train模型，共{epochs}epoch；训练及内部stop评估累计{seconds/60:.2f}分钟，不含开发、部署评估和解释耗时。

## 结果与决策

内部stop锁定 **{g}**。{outcome}

| 组别 | stop ACC均值 | Macro-F1 | 中性F1 |
|---|---:|---:|---:|
{chr(10).join(rows)}

![开发对照](assets/round7_representation/01_ablation.png)

A0为自然采样原Fusion，A1为均衡采样，A2再加监督对比；B0/B1/B2分别为时序交互的固定文本/等权/动态汇总。编码器均为在线MiniLM顶部两层微调。AB是否执行见combination.json；仅当两个分支均晋级才组合。

A2相对A1是否达到预设门槛：**{sel['contrastive_supported']}**；B2是否同时超过B0/B1并达到门槛：**{sel['dynamic_supported']}**。结构候选入选与机制证明是不同判断，不能把任意B组胜出都归因于动态选择。

## 完整训练与实际部署

| 组别 | FP32均值 | int8未校准均值 | 分组校准折外均值 | 全valid拟合偏置后均值 |
|---|---:|---:|---:|---:|
{chr(10).join(comparisons)}

固定交付seed{cfg['deployment_seed']}，最终int8在728条valid上ACC **{clean['accuracy']*100:.4f}%**，Macro-F1 **{clean['macro_f1']:.4f}**，一致性强度MAE **{clean['mae']:.4f}**，Pearson **{clean['pearson']:.4f}**。与同轮A0的配对部署ACC差值为{decision['paired_calibrated_delta']}（比例单位）。三种子均值、单种子和校准折外口径分别报告，不与第五轮test直接排名。

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
{classes}

![混淆矩阵](assets/round7_representation/02_confusion.png)

官方valid历史上反复用于研究，本轮后处理也使用valid；上述均为内部验证。本轮没有读取test。校准折外仅检验偏置步骤，不是整套研究的嵌套交叉验证。量化前后的均值差应与结构收益分开解释。

重训日志的`final_* acc 0`是未运行逐epoch验证时的打印占位，不是准确率为零；真实指标以锁定训练完成后的valid逐样本预测及本表为准。

## 第三问与复核

附件4全20条已生成极性、强度、概率、删除效应、精确Shapley、局部证据、位置映射与解释卡。音频预览{audit['previews']['audio']}条，视觉预览{audit['previews']['vision']}条；无有效位置时明确保留失败状态。附件4无标签，不计算准确率。

部署概率与固定12条valid逐样本路径、Shapley加和、训练哈希、最佳epoch与冻结权重检查通过。门控不是贡献；文本输入干预重新编码，动态模型干预后重新计算选择权重。

独立解压复验状态：{'已通过，20条预测、解释与映射精确一致。' if smoke and smoke['passed'] else '尚未完成；见后续package_smoke.json，不提前声明通过。'}

## 复现及产物

训练：`python C/scripts/run_q3_round7_representation.py --stage all`；部署：`python C/scripts/deploy_q3_round7_representation.py`；审计：`python C/scripts/check_q3_round7_representation.py`；报告：`python C/scripts/report_q3_round7_representation.py`。使用myenv及既有隔离依赖，训练需要原准备缓存和本地预训练版本。SHA256及环境见manifest和round7_sources。

日志/权重：`C/outputs/q3_round7_representation_v1/`；单模型候选包：`C/outputs/submission/q3_round7_representation_v1.zip`。完整FP32及其他种子权重保留在实验目录，不进入比赛包。独立包推理入口复用第六轮通用解释脚本，实际加载本轮模型。全队合包50MB上限仍需整体检查，不以本组件大小替代。
'''
    (ROOT/'docs/C/experiments/第七轮类别对比学习与时序动态融合实验结果.md').write_text(text,encoding='utf-8')
    evidence=[OUT/'selection.json',OUT/'lock.json',OUT/'final_grid.json',OUT/'audit.json',OUT/'deployment_decision.json',OUT/'quantization_audit.json',OUT/'preflight.json',OUT/'projector_removal_audit.json']
    for group in lock['refit_groups']: evidence.append(OUT/'calibration'/f'{group}.json')
    if smoke: evidence.append(OUT/'package_smoke.json')
    for p in evidence:
        shutil.copy2(p,ASSETS/p.name); dest=PACKAGE/'validation'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    for p in ASSETS.glob('*.png'):
        dest=PACKAGE/'plots'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    for source in closure([ROOT/'C/scripts/infer_q3_minilm_round6.py']):
        dest=PACKAGE/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    shutil.copy2(context.CFG,PACKAGE/'training_config.json');shutil.copy2(OUT/'lock.json',PACKAGE/'training_lock.json')
    (PACKAGE/'requirements.txt').write_text('numpy==1.26.4\ntorch==2.7.0\ntransformers==4.44.2\nonnxruntime==1.20.1\nscikit-learn>=1.3,<2\n',encoding='utf-8')
    (PACKAGE/'README.md').write_text(f'''# 第七轮第三问候选

模型{g}，固定种子{cfg['deployment_seed']}；内部valid ACC {clean['accuracy']*100:.4f}%。{outcome}

在解压目录安装requirements并执行：

```sh
python C/scripts/infer_q3_minilm_round6.py --package . --data-root /path/to/data --output results_new --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe
```

入口复用已验证的通用解释代码，模型由本包model.pt实际指定。data-root是包含赛题附件4的原data目录，不含复制原始数据。CPU编码采用ONNX逐样本batch1；tokenizer与权重均在包内。

results包含20条CSV/JSON、解释卡及有效音视频预览。附件4无标签；无有效视觉的样本不伪造关键帧。prob为已校准概率，raw_prob为未校准；不能重复加偏置。局部解释沿用logit字段名，实际语义是原预测类别的校准log概率变化。门控不等于贡献，映射精度需阅读mapping_status。

这是第三问组件候选，不等于全队总提交。全队50MB上限需合包后检查。训练记录见原项目C/outputs/q3_round7_representation_v1，推理不依赖训练缓存或原仓库。
''',encoding='utf-8')
    manifest={p.relative_to(PACKAGE).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write(PACKAGE/'manifest.json',manifest)
    with zipfile.ZipFile(PACKAGE.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in [*manifest,'manifest.json']:archive.write(PACKAGE/name,name)
    archive=PACKAGE.with_suffix('.zip');assert archive.stat().st_size<50_000_000
    write(OUT/'package_report.json',dict(bytes=archive.stat().st_size,sha256=digest(archive),files=len(manifest)+1,full_team_submission_checked=False))
    print('REPORT_PACKAGE',g,clean['accuracy'],decision,archive.stat().st_size,flush=True)


if __name__=='__main__':main()
