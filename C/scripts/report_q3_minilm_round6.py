"""Create third-question figures, experiment report, and a portable inference archive."""
import sys,ast,json,shutil,zipfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'C/outputs/q2_round3_env'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from C.scripts.run_q3_minilm_round6 import OUT,read,write,CFG,SPLIT,digest
PACKAGE=ROOT/'C/outputs/submission/q3_round6_minilm_v2'
DOC=ROOT/'docs/C/experiments'
ASSETS=DOC/'assets/round6_minilm'

def closure(entries):
    found=set();pending=list(entries)
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path);tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:
                if node.level:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    candidate=base.joinpath(*node.module.split('.')).with_suffix('.py')
                    if candidate.exists():pending.append(candidate)
                else:names=[node.module]
            for name in names:
                if name.split('.')[0] in ['A','B','C']:
                    candidate=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                    if candidate.exists():pending.append(candidate)
    return found

def main():
    ASSETS.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    sel=read(OUT/'selection.json');lock=read(OUT/'lock.json');grid=read(OUT/'final_grid.json');audit=read(OUT/'audit.json');cfg=read(CFG);records=read(PACKAGE/'results/attachment4_explanations.json');z=np.load(OUT/'final_grid.npz');cal=read(OUT/'calibration'/f"{lock['selected']}.json")
    def save(name):
        plt.tight_layout();plt.savefig(ASSETS/f'{name}.png',dpi=220,bbox_inches='tight');plt.savefig(ASSETS/f'{name}.svg',bbox_inches='tight');plt.close()
    names=list(sel['means']);values=[[m['scenarios'][0]['accuracy']*100 for m in sel['per_seed'][g]] for g in names]
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) for v in values],color=['#4775a7','#57a894','#d9954f','#999999'],capsize=4);ax.set_ylim(55,70);ax.set_ylabel('stop准确率（%）');ax.set_title('配对开发实验：三种子均值与样本标准差');save('01_ablation')
    cm=confusion_matrix(z['cls'],z['prob'][0].argmax(-1),labels=[0,1,2]);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{cm[i,j]/cm[i].sum():.1%}',ha='center',va='center',color='white' if cm[i,j]/cm[i].sum()>.55 else 'black')
    ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set_xlabel('预测');ax.set_ylabel('真实');ax.set_title('最终int8模型：728条内部验证');save('02_confusion')
    fig,ax=plt.subplots(figsize=(9,4));arr=np.array([[r['shapley']['signed'][m] for m in ['text','audio','vision']] for r in records]);im=ax.imshow(arr.T,aspect='auto',cmap='RdBu_r',vmin=-max(abs(arr).max(),1e-6),vmax=max(abs(arr).max(),1e-6));ax.set_yticks(range(3),['文本','音频','视觉']);ax.set_xticks(range(20),[r['sample_id'] for r in records],rotation=60);ax.set_title('附件4：固定预测类别log概率的模态Shapley');fig.colorbar(im,ax=ax,label='有符号贡献');save('03_shapley')
    fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True);example=records[0]
    for ax,m,label in zip(axes,['text','audio','vision'],['文本','音频','视觉']):
        yy=[np.nan if v is None else v for v in example['local'][m]['signed_logit_drop_by_position']];ax.bar(range(50),yy,color='#4775a7');ax.axhline(0,color='black',linewidth=.5);ax.set_ylabel(label)
    axes[0].set_title(f"附件4样本{example['sample_id']}：单位置干预的目标类log概率下降");axes[-1].set_xlabel('原对齐位置（从0开始）');save('04_local')
    fig,ax=plt.subplots(figsize=(7,4))
    for mode,color in zip(['text','audio','vision'],['#4775a7','#57a894','#d9954f']):
        vals=[grid['metrics']['scenarios'][0]['accuracy']]
        for rate in [.1,.3,.5]:vals.append(np.mean([m['accuracy'] for c,m in zip(grid['conditions'],grid['metrics']['scenarios']) if c[0]==mode and c[1]==rate]))
        ax.plot([0,10,30,50],np.array(vals)*100,'o-',label=mode,color=color)
    ax.set_xlabel('连续缺失比例（%，非零点对三种位置平均）');ax.set_ylabel('内部验证准确率（%）');ax.legend();save('05_missing')
    clean=grid['metrics']['scenarios'][0];missing=grid['metrics']['scenarios'][1:];g=lock['selected'];seed=cfg['deployment_seed'];final=read(OUT/'final'/f'{g}_{seed}'/'result.json')
    rows=[]
    for n in names:
        m=sel['means'][n]['scenarios'][0];rows.append(f"| {n} | {100*m['accuracy']:.2f}% | {m['macro_f1']:.4f} | {m['per_class_f1'][1]:.4f} |")
    states=[read(p) for p in (OUT/'state').glob('*.json')];train_seconds=sum(r['seconds'] for p,r in zip((OUT/'state').glob('*.json'),states) if p.stem.startswith(('inner_','final_')))
    fp32=[];int8=[]
    for s in cfg['seeds']:
        fp32.append(read(OUT/'valid'/f'fp32_{g}_{s}.json')['scenarios'][0]);int8.append(read(OUT/'deploy'/f'{g}_{s}'/'valid.json')['scenarios'][0])
    prec,rec,f1,support=precision_recall_fscore_support(z['cls'],z['prob'][0].argmax(-1),labels=[0,1,2],zero_division=0)
    perclass='\n'.join(f'| {label} | {p:.4f} | {r:.4f} | {f:.4f} | {int(n)} |' for label,p,r,f,n in zip(['负向','中性','正向'],prec,rec,f1,support))
    text=f'''# 第六轮：端到端MiniLM训练与第三问交付结果

状态：训练、量化校准、附件4预测解释及审计已完成。依据正确队友归档226b84f和[重设计方案](第六轮重设计：端到端MiniLM与第三问解释模型.md)，未沿用已撤回v4的七模型路线。本文为实验记录，不是第五轮论文的改写。

## 结果与选择

四组各三个种子完成120个开发epoch。相对R0，加宽R1和全编码器更新R2未达到预设稳定改进门槛，最终锁定 **{g}**，不是宣称扩大容量成功。随后用完整3395条train重训三个种子，epoch分别为{list(lock['epochs'][g].values())}；共计{120+sum(lock['epochs'][g].values())}个训练epoch。训练与其内部stop评估的阶段用时约{train_seconds/60:.2f}分钟，不含开发、量化、解释和审计。

| 组别 | stop ACC（三种子均值） | Macro-F1 | 中性F1 |
|---|---:|---:|---:|
{chr(10).join(rows)}

![开发对照](assets/round6_minilm/01_ablation.png)

R0为顶部两层微调＋dim64；R1为顶部两层微调＋dim192；R2为六层及嵌入微调＋dim192；R3为冻结文本＋dim192。R2确实参与了训练。最终模型含{final['parameters']:,}参数，其中{final['trainable_parameters']:,}参数参与训练；并非只优化旧冻结特征任务头。所有冻结参数在训练末保持不变。

## 完整训练后的精度与校准

| 口径 | 干净验证ACC |
|---|---:|
| FP32，三个完整train模型均值 | {100*np.mean([x['accuracy'] for x in fp32]):.4f}% |
| int8未校准，三个模型均值 | {100*np.mean([x['accuracy'] for x in int8]):.4f}% |
| int8五折分组校准折外，三个模型均值（候选校准检验） | {100*cal['oof']['scenarios'][0]['accuracy']:.4f}% |
| 最终采用偏置，完整valid拟合后三个模型均值 | {100*cal['full_fit']['scenarios'][0]['accuracy']:.4f}% |
| 预先固定交付seed{seed}，最终int8 | {100*clean['accuracy']:.4f}% |

最终偏置为`{cal['class_bias']}`；校准是否获准采用：{cal['used']}。种子{seed}在验证前已固定，未挑选验证集最高种子。FP32、int8、折外校准及全valid拟合是不同口径，不能混写成同一个成绩。官方valid历史上反复用于研究，本轮也用于后处理选择；全部称内部验证，未读取test用于本轮评估或选型。

交付单模型在728条valid上：ACC **{100*clean['accuracy']:.2f}%**（{int(np.trace(cm))}/728），Macro-F1 **{clean['macro_f1']:.4f}**，最终一致性强度MAE **{clean['mae']:.4f}**，Pearson **{clean['pearson']:.4f}**。63种缺失条件等权平均ACC为{100*np.mean([m['accuracy'] for m in missing]):.2f}%，Macro-F1为{np.mean([m['macro_f1'] for m in missing]):.4f}。强度同时保存原始回归值与按最终极性映射的服务值。

正确队友的历史部署单模型为63.74%、Macro-F1 0.6001、MAE 0.5961；第五轮65.61%属于历史test三种子均值。本轮不与第五轮test直接比较，不将跨协议差异归为单一因素。结构更大没有稳定胜出，本次可交付性与准确率提升必须分别评价。

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
{perclass}

![验证混淆矩阵](assets/round6_minilm/02_confusion.png)
![缺失评估](assets/round6_minilm/05_missing.png)

## 第三问产物与解释边界

附件4全20条已输出极性、强度、原始与校准概率、模态删除效应、精确三模态Shapley、主要参考模态、局部证据、原文/音频时段/关键帧定位与解释卡。无标签，不能计算准确率。审计记录的有效音频预览{audit['previews']['audio']}条、视觉预览{audit['previews']['vision']}条；映射失败或名义采样率近似状态保留在JSON，不能假称全为精确时间。

解释实际运行最终int8模型，文本UNK替换后重新编码，音视频置零并同步掩码，固定原预测类别的校准log概率。门控仅作诊断；有符号删除效应与Shapley分列，绝对占比并非因果百分比。Shapley八个子集加和通过检查；空集包含模型先验。没有正支持时允许主要模态未确定。

![模态作用](assets/round6_minilm/03_shapley.png)
![局部证据](assets/round6_minilm/04_local.png)

对valid每类前4条共12条固定样本复核部署推理与保存概率，并比较关键窗口与同宽随机窗口的干预变化；这是模型敏感性诊断，不是人工证据正确率。详细检查见产物`audit.json`。

## 复现与交付

训练：`python C/scripts/run_q3_minilm_round6.py --stage all`；导出与校准：`python C/scripts/deploy_q3_minilm_round6.py`。这两个命令在原项目环境及已核对的历史prepared缓存/分组文件下运行，日志、权重、manifest、锁定信息和逐样本预测在`C/outputs/q3_round6_minilm_v2/`。源码改变时断点恢复会拒绝混用旧状态。

比赛组件在`C/outputs/submission/q3_round6_minilm_v2/`及同名zip。内含独立推理代码、模型、tokenizer、标准化、偏置、结果、解释卡、图表及SHA256清单。解压后按README运行，无需历史特征缓存或训练checkpoint；从头训练则仍需原项目缓存和通用预训练权重。没有覆盖全队原提交包；全队合包后的50MB上限需另行核对。
'''
    if (OUT/'package_smoke.json').exists():
        smoke=read(OUT/'package_smoke.json');assert smoke['passed']
        text+='\n独立解压复验已通过：在不引用仓库实现的独立目录重新运行全部20条附件4样本，预测、解释及定位JSON与原结果完全一致。依赖使用已验证的隔离库目录，详见`package_smoke.json`。13号样本原始视觉无有效位置，故19张视觉预览是如实交付，不是遗漏。\n'
    (DOC/'第六轮端到端MiniLM训练与第三问交付结果.md').write_text(text,encoding='utf-8')
    # Persist compact figures and numerical evidence alongside the readable report.
    for source in [OUT/'selection.json',OUT/'final_grid.json',OUT/'audit.json',OUT/'calibration'/f'{g}.json']:
        shutil.copy2(source,ASSETS/source.name)
    plots=PACKAGE/'plots';plots.mkdir(exist_ok=True)
    for p in ASSETS.glob('*.png'):shutil.copy2(p,plots/p.name)
    entries=[ROOT/'C/scripts/infer_q3_minilm_round6.py']
    for source in closure(entries):
        dest=PACKAGE/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    validation=PACKAGE/'validation';validation.mkdir(exist_ok=True)
    for source in [OUT/'final_grid.json',OUT/'audit.json',OUT/'selection.json',OUT/'calibration'/f'{g}.json']:
        shutil.copy2(source,validation/source.name)
    if (OUT/'package_smoke.json').exists():
        shutil.copy2(OUT/'package_smoke.json',validation/'package_smoke.json')
        shutil.copy2(OUT/'package_smoke.json',ASSETS/'package_smoke.json')
    shutil.copy2(CFG,PACKAGE/'training_config.json');shutil.copy2(OUT/'lock.json',PACKAGE/'training_lock.json')
    (PACKAGE/'requirements.txt').write_text('numpy==1.26.4\ntorch==2.7.0\ntransformers==4.44.2\nonnxruntime==1.20.1\nscikit-learn>=1.3,<2\n',encoding='utf-8')
    (PACKAGE/'README.md').write_text(f'''# 第三问：MiniLM端到端训练候选

模型：{g}，固定seed{seed}，epoch{lock['epochs'][g][str(seed)]}。内部验证ACC {100*clean['accuracy']:.2f}%，Macro-F1 {clean['macro_f1']:.4f}；附件4无标签。

在解压目录安装requirements，然后执行：

```sh
python C/scripts/infer_q3_minilm_round6.py --package . --data-root /path/to/data --output results_new --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe
```

data-root包含原赛题附件4目录，不包含复制的原始数据。ffmpeg/ffprobe为可选系统工具；提供后生成媒体预览并核验视频时长。推理在CPU执行，ONNX逐样本batch1，不能任意合批替代。无需下载编码器，包内包含int8权重与tokenizer。

results包含20条CSV、详细JSON、cards、audio、frames；plots包含验证与解释图。prob列为校准后概率，raw_prob列为校准前概率，不得再次加偏置。logit命名沿用共享接口，在此候选的解释字段中表示固定原类别log概率。Shapley与单路删除效应不是同一分解。原始/最终强度均保留。

音视频证据时间可能依赖名义采样率及多匹配，须阅读mapping_status和详细证据；门控不是模态贡献，解释不等于人类情感因果。模型全局先验也影响空集输出。

validation包含内部验证与审计。该组件不等于全队最终提交包，不含训练用原始数据与FP32 checkpoint。训练源和完整日志保存在原项目C/scripts及C/outputs/q3_round6_minilm_v2。训练协议见training_config.json和training_lock.json。
''',encoding='utf-8')
    manifest={p.relative_to(PACKAGE).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write(PACKAGE/'manifest.json',manifest)
    archive=PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as f:
        for relative in [*manifest,'manifest.json']:f.write(PACKAGE/relative,relative)
    assert archive.stat().st_size<50_000_000
    write(OUT/'package_report.json',dict(bytes=archive.stat().st_size,sha256=digest(archive),files=len(manifest)+1,uncompressed_bytes=sum(x['bytes'] for x in manifest.values()),full_team_submission_checked=False))
    print('REPORT_PACKAGE',archive.stat().st_size,clean,flush=True)

if __name__=='__main__':main()
