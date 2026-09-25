"""Archive completed round-five metrics and diagnostic conclusions, without fitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from C.scripts.run_q2_optimization import read,write,digest
import numpy as np
OUT=ROOT/'C/outputs/q2_round5_experts_v1'

def main():
    audit=read(OUT/'audit.json');assert audit['passed']
    diagnosis=read(OUT/'diagnosis.json');selection=read(OUT/'selection.json');test=read(OUT/'test_summary.json');selected=selection['selected']
    lines=['# 第五轮独立专家与决策融合实验结果','','状态：独立专家、互补诊断、按视频分组折外训练、融合选型及64条件锁定复评完成，审计通过。',
        '',f'最终选择：**{selected}**。三个种子分别计算完整预测流程的指标后取均值，不跨种子集成。历史test仅为锁定后复评。',
        '','## 1 错误互补诊断（stop）','','| 种子 | 文本准确率 | 音视频准确率 | 文本错/音视频对 | 文本对/音视频错 | 共同错误 | oracle上限 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in diagnosis['rows']:
        s=row['scenarios'][0]
        lines.append(f"| {row['seed']} | {s['text_accuracy']:.2%} | {s['av_accuracy']:.2%} | {s['rescued']:.2%} | {s['harmed']:.2%} | {s['both_wrong']:.2%} | {s['oracle']:.2%} |")
    lines+=['',f"干净stop可纠正比例三种子平均为{diagnosis['mean_clean_rescue']:.2%}，达到预定分支规则，进入决策融合。oracle使用真实标签识别谁预测正确，仅为诊断上限，不能部署。stop用于专家epoch选择，诊断不是独立验证。",'',
        '## 2 内部valid选型','','| 候选 | 干净Accuracy±样本SD | 干净宏F1 | 缺失平均宏F1 | 四条件平均MAE | 合格 |','|---|---:|---:|---:|---:|---|']
    for n,v in selection['candidates'].items():
        s=v['mean']['scenarios'];sd=np.std([x['scenarios'][0]['accuracy'] for x in v['seeds']],ddof=1)
        lines.append(f"| {n} | {s[0]['accuracy']:.2%} ± {sd:.2%} | {s[0]['macro_f1']:.4f} | {np.mean([x['macro_f1'] for x in s[1:]]):.4f} | {np.mean([x['mae'] for x in s]):.4f} | {v['eligible']} |")
    lines+=['','宏F1/MAE约束相对第四轮加宽基线；候选按合格集合中的干净准确率均值选型。SD不是置信区间。','',
        '## 3 锁定后64条件复评','','| 方法 | 干净准确率±SD | 干净宏F1 | 干净MAE | 64条件平均准确率 | 64条件平均宏F1 | 64条件平均MAE |','|---|---:|---:|---:|---:|---:|---:|']
    for n,a in test['averages'].items():
        r=test['scenarios']['clean'][n];s=r['mean'];sd=np.std([v['accuracy'] for v in r['seeds']],ddof=1)
        lines.append(f"| {n} | {s['accuracy']:.2%} ± {sd:.2%} | {s['macro_f1']:.4f} | {s['mae']:.4f} | {a['accuracy']:.2%} | {a['macro_f1']:.4f} | {a['mae']:.4f} |")
    baseline=selection['candidates']['wide_control'];gate=selection['candidates']['learned_gate'];eq=selection['candidates']['equal_mix']
    delta=(gate['mean']['scenarios'][0]['accuracy']-baseline['mean']['scenarios'][0]['accuracy'])*100
    analysis=dict(selected=selected,gate_valid_delta_pp=delta,gate_vs_equal_pp=(gate['mean']['scenarios'][0]['accuracy']-eq['mean']['scenarios'][0]['accuracy'])*100,valid_weights={})
    analysis['gate_valid_seed_delta_pp']=[(a['scenarios'][0]['accuracy']-b['scenarios'][0]['accuracy'])*100 for a,b in zip(gate['seeds'],baseline['seeds'])]
    for seed in [20263001,20263002,20263003]:
        z=np.load(OUT/'valid_predictions'/f'{seed}.npz');analysis['valid_weights'][str(seed)]=z['gate_weights'].mean(1).tolist()
    if selected!='wide_control':
        clean=test['scenarios']['clean'];analysis['test_seed_delta_pp']=[(a['accuracy']-b['accuracy'])*100 for a,b in zip(clean[selected]['seeds'],clean['wide_control']['seeds'])]
        analysis['test_clean_delta_pp']=(clean[selected]['mean']['accuracy']-clean['wide_control']['mean']['accuracy'])*100
        analysis['scenario_wins_ties_losses']=[sum((v[selected]['mean']['accuracy']-v['wide_control']['mean']['accuracy']>1e-12) for v in test['scenarios'].values()),sum(abs(v[selected]['mean']['accuracy']-v['wide_control']['mean']['accuracy'])<=1e-12 for v in test['scenarios'].values()),sum(v[selected]['mean']['accuracy']-v['wide_control']['mean']['accuracy']< -1e-12 for v in test['scenarios'].values())]
        lines+=['',f"锁定候选相对加宽基线的干净test准确率变化为{analysis['test_clean_delta_pp']:+.2f}个百分点，逐种子变化为"+'、'.join(f'{x:+.2f}' for x in analysis['test_seed_delta_pp'])+'个百分点。',f"64条件准确率胜/平/负：{analysis['scenario_wins_ties_losses']}。条件之间共享样本且高度相关，不能作为64次独立实验进行显著性解释。"]
    else:lines+=['','没有新候选按预定规则超过并替代加宽基线，本轮保留第四轮模型。64条件沿用已核验的基线预测，不重复模型推理。']
    lines+=['','| 干净test逐类F1（三种子均值） | 负向 | 中性 | 正向 |','|---|---:|---:|---:|']
    for n,v in test['scenarios']['clean'].items():
        c=np.mean([x['per_class_f1'] for x in v['seeds']],axis=0)
        lines.append(f'| {n} | {c[0]:.4f} | {c[1]:.4f} | {c[2]:.4f} |')
    lines+=['','## 4 结论与局限','',f'学习门控相对加宽基线的valid准确率变化为{delta:+.2f}个百分点；相对简单融合变化为{analysis["gate_vs_equal_pp"]:+.2f}个百分点。是否部署由选型约束决定，不能只看oracle上限或单项准确率。',
        '', 'valid相对基线的逐种子差值为'+'、'.join(f'{x:+.2f}' for x in analysis['gate_valid_seed_delta_pp'])+'个百分点。平均值获选不能替代逐种子一致性检查。',
        '','文本与音视频专家有互补错误不等于融合器一定能识别这些样本。折外训练防止专家在自身训练样本上产生过于乐观的融合输入，但仍存在折内专家与完整fit专家分布差异、三个种子的有限覆盖及历史valid/test复用等局限。',
        '','本轮按预定规则进入了融合分支，未触发互补不足时的更换文本表示分支；没有运行新编码器实验，也不将其写成已完成。未增加外部情感训练数据，未用专项附件3/4调参，未修改正式提交模型。']
    states={p.stem:read(p)['seconds'] for p in (OUT/'state').glob('*.json')}
    timings=dict(full_av_seconds=sum(v for n,v in states.items() if n.startswith('audio_visual_')),fold_expert_seconds=sum(v for n,v in states.items() if n.startswith('fold')),gate_seconds=sum(v for n,v in states.items() if n.startswith('gate_')),selection_seconds=states['select'],evaluation_seconds=states['evaluate'])
    analysis['timing']=timings;write(OUT/'analysis.json',analysis)
    lines+=['','## 5 复现与审计','',f"新增完整fit音视频专家3个、折内专家18个，共378个expert epoch；学习门控3个，每个200步。复用第四轮文本与加宽基线权重。完整fit专家训练与stop评估{timings['full_av_seconds']/60:.2f}分钟；折内专家训练、stop及折外预测{timings['fold_expert_seconds']/60:.2f}分钟；门控{timings['gate_seconds']:.2f}秒；选型{timings['selection_seconds']:.2f}秒；64条件复评{timings['evaluation_seconds']/60:.2f}分钟。耗时不含开发，阶段有包含关系，不能将所有state时长直接相加。",'',
        '方案：[第五轮固定方案](第五轮独立专家与决策融合实验方案.md)。入口：`C/scripts/run_q2_round5.py --stage all`；审计：`C/scripts/check_q2_round5.py --audit`；生成本报告：`C/scripts/report_q2_round5.py`。原始产物：`C/outputs/q2_round5_experts_v1/`。小型指标和哈希归档：[assets/round5](assets/round5/README.md)。','']
    (ROOT/'docs/C/experiments/第五轮独立专家与决策融合实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    archive=ROOT/'docs/C/experiments/assets/round5';archive.mkdir(parents=True,exist_ok=True)
    names=['manifest.json','implementation.json','preflight.json','diagnosis.json','selection.json','lock.json','test_summary.json','analysis.json','audit.json']
    if (OUT/'resume_check.json').exists():names.append('resume_check.json')
    for name in names:(archive/name).write_bytes((OUT/name).read_bytes())
    write(archive/'source_manifest.json',{name:digest(OUT/name) for name in names})
    (archive/'README.md').write_text('# 第五轮实验归档\n\n保存实际指标、运行签名、锁定与审计结果，详见[结果报告](../../第五轮独立专家与决策融合实验结果.md)。权重、折外预测与训练日志保留在 `C/outputs/q2_round5_experts_v1/`。\n',encoding='utf-8')
    print(analysis)

if __name__=='__main__':main()
