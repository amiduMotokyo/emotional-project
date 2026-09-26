"""Add evidence tables from saved results only; no training or model inference."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/论文稿/问题二.md'
ASSETS=ROOT/'docs/C/paper/assets/q2_integrated'

def supplement():
    selection=ROOT/'C/outputs/q3_round7_representation_v1/selection.json'
    metrics=json.loads(selection.read_text(encoding='utf-8'))['per_seed']
    names={'A0':'自然采样、原Fusion','A1':'均衡采样、原Fusion','A2':'均衡采样＋对比学习','B0':'时序交互、固定文本','B1':'时序交互、等权分支','B2':'时序交互、动态分支'}
    lines=['| 方法 | dev ACC（%） | dev Macro-F1 | dev MAE | dev Pearson | 缺失Macro-F1 |','|---|---:|---:|---:|---:|---:|']
    raw=[]
    for group,name in names.items():
        runs=metrics[group];assert len(runs)==3
        clean=[r['scenarios'][0] for r in runs]
        assert all(r['n']==391 for r in clean)
        acc=np.array([r['accuracy']*100 for r in clean])
        vals=[float(np.mean([r[k] for r in clean])) for k in ['macro_f1','mae','pearson']]
        missing=float(np.mean([r['macro_f1'] for run in runs for r in run['scenarios'][1:]]))
        lines.append(f'| {group} {name} | {acc.mean():.2f}±{acc.std(ddof=1):.2f} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {missing:.4f} |')
        raw.append(dict(group=group,accuracy_percent_mean=float(acc.mean()),accuracy_percent_sd=float(acc.std(ddof=1)),macro_f1=vals[0],mae=vals[1],pearson=vals[2],missing_macro_f1=missing))
    gridfile=ROOT/'C/outputs/q2_accuracy_screen_20260925/minilm_finetune/validation_grid_full_calibrated_seed20260925.json'
    grid=json.loads(gridfile.read_text(encoding='utf-8'))['metrics'];rows=[r for r in grid if r['condition']['mode']!='clean'];assert len(rows)==63
    key=lambda r:(r['condition']['mode'],r['condition']['rate'],r['condition']['position'])
    assert len({key(r) for r in rows})==63 and all(r['n']==728 for r in rows)
    csvpath=SOURCE.parent/'问题二_最终基线63种缺失条件完整结果.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        fields=['mode','nominal_missing_rate','position','n','accuracy','macro_f1','mae','pearson','negative_f1','neutral_f1','positive_f1','observed_missing_text','observed_missing_audio','observed_missing_vision']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            o=dict(mode=r['condition']['mode'],nominal_missing_rate=r['condition']['rate'],position=r['condition']['position'],n=r['n'],**{k:r[k] for k in ['accuracy','macro_f1','mae','pearson']})
            o.update(dict(zip(['negative_f1','neutral_f1','positive_f1'],r['per_class_f1'])))
            o.update({'observed_missing_'+m:r['observed_missing_rate_by_modality'][m] for m in ['text','audio','vision']});writer.writerow(o)
    tables=[]
    for part,rate in enumerate([.1,.3,.5],1):
        lines2=[f'表5-28（{part}/3）最终模型缺失鲁棒性敏感性分析：{rate:.0%}缺失','', '| 缺失模态 | 位置 | ACC（%） | Macro-F1 | MAE | Pearson |','|---|---|---:|---:|---:|---:|']
        for r in rows:
            if r['condition']['rate']!=rate:continue
            mode=r['mode'].replace('text','文本').replace('audio','音频').replace('vision','视觉')
            pos={'start':'起始','middle':'中部','end':'末尾'}[r['condition']['position']]
            lines2.append(f"| {mode} | {pos} | {100*r['accuracy']:.2f} | {r['macro_f1']:.4f} | {r['mae']:.4f} | {r['pearson']:.4f} |")
        assert len(lines2)==25
        tables.append('\n'.join(lines2))
    section='''## 5.3.9 评价协议、模块消融与缺失鲁棒性敏感性分析

### 1 数据划分及所有结果的解释口径

本文将train内部按原视频划出的391条stop统一称为内部开发集（internal dev）；official valid专指附件2给定的728条验证样本；历史test专指已在早期实验中评估过的727条测试样本。内部dev属于train的子集，不是另一份官方验证集。表5-25给出各组结果的对应关系，以避免将不同划分、不同统计单位或不同精度下的分数直接排名。

表5-25 数据划分、使用方式与正文结果对应

| 数据／统计口径 | 样本数 | 使用方式 | 对应结果 |
|---|---:|---|---|
| fit／内部dev | 3004／391 | 按视频分组；开发结构与训练轮数 | 表5-19、表5-26；66.33%属于dev |
| official valid，冻结F₀ | 728 | 早期选模、鲁棒性及误差检查 | 表5-16、图5-15至图5-16；61.68%为单模型 |
| official valid，早期优化 | 728 | 各轮开发与结果比较 | 表5-17第三轮、表5-18的valid列；注意集成与均值之别 |
| official valid，在线候选 | 728 | 完整train重训后的报告及偏置校准 | 表5-20；61.3095%为FP32均值，62.50%为固定单模型 |
| official valid，最终B₀ | 728 | epoch与模型选择、偏置拟合和报告 | 表5-23、表5-28；63.7363%为int8校准后单模型 |
| 历史test | 727 | 早期已执行的测试结果，仅作历史记录 | 表5-17第二轮、表5-18的test列；65.61%为三种子均值 |
| 附件3专项 | 30，无标签 | 最终模型推理 | 表5-24；不计算ACC、F1、MAE或Pearson |

第二轮的内部折外选择分数属于开发阶段的模型组合评价，不是official valid的整集准确率；第三轮的61.54%是概率集成的official valid结果，不是三个模型准确率的算术平均。第六、七轮的“分组折外校准”也不改变数据来自official valid这一事实，只表示偏置拟合时留出相应折。

### 2 统一协议下的模块消融

表5-26使用第七轮已完成的六组受控实验，不拼接前后轮次的最优分数。共同条件为fit3004／dev391的视频分组划分、三个配对种子20267001—20267003、顶部两层MiniLM微调、batch32、10轮开发预算、相同学习率配置、0.3连续缺失概率及加权CE＋0.8SmoothL1。每个种子均在同一dev上按干净ACC选择epoch，平分时比较Macro-F1及一致性MAE；因此该表是开发选模结果，不是独立留出评价。

表5-26 统一协议模块消融：internal dev，FP32，三种子统计

'''+ '\n'.join(lines)+'''

ACC以均值±样本标准差表示，其余列为三种子均值；MAE对应一致性强度。缺失Macro-F1仅对文本、音频、视觉分别30%中部连续缺失三个开发条件平均，再对种子平均，不能等同于表5-28的63条件平均。

A1相对A0只改变采样方案，并保留原类别权重；A2相对A1增加训练期对比投影头及对比目标，用于检验这一完整监督分支的增量。B0、B1、B2采用相同交叉注意力主干，分别固定文本分支、等权汇总和动态汇总，构成分支选择机制的消融；这里的B0是第七轮实验编号，与最终模型符号B₀不同。A0到B0同时引入位置／模态标识及交互主干，属于结构整体替换，不能将其差值只归因于某一个注意力操作。

该表支持均衡与当前对比配置没有改善dev，且时序结构在dev具有局部收益；结合完整重训后FP32均值持平及固定部署种子未改善的结果，尚不足以证明动态分支选择能够稳定替换最终基线。它也不是最终B₀逐项关闭全部模块后的消融，未完成的最终模型消融不补填数值。

### 3 验证集复用与调参风险

official valid确实被多次使用。最终B₀的三个种子分别在该集选择最佳epoch，随后比较模型并搜索25种共享类别偏置；同一批标签既参与选择又用于报告63.7363%，存在选择乐观偏差。多轮结构和训练策略研究还使研究过程对验证反馈产生适应，不能因为某一轮没有用valid选择结构，就将整个研究的valid结果解释为未触及的测试结果。

五折偏置校准使每条折外预测所用偏置不直接拟合该条标签，但编码器和epoch已经受到同一official valid的上游选择影响。因此，折外校准仅评价后处理步骤，不是对“训练—选模—校准”整个流程的嵌套交叉验证；全valid拟合后的分数更应明确标为拟合后内部验证值。对同一验证集施加63种缺失也没有增加独立样本量，不能据此缩小泛化不确定性。

历史test已在前期多轮结果中被查看，本文只引用其当时输出，不将其重新定义为最终独立盲测，也不利用其分数重新排序最终模型。现有分组划分、配对种子及实际推理路径检查降低了部分混杂因素，但不能消除反复调参风险。严格确认最终泛化需要在模型、偏置与分析方案全部冻结后使用此前未接触的独立带标签样本；本研究尚无这样的最终评估，因而只作内部验证结论，不宣称已获得无偏泛化准确率或统计显著优势。

### 4 附件3 CSV与表5-24的一一对应

表5-24对应正文同目录的文件“问题二_最终第三问入口模型_附件3预测.csv”，采用B₀的seed20260925、第3轮检查点、int8文本编码器与偏置[0.2,0,0]。该文件有30条数据行、12个字段；sample_id为唯一关联键。表格为节约篇幅采用左右两栏，左栏01—15、右栏16—30，不能按整行横向读取顺序改变样本身份。正文只显示sample_id、polarity和intensity三个字段，其中类别翻译为中文，强度统一显示六位小数。

表5-27 附件3 CSV字段及与正文的对应关系

| CSV字段 | 含义及数值口径 | 表5-24对应 |
|---|---|---|
| sample_id | 原附件3对齐pkl文件名去掉扩展名；30个唯一ID | 样本编号，原值保留 |
| polarity | Negative／Neutral／Positive，取校准概率最大类 | 负向／中性／正向 |
| intensity | 极性一致性映射后的强度，范围[-3,3] | 强度，六位小数 |
| prob_negative、prob_neutral、prob_positive | 已加类别偏置后的三类概率，保留六位小数 | 不列入正文预测表 |
| text_observed_fraction、audio_observed_fraction、vision_observed_fraction | 在共同支持掩码内的有效观测比例，保留四位小数 | 不列入正文预测表 |
| gate_text、gate_audio、gate_vision | 融合模型内部的三路门控权重，保留四位小数 | 不列入正文预测表 |

CSV中的三类概率已经校准，不能再次施加偏置；它与验证归档中仍保留原始probabilities的JSON字段口径不同。概率因显示舍入，其和可能不严格等于1。观测比例以共同支持掩码为分母，分母至少取1；模型内部q则按50位置计算，二者不能互换。门控权重属于模型中间量，不等同于因果贡献。表5-24已按sample_id逐项核对类别及强度，并确认与此CSV一致；另一份冻结F₀的历史预测文件不作为表5-24的数据来源。

### 5 最终模型缺失鲁棒性敏感性分析：63条件完整结果

表5-28按名义缺失率分为三个续表，完整覆盖7种非空模态组合×3种比例×3种位置。全部结果属于最终B₀的official valid单模型，采用逐样本CPU ONNX int8、固定偏置及一致性强度输出，每行N=728。ACC以百分数显示，F1、MAE及Pearson保留四位小数；这里不再选择epoch、种子或偏置。三个续表的模型、数据划分和推理口径完全相同。

这属于同一已训练模型对输入损伤的敏感性分析，不是输入模态消融。严格的输入模态消融需要分别训练文本、音频、视觉、文本＋音频、文本＋视觉、音频＋视觉及三模态七种模型，并在相同协议下评价；仅在推理时屏蔽输入不能替代该实验。

同目录“问题二_最终基线63种缺失条件完整结果.csv”保存未按正文精度舍入的指标、各类F1及三路实际观测缺失率。名义比例按有效跨度定义，实际移除观测比例受原有空洞和取整影响，故两者分别保留。原始输入条件另见表5-23，不计入这63行。

'''+ '\n\n'.join(tables)+'''

### 6 三层消融的覆盖边界与最小对照矩阵

现有证据尚未构成完整的三层消融。训练策略层已完成自然采样、均衡采样及均衡加对比监督的统一对照，但缺失增强概率、类别权重、任务损失与编码器更新范围尚未全部纳入同一组实验。网络结构层已完成交互分支汇总方式的对照，但没有逐项关闭有效掩码、注意力池化及原Fusion门控的完整结果。输入层尚未完成七种模态组合的统一独立重训；第五轮文本与音视频专家只能说明该轮专家组合的表现，不能补齐这七种模型。

为区分已完成证据和待检验机制，表5-29列出围绕最终基线B₀的最小对照矩阵。该表是实验覆盖说明，不是已完成的性能消融表；未实测项目不填入其他轮次分数，也不以零值表示。基线本身没有时序交叉注意力，因此“无时序交互”与完整B₀是同一结构，不能作为独立消融行重复计数。若要验证时序模块，必须先定义含该模块的增强模型，再在同一训练协议下移除它。

表5-29 最小模块对照矩阵及现有证据状态（非性能结果表）

| 对照项 | 相对最终基线B₀的操作定义 | 同协议证据状态 |
|---|---|---|
| 完整模型 | 顶部两层微调、有效掩码、注意力池化、门控融合及联合损失 | 已有基线指标；其余待测变体尚未构成同组重训 |
| 无缺失增强 | 仅将人工连续缺失施加概率由0.3改为0；保留原始缺失和填充处理 | 尚无该最小矩阵的配对实测结果 |
| 无有效掩码 | 保留填充屏蔽；取消对真实缺失位置的显式可用性处理，输入零／UNK编码保持一致 | 尚无配对实测；须明确各掩码通路，不能与填充污染混为一谈 |
| 平均融合替代门控 | 仅将可用模态间学习权重改为等权，保留投影、池化、拼接和任务头 | 尚无原Fusion的配对实测；第七轮等权交互分支不等价 |
| CE only | 仅将回归损失系数设为0，保留分类权重及其他训练设置 | 尚无配对实测；未受监督的回归头不能作为有效强度模型 |
| 无时序交互 | B₀原本不含时序交叉注意力，与完整基线相同 | 不适合作为B₀的独立变体；第七轮只提供结构整体对照 |

若补做表5-29，应事先固定fit／dev的视频分组、随机种子、优化器、学习率、batch、训练预算、检查点规则和评价指标；每行只改变指定因素。损失或输入组合本身是被消融因素时，要求其他条件相同，而不能要求被改变的因素也保持不变。为获得严格固定训练步数的比较，还需预先确定共同训练轮数；表5-26现有实验共享10轮开发预算和选模规则，但各自最佳epoch并不相同，不能表述为相同轮数的最终检查点。

缺失增强的施加概率与被删片段的长度比例也必须分开：前者控制多少样本受到扰动，后者控制每个受损样本删除多长片段。“10%、30%、50%增强”若不说明是哪一个量，不能形成可复现对照。无掩码实验应保留基本padding处理，并逐一说明模态内聚合、可用比例及门控屏蔽如何变化；否则性能下降可能仅由填充参与运算造成。CE only及SmoothL1 only分别缺乏另一任务的直接监督，不应将未训练输出头的指标作为同等有效的双任务成绩。

因此，本文仅将表5-26称为统一协议模块消融，将表5-28称为最终模型缺失鲁棒性敏感性分析，不声称训练策略、网络结构和输入模态三层消融均已完成。尚缺的逐模块与独立模态重训证据，限制了对各组件必要性及因果增益的解释，但不改变现有实测结果的统计口径。
'''
    text=SOURCE.read_text(encoding='utf-8').split('\n## 5.3.9 ')[0].rstrip()
    SOURCE.write_text(text+'\n\n'+section,encoding='utf-8')
    (ASSETS/'unified_ablation_snapshot.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
    sourcefiles=[selection,gridfile,ROOT/'C/scripts/infer_q2_corrected.py',csvpath,SOURCE.parent/'问题二_最终第三问入口模型_附件3预测.csv']
    (ASSETS/'supplement_manifest.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sourcefiles},ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUPPLEMENT_OK: 6 ablations, 63 unique conditions, 30 prediction mappings')

if __name__=='__main__':supplement()
