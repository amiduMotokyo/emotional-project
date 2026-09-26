"""Verify all completed ablations and update paper tables; never select a deployment model."""
from pathlib import Path
import csv,json,hashlib,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'C/outputs/q2_three_layer_ablation_v2'
SOURCE=ROOT/'docs/论文稿/问题二.md'
ASSETS=ROOT/'docs/C/paper/assets/q2_integrated'

def report():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'C/configs/q2_three_layer_ablation.json').read_text(encoding='utf-8'))
    assert summary['complete'] and len(summary['runs'])==66
    by={g:[] for g in cfg['groups']};ids=None
    hashes={};flat=[]
    for row in summary['runs']:
        assert row['epoch']==10 and row['seed'] in cfg['seeds'] and row['scenarios'][0]['n']==391
        path=OUT/'runs'/f"{row['group']}_{row['seed']}"
        z=np.load(path/'dev_predictions.npz')
        if ids is None:ids=z['sample_id'].copy()
        assert np.array_equal(ids,z['sample_id'])
        assert z['prob'].shape==(4,391,3) and np.isfinite(z['prob']).all() and np.isfinite(z['reg']).all()
        m=row['scenarios'][0]
        if m['accuracy'] is not None:assert abs(float((z['prob'][0].argmax(-1)==z['cls']).mean())-m['accuracy'])<1e-12
        raw_mae=float(np.abs(z['reg'][0]-z['score']).mean())
        if row['raw_regression_mae'] is not None:assert abs(raw_mae-row['raw_regression_mae'][0])<1e-6
        by[row['group']].append(row)
        flat.append(dict(group=row['group'],seed=row['seed'],epochs=10,dev_n=391,accuracy=m['accuracy'],macro_f1=m['macro_f1'],raw_mae=None if row['raw_regression_mae'] is None else row['raw_regression_mae'][0],raw_pearson=None if row['raw_regression_pearson'] is None else row['raw_regression_pearson'][0],missing_macro_f1=None if m['macro_f1'] is None else float(np.mean([x['macro_f1'] for x in row['scenarios'][1:]])),seconds=row['seconds']))
        for f in [path/'result.json',path/'dev_predictions.npz',path/'checkpoint.pt']:hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for g in by:
        by[g].sort(key=lambda x:x['seed']);assert [r['seed'] for r in by[g]]==cfg['seeds']
    fpath=SOURCE.parent/'问题二_三层消融66次训练结果.csv'
    with fpath.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(sorted(flat,key=lambda r:(r['group'],r['seed'])))
    def values(g,k):
        rr=[r for r in flat if r['group']==g];return [r[k] for r in sorted(rr,key=lambda r:r['seed'])]
    def fmt(g,k,percent=False,sd=False):
        a=values(g,k)
        if a[0] is None:return '—'
        a=np.array(a)*(100 if percent else 1);digits=2 if percent else 4
        v=f'{a.mean():.{digits}f}'
        return v+(f'±{a.std(ddof=1):.{digits}f}' if sd else '')
    def table(groups):
        lines=['| 配置 | dev ACC（%） | Macro-F1 | 原始MAE | 原始Pearson | 缺失Macro-F1 |','|---|---:|---:|---:|---:|---:|']
        for g,name in groups:lines.append('| '+' | '.join([name,fmt(g,'accuracy',True,True),fmt(g,'macro_f1'),fmt(g,'raw_mae'),fmt(g,'raw_pearson'),fmt(g,'missing_macro_f1')])+' |')
        return '\n'.join(lines)
    training=[('full','完整参照（概率0.3）'),('prob_0','无人工缺失增强'),('prob_0.1','增强概率0.1'),('prob_0.5','增强概率0.5'),('length_0.1','固定缺失长度10%'),('length_0.3','固定缺失长度30%'),('length_0.5','固定缺失长度50%'),('unweighted','无类别权重'),('ce_only','CE only'),('reg_only','SmoothL1 only'),('frozen','冻结文本编码器')]
    structure=[('full','原Fusion（无交互／专家）'),('no_mask','隐藏新增缺失掩码'),('mean_pool','有效位置平均池化'),('mean_fusion','可用模态平均融合'),('interaction','加入序列交叉注意力'),('experts','加入联合训练专家分支')]
    inputs=[('input_T','文本'),('input_A','音频'),('input_V','视觉'),('input_TA','文本＋音频'),('input_TV','文本＋视觉'),('input_AV','音频＋视觉'),('full','文本＋音频＋视觉')]
    baseline=np.array(values('full','accuracy'))*100
    differences={g:((np.array(values(g,'accuracy'))*100-baseline).tolist() if values(g,'accuracy')[0] is not None else None) for g in by}
    ranked=sorted([(np.mean(d),g) for g,d in differences.items() if g!='full' and d is not None],reverse=True)
    gain,g=ranked[0];bestname=dict(training+structure+inputs)[g]
    coverage='''### 6 三层消融的补做范围与最小对照矩阵

在历史实验之外，补做22种配置、三个配对种子的统一训练，共66个独立模型、660个epoch。所有模型采用同一fit3004／dev391划分、固定10轮末轮检查点及相同优化器配置，不使用dev选epoch，也不读取official valid或test。本轮不与历史最佳epoch、量化校准结果直接排名；完整数值见第5.3.10节。

表5-29 最小模块对照矩阵与新增实测对应

| 对照项 | 实际操作定义 | 补做证据 |
|---|---|---|
| 完整模型 | 原Fusion、顶部两层微调、增强概率0.3、联合损失 | 三种子末轮参照，表5-30至表5-32 |
| 无缺失增强 | 仅关闭人工连续缺失，保留原始缺失及padding处理 | 表5-30，无人工缺失增强 |
| 无有效掩码 | 保留原始支持范围，仅隐藏新增人工缺失的显式掩码 | 表5-31；不等于取消全部有效掩码 |
| 平均融合替代门控 | 可用模态等权，保留池化、拼接和任务头 | 表5-31，可用模态平均融合 |
| CE only | 回归损失权重为零，其他条件相同 | 表5-30；回归指标不报告 |
| 无时序交互 | 原Fusion本来无交互，以增加交叉注意力形成对照 | 表5-31，原Fusion与加入交互 |

本次三层实验覆盖预定义的训练策略、网络模块和七种独立重训输入组合，不声称穷尽全部实现。例如，掩码实验保留padding和原始缺失处理，检验新增缺失可用性；专家实验是新增联合训练分支，并非第五轮独立预训练专家的重复。表5-26仍属于历史开发选模消融，表5-28仍是最终模型缺失鲁棒性敏感性分析，两者不因新增实验而改变含义。
'''
    section='''## 5.3.10 固定训练预算下的三层消融实测

### 1 统一协议与独立重训

采用附件2 train内部按视频分组的fit3004和dev391，三个种子为20267001、20267002、20267003，固定训练10轮并取末轮，batch32，AdamW、权重衰减0.01、梯度裁剪1。MiniLM顶部两层学习率2×10⁻⁵，Fusion学习率2×10⁻⁴；基准目标为逆平方根类别频率加权CE＋0.8SmoothL1。除指定因素外，数据顺序、共有参数初始化、训练预算及评价规则保持一致。此处dev沿用已有开发划分，仍不构成独立盲测。

干净dev和文本、音频、视觉各30%中部连续缺失构成四种评价环境。“缺失Macro-F1”只平均后三种环境，不是63条件均值；每个环境使用相同391条样本。ACC报告三种子均值±样本标准差，其余列为三种子均值。为使回归单任务与联合任务可比，三张表统一报告原始回归MAE和Pearson，不用分类头进行符号校正；它们与前文一致性强度指标不同。

### 2 训练策略消融

表5-30 训练策略消融：internal dev，FP32，固定10轮末轮

'''+table(training)+'''

增强概率决定每条训练样本是否受损，长度比例决定受损片段的跨度。完整参照的长度从10%、20%、30%、50%抽取；固定长度组仅固定该比例，增强概率仍为0.3。概率0.3组即完整参照，不重复训练。CE only的回归头无监督、SmoothL1 only的分类头无监督，对应指标以“—”表示，不把未训练任务输出作为有效能力。

### 3 网络模块消融

表5-31 网络模块消融：internal dev，FP32，固定10轮末轮

'''+table(structure)+'''

隐藏新增缺失掩码时仍将受损输入替换为零或UNK，但池化、可用比例和门控保留缺失前的原始掩码；这隔离新增人工缺失的显式可用性作用，不引入padding污染。均值池化只替换模态内注意力汇聚，平均融合只替换可用模态门控权重。

交互组在原投影和池化之间增加共享四头序列交叉注意力与残差归一化，故比较的是这个交互模块整体，并非单个运算的纯参数量控制。专家组新增文本、音视频两个线性任务分支，其可用分支均值与原主头输出各占一半，通过相同最终任务损失联合训练；没有额外专家预训练或辅助损失。无交互、无专家均复用完整参照，不重复计为两个新模型。

### 4 输入模态消融

表5-32 独立重训的输入模态消融：internal dev，FP32，固定10轮末轮

'''+table(inputs)+'''

七种输入组合分别独立学习模型参数，未用模态输入与有效掩码均被移除，不向门控传递其观测或可用比例；无文本组不运行文本编码器。保留统一接口与任务头以控制额外结构差异，未用投影不能读取原始观测。训练期与评价期始终使用同一输入组合，因此本表不同于固定模型在推理时遮挡模态的敏感性分析。共享三模态参照只训练一次，表间复用不增加独立样本数。

### 5 配对差值与结论边界

'''+f'完整参照的dev ACC为{fmt("full","accuracy",True,True)}%，Macro-F1为{fmt("full","macro_f1")}。本轮干净dev平均ACC最高的新增配置为“{bestname}”，相对完整参照变化{gain:+.2f}个百分点；三个配对种子差值为'+ '、'.join(f'{v:+.2f}' for v in differences[g])+'''个百分点。这是本轮开发数据的描述性比较，不代表已通过独立验证，也不据此替换最终预测器。

不同配置改变了训练分布、监督目标或有效表示能力，改进应结合Macro-F1、强度误差及缺失条件共同解释。单个指标或种子提高不能证明组件普遍必要或无效；参数新增的交互和专家组也不是严格等参数量对照。完整逐种子指标保存在“问题二_三层消融66次训练结果.csv”，缺少监督的任务留空。新增实测填补本次预定义三层对照，但不消除历史验证复用风险；最终B₀及附件3预测保持原先锁定结果。
'''
    text=SOURCE.read_text(encoding='utf-8').split('\n## 5.3.10 ')[0]
    marker='### 6 三层消融的'
    at=text.index(marker,text.index('## 5.3.9'))
    text=text[:at]+coverage+'\n'+section
    SOURCE.write_text(text,encoding='utf-8')
    ASSETS.mkdir(parents=True,exist_ok=True)
    (ASSETS/'three_layer_manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(complete=True,groups=22,runs=66,epochs=660,training_seconds=sum(r['seconds'] for r in flat),same_dev_ids=True,paired_accuracy_delta_pp=differences)
    (OUT/'paper_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('THREE_LAYER_REPORT_OK',result['training_seconds'])

if __name__=='__main__':report()
