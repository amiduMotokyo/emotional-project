"""Integrate the selected frozen-BERT single model; preserve historical experiment identities."""
from pathlib import Path
import csv,json,re,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'B/q2_single_model_final/q2_single_model_final'
SOURCE=ROOT/'docs/论文稿/问题二.md'
ASSETS=ROOT/'docs/C/paper/assets/q2_integrated'

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def integrate():
    metrics=json.loads((PACKAGE/'results/final_metrics.json').read_text(encoding='utf-8'))
    predictions={}
    for split in ['valid','test']:
        rows=read_csv(PACKAGE/f'results/{split}_predictions.csv');predictions[split]=rows
        y=np.array([float(r['true_intensity']) for r in rows]);c=np.array([int(r['true_class']) for r in rows]);r=np.array([float(x['predicted_intensity']) for x in rows])
        for kind,prefix in [('raw','raw_probability_'),('calibrated','biased_probability_')]:
            p=np.array([[float(x[prefix+k]) for k in ['negative','neutral','positive']] for x in rows]);label=p.argmax(1);m=metrics[split][kind]
            cm=np.bincount(3*c+label,minlength=9).reshape(3,3)
            assert abs(float((c==label).mean())-m['accuracy'])<1e-12
            assert abs(float(np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1))))-m['macro_f1'])<1e-12
            assert abs(float(np.abs(y-r).mean())-m['mae'])<1e-6
            assert abs(np.corrcoef(y,r)[0,1]-m['pearson'])<1e-6
            assert np.array_equal(cm,m['confusion'])
    special=read_csv(PACKAGE/'attachment3_predictions.csv');assert len(special)==len({x['sample_id'] for x in special})==30
    labelmap={'Negative':'负向','Neutral':'中性','Positive':'正向'}
    conflicts=sum((0 if float(x['predicted_intensity'])<0 else 2 if float(x['predicted_intensity'])>0 else 1)!=['Negative','Neutral','Positive'].index(x['predicted_polarity']) for x in special)
    grid=read_csv(PACKAGE/'results/validation_missingness.csv');assert len(grid)==144
    def group_rows(kind):
        out={}
        for r in grid:
            if r['kind']==kind:out.setdefault((r['modalities'],r['rate'],r['position']),[]).append(r)
        assert all(len(v)==3 for v in out.values());return out
    rates=group_rows('rate');positions=group_rows('position')
    def mean(rr,k):return float(np.mean([float(r[k]) for r in rr]))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'font.size':10})
    def save(fig,name):
        fig.savefig(ASSETS/f'{name}.png',dpi=240,bbox_inches='tight');fig.savefig(ASSETS/f'{name}.svg',bbox_inches='tight');plt.close(fig)
    cm=np.array(metrics['valid']['calibrated']['confusion']);fig,aa=plt.subplots(1,2,figsize=(10,4.1))
    aa[0].imshow(cm/cm.sum(1)[:,None],cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):aa[0].text(j,i,f'{cm[i,j]}\n{cm[i,j]/cm[i].sum():.1%}',ha='center',va='center',color='white' if cm[i,j]/cm[i].sum()>.55 else 'black')
    aa[0].set(xticks=range(3),yticks=range(3),xticklabels=['负向','中性','正向'],yticklabels=['负向','中性','正向'],xlabel='预测类别',ylabel='真实类别')
    aa[1].scatter([float(r['true_intensity']) for r in predictions['valid']],[float(r['predicted_intensity']) for r in predictions['valid']],s=12,alpha=.25,color='#355C7D');aa[1].plot([-3,3],[-3,3],'--',color='#D8933D');aa[1].set(xlabel='真实强度',ylabel='独立回归头预测强度',xlim=(-3.1,3.1),ylim=(-3.1,3.1));fig.tight_layout();save(fig,'single_final_diagnostics')
    fig,aa=plt.subplots(1,2,figsize=(10,4.1))
    for mod,label,color in zip(['T','A','V','TAV'],['文本','音频','视觉','三模态'],['#355C7D','#6C9A8B','#D8933D','#9467A0']):
        aa[0].plot([10,20,30,50,70],[100*mean(rates[(mod,str(r),'random')],'accuracy') for r in [.1,.2,.3,.5,.7]],'-o',label=label,color=color)
    aa[0].set(xlabel='名义缺失比例（%）',ylabel='valid ACC（%）',title='随机位置：三个扰动种子均值');aa[0].legend(fontsize=8);aa[0].grid(alpha=.2)
    mods=['T','A','V','TA','TV','AV','TAV'];ps=['front','middle','back','random']
    heat=np.array([[mean(positions[(m,'0.3',p)],'macro_f1') for p in ps] for m in mods]);aa[1].imshow(heat,cmap='Blues',aspect='auto')
    aa[1].set(xticks=range(4),xticklabels=['起始','中部','末尾','随机'],yticks=range(7),yticklabels=mods,title='30%缺失：Macro-F1')
    for i in range(7):
        for j in range(4):aa[1].text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center',color='white' if heat[i,j]>(heat.min()+heat.max())/2 else 'black')
    fig.tight_layout();save(fig,'single_final_missingness')
    fig,ax=plt.subplots(figsize=(11,4.5));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,4.5))
    boxes=[(.1,3,2.5,1,'冻结BERT 768维\n音频74维／视觉35维'),(3.1,3,3.1,1,'观测标记＋缺失嵌入\n32维投影＋位置编码'),(7,3,3.7,1,'三路单层四头Transformer\n模态内时序表示'),(.3,1,3.1,1,'逐位置动态模态融合\n全缺失位置回退向量'),(4.1,1,2.6,1,'时间注意力池化\n共享隐藏表示'),(7.4,1,3.3,1,'分类偏置校准\n独立回归强度')]
    for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc='#EDF2F6',ec='#355C7D'));ax.text(x+w/2,y+h/2,t,ha='center',va='center')
    for a,b in [((2.7,3.5),(3,3.5)),((6.3,3.5),(6.9,3.5)),((8.8,2.9),(1.85,2.1)),((3.5,1.5),(4,1.5)),((6.8,1.5),(7.3,1.5))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#355C7D'})
    ax.text(5.5,.25,'训练：完整／缺失双视图监督＋保留比例加权一致性；推理：单模型、无集成',ha='center');save(fig,'single_final_architecture')
    chapter=r'''### 5 最终模型：冻结BERT与缺失感知时序融合

最终预测器记为S₀，采用冻结bert-base-uncased、模态内时序编码、逐位置动态融合和双视图一致性训练。与冻结MiniLM基线F₀、校准微调MiniLM基线B₀相比，S₀不仅改变文本表示，还改变模态交互位置与训练目标，因而不能将其性能差异归因于单一模块。S₀只加载一个32维预测模型，seed43，第3轮检查点，不进行概率或参数集成；其校准后official valid正确477/728条。

文本由text_bert经过冻结BERT得到50×768序列，音频和视觉维度为50×74与50×35。定义Pₜ为排除CLS、SEP及填充后的公共内容支持，Oₜᵐ为各模态在该支持内的观测状态；文本UNK不作为可用观测，音视频全零行按不可用处理，但不推断其具体成因。三路标准化参数仅由train的有效观测估计，标准差下限10⁻⁵，标准化结果裁剪到[-10,10]，无效观测置零。

每一路先附加观测标记、投影到32维；缺失位置采用可学习模态缺失向量，再加入幅度0.1的正弦位置编码：

$$
u_t^m=P_t\left[O_t^m\phi_m([x_t^m;O_t^m])+(1-O_t^m)e_{\mathrm{miss}}^m+0.1\,\mathrm{PE}_t\right]. \tag{5.3-16}
$$

其中φ包含Linear、LayerNorm和GELU。三路分别经过一层、四头、前馈宽度64的pre-norm Transformer，Dropout为0.2；注意力键屏蔽P之外的填充，保留内容支持内的缺失嵌入，使编码器可学习局部上下文关系。该结构是模态内自注意力，不是将三路序列直接送入跨模态注意力。

将三路当前位置隐藏状态和观测标记拼接，经99→32→3网络产生动态模态权重，屏蔽当前不可用模态后归一化：

$$
a_t^m=\frac{O_t^m\exp(e_t^m)}{\sum_k O_t^k\exp(e_t^k)},\qquad z_t=\sum_m a_t^m h_t^m. \tag{5.3-17}
$$

若该位置三路均不可用，则权重全为零，融合表示回退为可学习全缺失向量与位置编码之和。再在公共支持P上进行时间注意力池化，并由共享隐藏层输出分类和回归：

$$
\beta_t=\frac{P_t\exp(w^\mathsf{T}z_t)}{\sum_uP_u\exp(w^\mathsf{T}z_u)},\quad h=\sum_t\beta_tz_t,\quad r=3\tanh(w_r^\mathsf{T}g(h)+b_r). \tag{5.3-18}
$$

模型先在每个位置融合可用模态，再对位置汇聚，区别于F₀／B₀先将每一路压缩为单个向量的流程。逐位置模态权重和时间注意力可为第三问提供中间表示，但不能直接当作因果贡献。总体流程见图5-21。

### 6 双视图监督与保留比例加权一致性

训练使用3395条train，不更新BERT参数。六组固定缺失视图的随机种子为91000—91005，每个epoch为每条样本抽取其中一组；每组视图约20%的样本不增加缺失，其余样本以0.5、0.35、0.15的概率选择一、二、三路模态，片段长度比例在0.1—0.6之间均匀抽取，包含同步和异步区间。文本在BERT前替换为UNK并重新编码，避免完整上下文泄漏。

一个batch同时包含原始视图f和缺失视图d。监督损失采用Huber回归加0.5倍逆平方根类别频率加权交叉熵：

$$
L_{\mathrm{sup}}^v=\mathrm{Huber}(r^v,y)+0.5\,\mathrm{CE}_w(l^v,c),\qquad v\in\{f,d\}. \tag{5.3-19}
$$

以缺失视图相对原始视图的观测保留比例ρᵢ加权一致性项；原始视图输出在一致性分支中停止梯度：

$$
\rho_i=\frac{\sum_{t,m}O_{i,t}^{d,m}}{\max(1,\sum_{t,m}O_{i,t}^{f,m})},\quad L_{\mathrm{con}}=\frac1N\sum_i\rho_i\left[|r_i^d-\mathrm{sg}(r_i^f)|+0.5D_{\mathrm{KL}}(\mathrm{sg}(p_i^f)\Vert p_i^d)\right]. \tag{5.3-20}
$$

$$
L=\tfrac12(L_{\mathrm{sup}}^f+L_{\mathrm{sup}}^d)+0.3L_{\mathrm{con}}. \tag{5.3-21}
$$

保留比例越低，一致性约束越弱，避免要求信息严重不足的样本机械复制完整视图预测。这一一致性指同一样本不同输入视图之间的预测约束，不是分类极性与回归符号之间的强制一致。

优化器为AdamW，初始学习率8×10⁻⁴、weight decay0.001、batch128、梯度裁剪1；余弦调度在20轮内将学习率降至8×10⁻⁵。完整训练预算为20轮，检查点及类别偏置按official valid选择，最终取第3轮。所用通用编码器固定为bert-base-uncased的revision 86b5e0934494bd15c9632b12f734a8a67f723594；冻结编码器约440MB，不能把278746字节的预测网络参数文件误当作全部推理依赖。

### 7 决策边界校准、结果及选择依据

分类偏置在负向和中性各[-0.6,0.6]、步长0.025的网格中搜索，正向固定零，共49²=2401种组合；按完整验证集正确数、Macro-F1、较小偏置范数等规则选择。最终单模型偏置为：

$$
p'_c=\frac{\exp(\log p_c+b_c)}{\sum_k\exp(\log p_k+b_k)},\qquad b=(0.375,0.275,0). \tag{5.3-22}
$$

该处理只改变分类边界，不改变回归强度，也不证明概率置信度已经校准。模型与偏置共同经过验证选择，不能将该分数解释为独立泛化估计。历史测试数据此前已经用于结果报告；当前测试只评价固定模型，仍不是新获得的盲测。

表5-23 最终S₀单模型的分类与原始强度结果

| 数据及分类口径 | ACC（%） | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| official valid，未校准 | 62.9121 | 0.576268 | 0.594934 | 0.660018 |
| official valid，校准后 | 65.5220 | 0.629518 | 0.594934 | 0.660018 |
| 历史test，未校准 | 66.5750 | 0.574567 | 0.623140 | 0.688214 |
| 历史test，校准后 | 66.0248 | 0.602987 | 0.623140 | 0.688214 |

验证校准使正确分类由458条增加至477条，Macro-F1由0.5763提高至0.6295；历史test的ACC则由66.5750%降至66.0248%，而Macro-F1提高。校准并非所有数据与指标上均有增益。验证混淆矩阵中，中性正确数由51增至81，召回率由27.72%增至44.02%，但仍有65条中性误判为正向。

选择S₀的依据是已完成搜索中的单模型验证表现、局部缺失评价以及可复现的专项输出。B₀的63.7363%及第六、七轮固定候选的62.77%、62.50%仅作历史验证背景；不同编码器、训练与选择协议使其差值不能作为单因素因果证据。第五轮65.61%是历史test三种子均值，更不能和65.5220%的valid单模型结果直接排名。第三问使用同一S₀预测器分析模态贡献及局部证据，以保证解释对象与情感输出一致。

![最终单模型分类与回归](../C/paper/assets/q2_integrated/single_final_diagnostics.png)

图5-19 最终S₀的official valid分类与回归诊断。分类采用固定偏置，强度保留原始回归头；散点图虚线表示理想预测关系。

![最终单模型缺失敏感性](../C/paper/assets/q2_integrated/single_final_missingness.png)

图5-20 最终S₀的缺失敏感性。左图为四类模态组合、五种比例与随机位置的ACC，右图为30%比例下七类模态组合、四种位置的Macro-F1；均对三个扰动种子平均，而不是三个模型的训练均值。

S₀采用分类和强度独立输出。校准后official valid有46/728条、历史test有38/727条出现负类预测正强度或正类预测负强度；此外分别有157条、124条中性预测对应非零强度。两项分别统计，不能把46或38当作包含中性冲突的总数。该差异作为方法局限保留，不施加符号修正来改变已有指标；前文F₀／B₀的一致性解码公式不适用于S₀。
'''
    chapter=re.sub(r'\\tag\{5\.3-(\d+)\}',lambda m:r'\tag{5.3-'+str(int(m[1])-1)+'}',chapter)
    table=['| 样本编号 | 极性 | 强度 | 样本编号 | 极性 | 强度 |','|---|---|---:|---|---|---:|']
    for i in range(15):
        rr=[special[i],special[i+15]];table.append('| '+' | '.join(v for r in rr for v in [r['sample_id'],labelmap[r['predicted_polarity']],f"{float(r['predicted_intensity']):.6f}"])+' |')
    counts={k:sum(r['predicted_polarity']==k for r in special) for k in labelmap}
    specialtext='''## 5.3.8 附件3专项预测及模型适用范围

使用最终S₀模型对附件3全部30条对齐样本推理。分类取偏置处理后的最大概率类别，强度直接取独立回归头输出；不使用专项标签、不进行专项调参。表5-24与配套CSV逐样本对应。

表5-24 附件3全部30条样本的S₀预测（强度保留六位小数）

'''+ '\n'.join(table)+f'''

预测负向{counts['Negative']}条、中性{counts['Neutral']}条、正向{counts['Positive']}条，其中{conflicts}条极性与强度符号不一致，包括中性类别对应非零强度的情况。例如附件3_02分类为负向、回归强度为0.117791，两者均保留实际输出，不能通过修改表格使其看似一致。预测类别分布不等于真实类别分布，无标签专项样本不能计算准确率。

S₀通过显式观测状态、缺失嵌入、局部时序编码与保留比例加权一致性学习处理局部缺失。其限制是分类和强度边界未被强制统一、模型选择多次利用验证反馈，且对齐位置不等于真实时间。三层消融表5-30至表5-32针对MiniLM–Fusion参照，不能用来宣称S₀的每个新增组件已获得独立消融验证。第三问以S₀的固定输出为解释目标，并应将分类证据与强度证据区分表述。

冻结BERT及环境依赖不包含在轻量预测参数文件内，因而该文件大小不能证明完整提交材料已满足容量约束。推理需准备相同BERT revision、标准化统计、模型参数及固定偏置。原始专项重编码核验中30条类别一致、预测最大差异为零，支持所核查推理路径的可复现性，但不构成无标签样本上的性能验证。
'''
    s=SOURCE.read_text(encoding='utf-8')
    s=s.split('\n## 5.3.11 ')[0]
    start=s.index('### 5 为什么最终保留校准微调基线') if '### 5 为什么最终保留校准微调基线' in s else s.index('### 5 最终模型：')
    end=s.index('## 5.3.9',start)
    s=s[:start]+chapter+'\n'+specialtext+'\n'+s[end:]
    # Existing MiniLM evidence remains historical, never relabeled as S0 measurements.
    s=s.replace('最终模型符号B₀','历史基线符号B₀').replace('最终基线B₀','历史基线B₀').replace('最终B₀','历史基线B₀').replace('最终模型缺失鲁棒性敏感性分析','历史B₀缺失鲁棒性敏感性分析')
    s=s.replace('最终专项输出采用B₀','最终专项输出采用S₀').replace('最终预测和第三问采用B₀','最终预测和第三问采用S₀')
    s=s.replace('使用附件词元输入；MiniLM输出384维序列','使用附件词元；历史MiniLM输出384维，最终BERT输出768维')
    s=s.replace('联合回归并输出与极性一致的强度','S₀独立回归；历史F₀／B₀采用极性一致性映射')
    s=s.replace('## 5.3.3 缺失感知多模态模型','## 5.3.3 缺失感知Fusion基线模型')
    s=s.replace('推理阶段的极性一致性约束为：','F₀／B₀推理阶段的极性一致性约束为：')
    s=s.replace('该表支持均衡与当前对比配置没有改善dev，且时序结构在dev具有局部收益；结合完整重训后FP32均值持平及固定部署种子未改善的结果，尚不足以证明动态分支选择能够稳定替换最终基线。','该表支持均衡与当前对比配置没有改善dev，且时序结构在dev具有局部收益；结合完整重训后FP32均值持平及固定部署种子未改善的结果，尚不足以证明动态分支选择能够稳定替换同轮Fusion参照。')
    s=s.replace('这里保留“优化前的方法”是指保留已有校准微调配置，没有继续叠加未获稳定支持的扩容、专家融合和时序动态模块，并非撤销基线本身的微调与校准。','B₀仅作为历史校准微调对照，最终S₀的方法与结果见第5.3.7节。')
    s=s.replace('最终采用模型及后续优化参照来源','历史校准微调对照及消融参照来源').replace('使用最终B₀模型推理','使用最终S₀模型推理')
    s=s.replace('因而最终模型必须按实际逐样本CPU ONNX路径评估，再对校准后的极性和强度一并检查。这也是保留完整基线推理方案，而不是只依据训练阶段最高准确率选型的重要原因。','因此，MiniLM量化候选应在相应CPU ONNX路径上评估。这一检查只适用于其自身推理实现，不能替代S₀的冻结BERT推理检查。')
    s=s.replace('最终B₀及附件3预测保持原先锁定结果','最终S₀及其附件3预测不由这些开发消融结果重新选择')
    s=s.replace('历史基线B₀及附件3预测保持原先锁定结果','最终S₀及其附件3预测不由这些开发消融结果重新选择')
    s=s.replace('保持原先锁定结果','不据此重新选择')
    s=s.replace('表5-23、表5-28；63.7363%为int8校准后单模型','表5-28；63.7363%为历史int8校准后单模型')
    row='| official valid，S₀ | 728 | 模型、epoch及2401种偏置选择 | 表5-23；65.5220%为最终单模型 |\n'
    if row not in s:s=s.replace('| 历史test | 727 |',row+'| 历史test | 727 |',1)
    s=s.replace('表5-17第二轮、表5-18的test列；65.61%为三种子均值','表5-17、表5-18的历史test；表5-23的66.0248%为S₀单模型')
    # Replace the CSV contract, including field names and independent-output semantics.
    start=s.index('### 4 附件3 CSV与表5-24的一一对应');end=s.index('### 5 ',start)
    s=s[:start]+'''### 4 附件3 CSV与表5-24的一一对应

表5-24对应“问题二_最终第三问入口模型_附件3预测.csv”：S₀，seed43，第3轮，冻结BERT与32维预测网络，偏置[0.375,0.275,0]。文件包含30条记录和10个字段；sample_id为唯一关联键。表格左栏01—15、右栏16—30，类别翻译为中文，强度显示六位小数，CSV保留原有精度。

表5-27 最终S₀专项CSV字段及正文对应

| CSV字段 | 含义及口径 | 表5-24对应 |
|---|---|---|
| sample_id | 附件3原始对齐文件名，不含扩展名 | 样本编号 |
| predicted_polarity | 校准后的最大概率类别 | 极性，翻译为中文 |
| predicted_intensity | 原始回归强度，范围[-3,3]；不强制符号一致 | 强度，显示六位小数 |
| raw_head_polarity | 校准前分类头的最大概率类别 | 未列出 |
| raw_probability_negative、raw_probability_neutral、raw_probability_positive | 校准前三类概率 | 未列出 |
| biased_probability_negative、biased_probability_neutral、biased_probability_positive | 固定偏置处理后的三类概率，不应再次校准 | 未列出 |

该CSV不包含观测比例或门控权重字段。分类与回归符号不同属于独立输出，不应通过再次校准、改写类别或强制置零掩盖。30条类别和强度按sample_id逐一对应，历史MiniLM的专项预测不作为表5-24的数据来源。

'''+s[end:]
    s=s.replace('原始输入条件另见表5-23，不计入这63行。','原始输入B₀的历史ACC为63.7363%，不计入这63行；表5-23则属于S₀，二者不可混用。')
    s=s.replace('问题二_最终基线63种缺失条件完整结果.csv','问题二_历史MiniLM基线63种缺失条件完整结果.csv')
    risk='S₀对模型、epoch及2401种偏置组合使用official valid进行选择，存在额外搜索选择效应；其65.5220%同样不是独立泛化估计。S₀不采用MiniLM的五折共享偏置流程，不能把此前的折外校准证据移用于S₀。\n\n'
    marker='五折偏置校准使每条折外预测'
    if risk not in s:s=s.replace(marker,risk+marker)
    # Complete current-model missingness tables, preserve 144 source rows in CSV.
    def gt(title,items):
        lines=[title,'','| 模态 | 比例 | 位置 | ACC（%） | Macro-F1 | MAE | Pearson |','|---|---:|---|---:|---:|---:|---:|']
        for (m,r,p),rr in items:
            lines.append(f"| {m} | {float(r):.0%} | {dict(front='起始',middle='中部',back='末尾',random='随机')[p]} | {100*mean(rr,'accuracy'):.2f} | {mean(rr,'macro_f1'):.4f} | {mean(rr,'mae'):.4f} | {mean(rr,'pearson'):.4f} |")
        return '\n'.join(lines)
    s+='''\n## 5.3.11 最终S₀的结构示意与缺失条件结果

![最终S₀结构](../C/paper/assets/q2_integrated/single_final_architecture.png)

图5-21 最终S₀的单模型预测流程。BERT保持冻结；缺失嵌入、模态内Transformer、逐位置动态融合及任务头参与训练。分类与强度分别输出，双视图一致性用于训练期。

S₀的缺失评估包括两项预定义网格：比例分析采用T、A、V、TAV四种组合、10%、20%、30%、50%、70%五种比例和随机位置；位置分析采用七种非空模态组合、固定30%比例和起始、中部、末尾、随机四种位置。每项使用701、702、703三个扰动种子，分别记录60与84行，共144行。两个网格在随机位置、30%比例的四种组合处重叠，因此144行不是144个独立条件，更不是63条件全因子网格。

以下表格分别对三个扰动种子取均值。固定起始、中部、末尾时，不同扰动种子可能生成相同区间，重复结果不构成独立重复训练。缺失区间按公共有效内容位置列表选取，与F₀／B₀按跨度定义的协议不同，不能把跨协议差值直接解释为模型鲁棒性增益。三类指标均来自728条official valid；强度保留原始回归头，不做符号校正。

'''+gt('表5-33 最终S₀缺失比例敏感性（official valid，三个扰动种子均值）',list(rates.items()))+'\n\n'+gt('表5-34（1/2）最终S₀缺失位置敏感性（30%，official valid）',list(positions.items())[:14])+'\n\n'+gt('表5-34（2/2）最终S₀缺失位置敏感性（30%，official valid）',list(positions.items())[14:])+'''

原始144行保存在“问题二_最终S0缺失敏感性144行.csv”，包含kind、modalities、rate、position、mask_seed、accuracy、macro_f1、weighted_f1、mae及pearson；与正文均值表通过前三个条件字段和kind对应。本文分别汇总两个网格，不把交叠行重复加权为单一总体鲁棒性得分。专项预测与上述验证扰动结果用于不同目的，均不提供新独立盲测证据。
'''
    SOURCE.write_text(s,encoding='utf-8')
    shutil.copyfile(PACKAGE/'attachment3_predictions.csv',SOURCE.parent/'问题二_最终第三问入口模型_附件3预测.csv')
    shutil.copyfile(PACKAGE/'results/validation_missingness.csv',SOURCE.parent/'问题二_最终S0缺失敏感性144行.csv')
    old=SOURCE.parent/'问题二_最终基线63种缺失条件完整结果.csv'
    if old.exists():old.replace(SOURCE.parent/'问题二_历史MiniLM基线63种缺失条件完整结果.csv')
    sources=[PACKAGE/'protocol.json',PACKAGE/'src/model.py',PACKAGE/'src/train_search.py',PACKAGE/'src/common.py',PACKAGE/'src/prepare.py',PACKAGE/'attachment3_predictions.csv',PACKAGE/'results/final_metrics.json',PACKAGE/'results/validation_missingness.csv',PACKAGE/'results/valid_predictions.csv',PACKAGE/'results/test_predictions.csv',PACKAGE/'results/selection_frozen.json',PACKAGE/'checkpoints/width32_seed43_cal.pt']
    (ASSETS/'single_final_source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},ensure_ascii=False,indent=2),encoding='utf-8')
    (ASSETS/'single_final_content_check.json').write_text(json.dumps(dict(valid_rows=728,test_rows=727,special_rows=30,metrics_recomputed_from_saved_predictions=True,special_sign_conflicts=conflicts,missing_rows=144,rate_groups=len(rates),position_groups=len(positions)),indent=2),encoding='utf-8')
    print('SINGLE_FINAL_VERIFIED',conflicts,'special sign conflicts')

if __name__=='__main__':integrate()
