"""Render round-four paper figures from hash-verified saved results only."""
import sys
import csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from C.scripts.plot_q2_round3_paper import read, write, sha, decorate
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'C/outputs/q2_round4_structure_v1'
ASSETS=ROOT/'docs/C/paper/assets/round4'
NAMES=dict(baseline='原基线',text_only='仅文本',wide_control='加宽基线',factorized='分解头',interaction='序列交互',combined='交互＋分解头')
COLORS=['#355C7D','#6C9A8B','#D8933D','#9467A0']

def snapshot():
    expected={}
    for p in (RUN/'state').glob('*.json'):expected.update(read(p)['artifacts'])
    sources={}
    def checked(name):
        p=RUN/name
        assert sha(p)==expected[str(p)],name
        sources[name]=sha(p)
        return p
    d=dict(selection=read(checked('selection.json')),test=read(checked('test_summary.json')),histories={},confusion={})
    for p in sorted((RUN/'models').glob('*/history.json')):
        d['histories'][p.parent.name]=read(checked(p.relative_to(RUN).as_posix()))
    z=np.load(checked('test/clean.npz'))
    for family in ['baseline','wide_control']:
        cm=sum(np.bincount(z['cls']*3+z[f'{family}_{s}_prob'].argmax(-1),minlength=9).reshape(3,3) for s in [20263001,20263002,20263003])
        assert cm.sum()==2181
        assert abs(np.trace(cm)/2181-d['test']['scenarios']['clean'][family]['mean']['accuracy'])<1e-12
        d['confusion'][family]=cm.tolist()
    assert d['selection']['selected']=='wide_control'
    write(ASSETS/'plot_data.json',d)
    refs=['A/problem2_training_v2/plot_weighted_results.py','B/scripts/plot_q3.py','C/scripts/plot_q2_round3_paper.py']
    write(ASSETS/'source_manifest.json',dict(source_sha256=sources,plot_data_sha256=sha(ASSETS/'plot_data.json'),style_references={p:sha(ROOT/p) for p in refs}))
    return d

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(ASSETS/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def figures(d):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'path'})
    families=list(NAMES)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['(a) valid 准确率','(b) valid 宏 F1']):
        for i,n in enumerate(families):
            vals=[v['scenarios'][0][metric] for v in d['selection']['families'][n]['seeds']]
            ax.errorbar(np.mean(vals),i,xerr=np.std(vals,ddof=1),fmt='o',capsize=4,color=COLORS[1] if n=='wide_control' else COLORS[0])
        ax.set_yticks(range(6),[NAMES[n] for n in families]);ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1));ax.set_title(title);decorate(ax)
    save(fig,'fig1_valid_structures')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,split in zip(axes,['valid','test']):
        for i,n in enumerate(['baseline','wide_control']):
            vals=[v['scenarios'][0]['accuracy'] for v in d['selection']['families'][n]['seeds']] if split=='valid' else [v['accuracy'] for v in d['test']['scenarios']['clean'][n]['seeds']]
            ax.plot([1,2,3],vals,'o-',color=COLORS[i],label=NAMES[n])
        ax.set_xticks([1,2,3],['3001','3002','3003']);ax.set_xlabel('种子后四位（完整前缀 2026）');ax.set_title(f'{split}：同种子编号对照');ax.yaxis.set_major_formatter(PercentFormatter(1));decorate(ax);ax.legend()
    save(fig,'fig2_seed_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for n,color in zip(['baseline','wide_control','interaction','combined'],COLORS):
        h=[v for k,v in d['histories'].items() if k.rsplit('_',1)[0]==n]
        for ax,key in zip(axes,['loss','accuracy']):
            vals=np.array([[r['loss'] if key=='loss' else r['clean']['accuracy'] for r in rows] for rows in h]);mean=vals.mean(0);sd=vals.std(0,ddof=1)
            ax.plot(range(1,19),mean,label=NAMES[n],color=color);ax.fill_between(range(1,19),mean-sd,mean+sd,color=color,alpha=.12)
    for ax in axes:decorate(ax);ax.set_xlabel('训练 epoch');ax.set_xticks([1,3,6,9,12,15,18])
    axes[0].set_title('(a) 训练损失');axes[0].legend(fontsize=8);axes[1].set_title('(b) stop 准确率');axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    save(fig,'fig3_learning_curves')
    fig,axes=plt.subplots(2,4,figsize=(12,5.4),layout='constrained',sharey=True)
    modes=['text','audio','vision','text-audio','text-vision','audio-vision','text-audio-vision']
    labels=['文本','音频','视觉','文本＋音频','文本＋视觉','音频＋视觉','三模态']
    for ax,mode,label in zip(axes.flat,modes,labels):
        for n,color in zip(['baseline','wide_control'],COLORS):
            values=[d['test']['scenarios']['clean'][n]['mean']['macro_f1']]+[np.mean([d['test']['scenarios'][f'{mode}_{r:02d}_{p}'][n]['mean']['macro_f1'] for p in ['start','middle','end']]) for r in [10,30,50]]
            ax.plot([0,.1,.3,.5],values,'o-',color=color,label=NAMES[n],markersize=3)
        ax.set_title(label);ax.set_xticks([0,.1,.3,.5]);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('人工缺失比例');decorate(ax)
    axes.flat[7].axis('off');axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    save(fig,'fig4_missing_rates')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,n in zip(axes,['baseline','wide_control']):
        cm=np.array(d['confusion'][n]);pct=cm/cm.sum(1,keepdims=True);im=ax.imshow(pct,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{pct[i,j]:.1%}',ha='center',va='center',color='white' if pct[i,j]>.55 else '#1C2A39')
        ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set_xlabel('预测类别');ax.set_ylabel('真实类别');ax.set_title(NAMES[n]+'：三模型累计计数')
    fig.colorbar(im,ax=axes,format=PercentFormatter(1),shrink=.8,label='真实类别内比例')
    save(fig,'fig5_test_confusion')
    with (ASSETS/'plot_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['family','parameters','valid_accuracy','valid_sd','valid_macro_f1','missing_macro_f1'])
        for n,r in d['selection']['families'].items():
            s=r['mean']['scenarios'];w.writerow([n,r['parameters'],s[0]['accuracy'],r['accuracy_sd'],s[0]['macro_f1'],np.mean([v['macro_f1'] for v in s[1:]])])
    write(ASSETS/'render_manifest.json',dict(script_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in ASSETS.glob('fig*.*')}))

if __name__=='__main__':
    ASSETS.mkdir(parents=True,exist_ok=True)
    d=snapshot() if '--snapshot' in sys.argv or not (ASSETS/'plot_data.json').exists() else read(ASSETS/'plot_data.json')
    assert sha(ASSETS/'plot_data.json')==read(ASSETS/'source_manifest.json')['plot_data_sha256']
    figures(d)
    print('Rendered 5 figures; verified saved data; no inference.')
