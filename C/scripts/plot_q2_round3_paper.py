"""Snapshot verified saved round-three results and render publication figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'C/outputs/q2_round3_accuracy_v1'
ASSETS = ROOT/'docs/C/paper/assets/round3'
NAMES = dict(baseline='基线', plain_ce='普通CE', low_reg='低回归权重', low_missing='低缺失概率', combined='联合调整', frozen='文本冻结', tuned='顶部两层微调')
COLORS = ['#355C7D','#6C9A8B','#D8933D','#9467A0']

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def snapshot():
    expected = {}
    for p in (RUN/'state').glob('*.json'):
        expected.update(read(p)['artifacts'])
    sources = {}
    def checked(name):
        p = RUN/name
        assert sha(p) == expected[str(p)], name
        sources[name] = sha(p)
        return p
    data = dict(recipe=read(checked('recipe.json')), selection=read(checked('selection.json')),
                test=read(checked('test_summary.json')), histories={}, stop={})
    for p in sorted((RUN/'models').glob('*/history.json')):
        name = str(p.relative_to(RUN)).replace('\\','/')
        data['histories'][p.parent.name] = read(checked(name))
        data['stop'][p.parent.name] = read(checked(name.replace('history.json','result.json')))
    z = np.load(checked('test/clean.npz'))
    cm = np.bincount(z['cls']*3+z['baseline_prob'].argmax(-1), minlength=9).reshape(3,3)
    assert cm.sum()==727
    assert abs(np.trace(cm)/727-data['test']['scenarios']['clean']['baseline']['accuracy'])<1e-12
    data['confusion'] = cm.tolist()
    assert data['selection']['selected']=='baseline_mean'
    write(ASSETS/'plot_data.json',data)
    refs=['A/problem2_training_v2/plot_weighted_results.py','B/scripts/plot_q3.py','C/scripts/plot_q2_round2_paper.py']
    write(ASSETS/'source_manifest.json',dict(source_sha256=sources, plot_data_sha256=sha(ASSETS/'plot_data.json'),
        style_references={p:sha(ROOT/p) for p in refs}, policy='Saved, audited predictions only; no model inference or reselection.'))
    return data

def save(fig,name):
    for ext in ('png','svg'):
        fig.savefig(ASSETS/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def decorate(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)

def figures(d):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'path'})
    families=['baseline','plain_ce','low_reg','low_missing','combined']
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['(a) stop 准确率','(b) stop 宏 F1']):
        for i,n in enumerate(families):
            vals=[v['stop']['scenarios'][0][metric] for k,v in d['stop'].items() if k.rsplit('_',1)[0]==n]
            ax.errorbar(np.mean(vals),i,xerr=np.std(vals,ddof=1),fmt='o',capsize=4,color=COLORS[0] if i==0 else COLORS[1])
        ax.set_yticks(range(5),[NAMES[n] for n in families]); ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title); decorate(ax)
    save(fig,'fig1_stop_ablation')
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout='constrained')
    labels=['baseline_mean','combined_mean','frozen_mean','tuned_mean','frozen_mean_bias']
    texts=['基线等权','联合调整等权','冻结等权','微调等权','冻结等权＋偏置']
    for j,(ax,title) in enumerate(zip(axes,['(a) 干净准确率','(b) 干净宏 F1','(c) 缺失平均宏 F1'])):
        vals=[]
        for n in labels:
            r=d['selection']['candidates'][n]['metrics']['scenarios']
            vals.append(r[0]['accuracy'] if j==0 else r[0]['macro_f1'] if j==1 else np.mean([x['macro_f1'] for x in r[1:]]))
        ax.scatter(vals,range(5),c=[COLORS[0]]+[COLORS[2]]*4,s=45)
        ax.axvline(vals[0],color=COLORS[0],ls=':',lw=1)
        if j: ax.axvline(vals[0]-.01,color='#C25553',ls='--',lw=1,label='退化下限'); ax.legend(loc='lower left',fontsize=8)
        ax.set_yticks(range(5),texts if j==0 else ['']*5); ax.set_ylim(4.7,-.7)
        ax.set_title(title); ax.xaxis.set_major_locator(MaxNLocator(4)); ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1)); decorate(ax)
    save(fig,'fig2_validation_tradeoff')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for family,color in zip(['baseline','combined','frozen','tuned'],COLORS):
        h=[v for k,v in d['histories'].items() if k.rsplit('_',1)[0]==family]
        for ax,kind in zip(axes,['loss','accuracy']):
            values=np.array([[r['loss'] if kind=='loss' else r['clean']['accuracy'] for r in rows] for rows in h])
            mean,sd=values.mean(0),values.std(0,ddof=1)
            ax.plot(range(1,19),mean,color=color,label=NAMES[family],lw=1.6)
            ax.fill_between(range(1,19),mean-sd,mean+sd,color=color,alpha=.12)
    for ax in axes: decorate(ax); ax.set_xticks([1,3,6,9,12,15,18]); ax.set_xlabel('训练 epoch')
    axes[0].set_title('(a) 训练目标（各配方权重不同）'); axes[0].set_ylabel('损失'); axes[0].legend(fontsize=8)
    axes[1].set_title('(b) stop 准确率'); axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].set_ylabel('Accuracy')
    save(fig,'fig3_learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    modes=['text','audio','vision','text-audio','text-vision','audio-vision','text-audio-vision']
    names=['文本','音频','视觉','文本＋音频','文本＋视觉','音频＋视觉','三模态']
    for mode,name,color,marker in zip(modes,names,[*COLORS,'#C25553','#7F7F7F','#A1885F'],['o','s','^','D','v','P','X']):
        for ax,metric in zip(axes,['accuracy','macro_f1']):
            values=[d['test']['scenarios']['clean']['baseline'][metric]]+[np.mean([d['test']['scenarios'][f'{mode}_{r:02d}_{p}']['baseline'][metric] for p in ['start','middle','end']]) for r in [10,30,50]]
            ax.plot([0,.1,.3,.5],values,marker=marker,color=color,label=name,markersize=4)
    for ax,title in zip(axes,['(a) 准确率','(b) 宏 F1']):
        decorate(ax); ax.set_title(title); ax.set_xlabel('人工缺失比例'); ax.set_xticks([0,.1,.3,.5]); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8,ncol=2,loc='lower left')
    save(fig,'fig4_missing_rates')
    fig,ax=plt.subplots(figsize=(6.2,4.6),layout='constrained')
    cm=np.array(d['confusion']); pct=cm/cm.sum(1,keepdims=True)
    im=ax.imshow(pct,cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{cm[i,j]}\n{pct[i,j]:.1%}',ha='center',va='center',color='white' if pct[i,j]>.55 else '#1C2A39',fontsize=12)
    ax.set_xticks(range(3),['负向','中性','正向']); ax.set_yticks(range(3),['负向','中性','正向'])
    ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别'); ax.set_title('锁定基线：干净测试集（727条）')
    fig.colorbar(im,ax=ax,format=PercentFormatter(1),label='真实类别内比例',shrink=.85)
    save(fig,'fig5_test_confusion')
    with (ASSETS/'plot_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','valid_accuracy','valid_macro_f1','missing_mean_macro_f1','eligible'])
        for n,r in d['selection']['candidates'].items():
            s=r['metrics']['scenarios'];w.writerow([n,s[0]['accuracy'],s[0]['macro_f1'],np.mean([v['macro_f1'] for v in s[1:]]),r['eligible']])
    write(ASSETS/'render_manifest.json',dict(script_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in ASSETS.glob('fig*.*')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    ASSETS.mkdir(parents=True,exist_ok=True)
    d=snapshot() if a.snapshot or not (ASSETS/'plot_data.json').exists() else read(ASSETS/'plot_data.json')
    assert sha(ASSETS/'plot_data.json')==read(ASSETS/'source_manifest.json')['plot_data_sha256']
    figures(d)
    print('Rendered five figures as 300 dpi PNG and SVG.')
