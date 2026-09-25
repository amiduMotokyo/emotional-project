"""Hash-verified round-five paper snapshots and figures; no model inference."""
import sys
import csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from C.scripts.plot_q2_round3_paper import read,write,sha,decorate
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MaxNLocator
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'C/outputs/q2_round5_experts_v1'
ASSETS=ROOT/'docs/C/paper/assets/round5'
NAMES=dict(wide_control='加宽基线',text_only='文本专家',audio_visual='音视频专家',equal_mix='等权融合',learned_gate='学习门控')
COLORS=['#355C7D','#6C9A8B','#D8933D','#9467A0']

def snapshot():
    expected={}
    for p in (RUN/'state').glob('*.json'):expected.update(read(p)['artifacts'])
    sources={}
    def checked(name):
        p=RUN/name;assert sha(p)==expected[str(p)],name;sources[name]=sha(p);return p
    d={n:read(checked(f'{n}.json')) for n in ['diagnosis','selection','test_summary']}
    d['weights']=[];d['confusion']={}
    for s in [20263001,20263002,20263003]:
        z=np.load(checked(f'valid_predictions/{s}.npz'));d['weights'].append(z['gate_weights'].mean(1).tolist())
    z=np.load(checked('test/clean.npz'))
    for n in ['wide_control','learned_gate']:
        cm=sum(np.bincount(z['cls']*3+z[f'{n}_{s}_prob'].argmax(-1),minlength=9).reshape(3,3) for s in [20263001,20263002,20263003])
        assert cm.sum()==2181 and abs(cm.trace()/2181-d['test_summary']['scenarios']['clean'][n]['mean']['accuracy'])<1e-12
        d['confusion'][n]=cm.tolist()
    write(ASSETS/'plot_data.json',d)
    refs=['A/problem2_training_v2/plot_weighted_results.py','B/scripts/plot_q3.py','C/scripts/plot_q2_round4_paper.py']
    write(ASSETS/'source_manifest.json',dict(source_sha256=sources,plot_data_sha256=sha(ASSETS/'plot_data.json'),style_references={p:sha(ROOT/p) for p in refs}))
    return d

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(ASSETS/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def figures(d):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'path'})
    fig,ax=plt.subplots(figsize=(10,3),layout='constrained');left=np.zeros(3)
    s=[r['scenarios'][0] for r in d['diagnosis']['rows']]
    vals=[[r['text_accuracy']-r['harmed'] for r in s],[r['rescued'] for r in s],[r['harmed'] for r in s],[r['both_wrong'] for r in s]]
    for x,label,color in zip(vals,['两者都对','文本错／音视频对','文本对／音视频错','两者都错'],COLORS):
        ax.barh(range(3),x,left=left,label=label,color=color)
        for i,(v,l) in enumerate(zip(x,left)):ax.text(l+v/2,i,f'{v:.1%}',ha='center',va='center',color='white')
        left+=x
    ax.set_yticks(range(3),['20263001','20263002','20263003']);ax.invert_yaxis();ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('占全部 stop 样本的比例');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.22),fontsize=9);decorate(ax)
    save(fig,'fig1_error_complementarity')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for ax,key,title in zip(axes,['accuracy','macro_f1'],['(a) valid 准确率','(b) valid 宏 F1']):
        for i,n in enumerate(NAMES):
            v=[r['scenarios'][0][key] for r in d['selection']['candidates'][n]['seeds']]
            ax.errorbar(np.mean(v),i,xerr=np.std(v,ddof=1),fmt='o',capsize=4,color=COLORS[1] if n=='learned_gate' else COLORS[0])
        ax.set_yticks(range(5),list(NAMES.values()));ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1));ax.set_title(title);decorate(ax)
    save(fig,'fig2_valid_candidates')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');w=np.array(d['weights'])[:,:,0]
    for i,color in enumerate(COLORS[:3]):axes[0].plot(range(4),w[i],'o-',label=str(20263001+i),color=color)
    axes[0].set_xticks(range(4),['干净','文本缺失','音频缺失','视觉缺失']);axes[0].set_ylim(0,1);axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].set_title('(a) valid 平均文本专家权重');axes[0].legend(fontsize=8)
    for split,color in zip(['valid','test'],COLORS):
        rows=d['selection']['candidates'] if split=='valid' else d['test_summary']['scenarios']['clean']
        a=rows['learned_gate']['seeds'];b=rows['wide_control']['seeds']
        delta=[100*((x['scenarios'][0] if split=='valid' else x)['accuracy']-(y['scenarios'][0] if split=='valid' else y)['accuracy']) for x,y in zip(a,b)]
        axes[1].plot(range(3),delta,'o-',label=split,color=color)
    axes[1].axhline(0,color='gray',ls='--');axes[1].set_xticks(range(3),['20263001','20263002','20263003']);axes[1].set_title('(b) 门控相对基线的准确率变化');axes[1].set_ylabel('百分点');axes[1].legend()
    for ax in axes:decorate(ax)
    save(fig,'fig3_gates_and_seeds')
    fig,axes=plt.subplots(2,4,figsize=(12,5.4),layout='constrained',sharey=True)
    modes=['text','audio','vision','text-audio','text-vision','audio-vision','text-audio-vision'];labels=['文本','音频','视觉','文本＋音频','文本＋视觉','音频＋视觉','三模态']
    results=d['test_summary']['scenarios']
    for ax,mode,label in zip(axes.flat,modes,labels):
        for n,color in zip(['wide_control','learned_gate'],COLORS):
            v=[results['clean'][n]['mean']['accuracy']]+[np.mean([results[f'{mode}_{r:02d}_{p}'][n]['mean']['accuracy'] for p in ['start','middle','end']]) for r in [10,30,50]]
            ax.plot([0,.1,.3,.5],v,'o-',color=color,label=NAMES[n],markersize=3)
        ax.set_title(label);ax.set_xticks([0,.1,.3,.5]);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('人工缺失比例');decorate(ax)
    axes.flat[7].axis('off');axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    save(fig,'fig4_missing_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,n in zip(axes,['wide_control','learned_gate']):
        cm=np.array(d['confusion'][n]);pct=cm/cm.sum(1,keepdims=True);im=ax.imshow(pct,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{pct[i,j]:.1%}',ha='center',va='center',color='white' if pct[i,j]>.55 else '#1C2A39')
        ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set_xlabel('预测类别');ax.set_ylabel('真实类别');ax.set_title(NAMES[n]+'：三种子累计计数')
    fig.colorbar(im,ax=axes,format=PercentFormatter(1),shrink=.8,label='真实类别内比例');save(fig,'fig5_test_confusion')
    with (ASSETS/'plot_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['candidate','accuracy','macro_f1','missing_macro_f1','eligible'])
        for n,r in d['selection']['candidates'].items():
            s=r['mean']['scenarios'];writer.writerow([n,s[0]['accuracy'],s[0]['macro_f1'],np.mean([x['macro_f1'] for x in s[1:]]),r['eligible']])
    write(ASSETS/'render_manifest.json',dict(script_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in ASSETS.glob('fig*.*')}))

if __name__=='__main__':
    ASSETS.mkdir(parents=True,exist_ok=True)
    d=snapshot() if '--snapshot' in sys.argv or not (ASSETS/'plot_data.json').exists() else read(ASSETS/'plot_data.json')
    assert sha(ASSETS/'plot_data.json')==read(ASSETS/'source_manifest.json')['plot_data_sha256'];figures(d)
    print('Rendered 5 verified figures, PNG 300 dpi and SVG.')
