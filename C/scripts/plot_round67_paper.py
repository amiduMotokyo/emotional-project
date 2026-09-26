"""Verified saved-result snapshots and publication figures for rounds six/seven."""
import sys,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from C.scripts.plot_q2_round3_paper import decorate
ASSETS=ROOT/'docs/C/paper/assets/round67'
RUNS={6:ROOT/'C/outputs/q3_round6_minilm_v2',7:ROOT/'C/outputs/q3_round7_representation_v1'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ASSETS.mkdir(parents=True,exist_ok=True);data={};sources={};rows=[]
    for round_,root in RUNS.items():
        expected={}
        for f in (root/'state').glob('*.json'):expected.update(read(f)['artifacts'])
        def checked(name):
            f=root/name;assert sha(f)==expected[name],name;sources[str(f.relative_to(ROOT))]=sha(f);return f
        selected=read(checked('selection.json'));lock=read(checked('lock.json'));grid=read(checked('final_grid.json'))
        d=dict(selection=selected,lock=lock,grid=grid,validation={})
        z=np.load(checked('final_grid.npz'))
        cm=np.bincount(z['cls']*3+z['prob'][0].argmax(-1),minlength=9).reshape(3,3)
        assert cm.sum()==728 and abs(cm.trace()/728-grid['metrics']['scenarios'][0]['accuracy'])<1e-12
        d['confusion']=cm.tolist()
        for g in lock['refit_groups']:
            cal=read(checked(f'calibration/{g}.json'));vals={'FP32':[],'int8':[],'校准后':[]}
            for seed in [20267001,20267002,20267003]:
                fp=read(checked(f'valid/fp32_{g}_{seed}.json'))['scenarios'][0]['accuracy']
                raw=read(checked(f'deploy/{g}_{seed}/valid.json'))['scenarios'][0]['accuracy']
                a=np.load(checked(f'deploy/{g}_{seed}/valid.npz'));pred=(np.log(a['prob'][0].clip(1e-12))+cal['class_bias']).argmax(-1)
                acc=float(np.mean(pred==a['cls']))
                for k,v in zip(vals,[fp,raw,acc]):vals[k].append(v);rows.append([round_,g,'valid',k,seed,v])
            d['validation'][g]=dict(accuracy=vals,calibration=cal)
        for g,metrics in selected['per_seed'].items():
            for seed,m in zip([20267001,20267002,20267003],metrics):rows.append([round_,g,'stop','FP32',seed,m['scenarios'][0]['accuracy']])
        data[str(round_)]=d
    write(ASSETS/'plot_data.json',data)
    with (ASSETS/'plot_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['round','group','split','precision','seed','accuracy']);w.writerows(rows)
    refs=['B/scripts/plot_q3.py','docs/A/paper/problem1_section_5_2.md','C/scripts/plot_q2_round5_paper.py']
    write(ASSETS/'source_manifest.json',dict(source_sha256=sources,plot_data_sha256=sha(ASSETS/'plot_data.json'),style_references={p:sha(ROOT/p) for p in refs},policy='Saved results only; no training, test evaluation, or model reselection.'))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'path'})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(ASSETS/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.4),layout='constrained')
    for ax,r in zip(axes,['6','7']):
        s=data[r]['selection'];names=list(s['per_seed']);values=[[v['scenarios'][0]['accuracy']*100 for v in s['per_seed'][g]] for g in names]
        ax.errorbar(range(len(names)),np.mean(values,axis=1),yerr=np.std(values,axis=1,ddof=1),fmt='o',capsize=4,color='#355C7D')
        ax.set_xticks(range(len(names)),names);ax.set_ylim(59,69);ax.set_ylabel('stop准确率（%）');ax.set_title(f'第{r}轮：均值 ± 样本SD');decorate(ax)
    save(fig,'fig1_development')
    fig,ax=plt.subplots(figsize=(8.4,3.6),layout='constrained')
    for g,color in [('A0','#355C7D'),('B2','#6C9A8B')]:
        vals=data['7']['validation'][g]['accuracy'];a=np.array(list(vals.values()))*100
        ax.errorbar(range(3),a.mean(1),yerr=a.std(1,ddof=1),fmt='o-',capsize=4,color=color,label=g)
    ax.set_xticks(range(3),['FP32','int8未校准','int8＋全valid拟合偏置']);ax.set_ylim(57,65);ax.set_ylabel('valid准确率（%）');ax.legend();decorate(ax);save(fig,'fig2_precision')
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.3),layout='constrained')
    va=data['7']['validation'];a=np.array(va['A0']['accuracy']['校准后']);b=np.array(va['B2']['accuracy']['校准后'])
    delta=(b-a)*100;axes[0].bar(['种子1（交付）','种子2','种子3'],delta,color=['#D8933D','#6C9A8B','#6C9A8B']);axes[0].axhline(0,color='#777777',lw=.8)
    for i,v in enumerate(delta):axes[0].text(i,v+(.09 if v>=0 else -.12),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top')
    axes[0].set_ylim(-.85,2.8);axes[0].set_ylabel('B2−A0准确率（百分点）');axes[0].set_title('逐种子部署差异')
    for i,g in enumerate(['A0','B2']):
        ys=[va[g]['calibration'][k]['scenarios'][0]['accuracy']*100 for k in ['baseline','oof','full_fit']]
        axes[1].plot(range(3),ys,'o-',label=g,color=['#355C7D','#6C9A8B'][i])
    axes[1].set_xticks(range(3),['无偏置','折外校准','全valid拟合']);axes[1].set_ylabel('三种子平均准确率（%）');axes[1].legend();axes[1].set_title('校准口径分开报告')
    for ax in axes:decorate(ax)
    save(fig,'fig3_seed_calibration')
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.7),layout='constrained')
    for ax,r,title in zip(axes,['6','7'],['第六轮R0（同第七轮A0）','第七轮B2']):
        cm=np.array(data[r]['confusion']);rate=cm/cm.sum(1,keepdims=True);ax.imshow(rate,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{rate[i,j]:.1%}',ha='center',va='center',color='white' if rate[i,j]>.5 else '#1C2A39')
        ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),['负向','中性','正向']);ax.set_xlabel('预测类别');ax.set_ylabel('真实类别');ax.set_title(title)
    save(fig,'fig4_confusion')
    write(ASSETS/'render_manifest.json',{p.name:sha(p) for p in sorted(ASSETS.glob('fig*'))})
    print('PAPER_SNAPSHOTS_AND_4_FIGURES_OK')

if __name__=='__main__':main()
