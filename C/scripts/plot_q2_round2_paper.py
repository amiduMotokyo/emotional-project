"""Archive measured round-two plot data and render paper figures; no training.

First run reads immutable experiment artifacts. Subsequent runs can render solely
from paper/assets/round2/plot_data.json, without any raw data or model files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/C/paper/assets/round2'
RUN = ROOT / 'C/outputs/q2_round2_ensemble_v1'
COLORS = {'R0': '#355C7D', 'R2': '#6C9A8B', 'R6': '#D8933D'}
NAMES = {'R0': 'R0 固定单模型', 'R2': 'R2 等权集成', 'R6': 'R6 正则DE'}
MODES = ['text', 'audio', 'vision', 'text-audio', 'text-vision', 'audio-vision', 'text-audio-vision']
STYLE_SOURCES = ['A/problem2_training_v2/plot_weighted_results.py', 'B/scripts/plot_q3.py',
                 'docs/A/paper/problem1_section_5_2.md', 'docs/B/paper/q3_explain_model.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def snapshot():
    expected = {}
    for marker in (RUN / 'state').glob('*.json'):
        for path, digest in read(marker)['artifacts'].items():
            resolved = str(Path(path).resolve())
            if resolved in expected and expected[resolved] != digest:
                raise ValueError('Conflicting artifact provenance')
            expected[resolved] = digest
    provenance = read(RUN / 'report_provenance.json')
    expected[str((RUN / 'bootstrap.json').resolve())] = provenance['sources']['bootstrap.json']
    sources = {}

    def checked(name):
        path = RUN / name
        actual = sha(path)
        if actual != expected.get(str(path.resolve())):
            raise ValueError(f'Unverified source: {name}')
        sources[name] = actual
        return path

    results = read(checked('crossfit_summary.json'))['results']
    intervals = read(checked('bootstrap.json'))
    grid = read(checked('test_grid.json'))['scenarios']
    solutions = read(checked('final_ensembles.json'))['ensembles']['R6']
    locked = read(checked('lock.json'))
    assert locked['selected'] == 'R6' and len(grid) == 64
    output = dict(oof=results, bootstrap=intervals, test={}, curves={}, confusion={},
                  selected='R6', weights=np.mean([s['weights'] for s in solutions], axis=0).tolist(),
                  model_ids=['A_L_20261001', 'A_L_20261002', 'A_D_20261001',
                             'A_D_20261002', 'A_H_20261001', 'A_H_20261002'])
    for method in COLORS:
        output['test'][method] = dict(mean_S=float(np.mean([
            v['metrics'][method]['raw']['selection_score'] for v in grid.values()])),
            clean=grid['clean']['metrics'][method]['raw']['scenarios'][0])
        output['curves'][method] = {}
        for mode in MODES:
            output['curves'][method][mode] = [output['test'][method]['clean']['macro_f1']] + [
                float(np.mean([grid[f'{mode}_{rate:02d}_{pos}']['metrics'][method]['raw']['scenarios'][0]['macro_f1']
                               for pos in ('start', 'middle', 'end')])) for rate in (10, 30, 50)]
    with np.load(checked('evaluation/test/clean.npz')) as prediction:
        labels = prediction['cls']
        assert len(labels) == len(np.unique(prediction['sample_id'])) == 727
        for method in COLORS:
            predicted = prediction[method+'_probs'][0].argmax(-1)
            matrix = np.bincount(3*labels+predicted, minlength=9).reshape(3, 3)
            assert matrix.sum() == 727
            assert abs(np.trace(matrix)/727 - output['test'][method]['clean']['accuracy']) < 1e-12
            output['confusion'][method] = matrix.tolist()
    assert abs(sum(output['weights']) - 1) < 1e-12 and min(output['weights']) > 0
    write(ASSETS / 'plot_data.json', output)
    write(ASSETS / 'source_manifest.json', dict(run=str(RUN.relative_to(ROOT)), source_sha256=sources,
        plot_data_sha256=sha(ASSETS / 'plot_data.json'),
        style_references={p: sha(ROOT/p) for p in STYLE_SOURCES},
        policy='Read-only saved predictions; no retraining, refitting, new test inference or reselection.'))
    return output


def save(fig, name):
    for suffix in ('png', 'svg'):
        fig.savefig(ASSETS/f'{name}.{suffix}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def decorate(ax, axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=axis, color='#E5E8EB', linewidth=.7)
    ax.set_axisbelow(True)


def figures(data):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'legend.fontsize': 9, 'svg.fonttype': 'path'})
    # Scores are point plots on explicitly local scales, not truncated bars.
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), layout='constrained')
    methods = [f'R{i}' for i in range(8)]
    for library, offset, color, marker in [('A', -.12, '#355C7D', 'o'), ('B', .12, '#6C9A8B', 's')]:
        values = [data['oof'][library][m]['oof']['selection_score'] for m in methods]
        axes[0].plot(values, np.arange(8)+offset, linestyle='none', marker=marker,
                     color=color, label=f'库{library}', markersize=6)
        delta = np.array([values[i]-values[0] for i in range(1, 8)])
        bounds = np.array([data['bootstrap'][library][m]['delta_ci95'] for m in methods[1:]])
        axes[1].errorbar(delta, np.arange(7)+offset, xerr=[delta-bounds[:,0], bounds[:,1]-delta],
                         fmt=marker, color=color, capsize=3, markersize=5, label=f'库{library}')
    for ax, labels in zip(axes, [methods, methods[1:]]):
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        decorate(ax, 'x')
    axes[0].set(xlim=(.45,.525), xlabel='四场景折外综合分数 S（局部刻度）', title='(a) 两套模型库的折外比较')
    axes[0].legend(loc='lower left')
    axes[1].axvline(0, color='#89939B', linestyle='--', linewidth=1)
    axes[1].set(xlabel='相对同库 R0 的 ΔS', title='(b) 视频簇配对 bootstrap：95%区间')
    save(fig, 'fig1_oof_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), layout='constrained')
    for i, method in enumerate(COLORS):
        mean = data['test'][method]['mean_S']
        acc = data['test'][method]['clean']['accuracy']
        axes[0].scatter(mean, i, color=COLORS[method], s=75, zorder=3)
        axes[0].annotate(f'{mean:.4f}', (mean,i), xytext=(8,0), textcoords='offset points', va='center')
        axes[1].scatter(acc, i, color=COLORS[method], s=75, zorder=3)
        axes[1].annotate(f'{acc:.2%}', (acc,i), xytext=(8,0), textcoords='offset points', va='center')
    for ax in axes:
        ax.set_yticks(range(3), list(NAMES.values()))
        ax.set_ylim(2.5,-.5)
        decorate(ax,'x')
    axes[0].set(xlim=(.497,.514), xlabel='64条件平均 S（局部刻度）', title='(a) 锁定后 test：完整缺失面板')
    axes[1].set(xlim=(.635,.66), xlabel='准确率（局部刻度）', title='(b) 同一 test：clean 准确率')
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    save(fig,'fig2_test_tradeoff')

    fig, axes = plt.subplots(2,4,figsize=(12.4,6.0),layout='constrained',sharex=True,sharey=True)
    titles=['T 文本','A 音频','V 视觉','T+A','T+V','A+V','T+A+V']
    all_values=np.array([data['curves'][m][s] for m in COLORS for s in MODES])
    ymin, ymax = np.floor(all_values.min()*100)/100-.005, np.ceil(all_values.max()*100)/100+.005
    for ax, mode, title in zip(axes.flat,MODES,titles):
        for method, marker in zip(COLORS, ['o','s','^']):
            ax.plot([0,.1,.3,.5], data['curves'][method][mode], color=COLORS[method],
                    marker=marker, linewidth=1.7, markersize=4, label=NAMES[method])
        ax.set_title(title)
        ax.set_xticks([0,.1,.3,.5],['0','10%','30%','50%'])
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_ylim(ymin,ymax)
        decorate(ax)
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
    axes.flat[-1].text(.5,.18,'每个非零缺失率平均三个位置\n0 表示共同的 clean 条件',ha='center',transform=axes.flat[-1].transAxes,fontsize=9)
    fig.supxlabel('人工缺失比例')
    fig.supylabel('test Macro-F1（局部刻度）')
    save(fig,'fig3_missing_rates')

    fig, axes=plt.subplots(1,3,figsize=(12.0,4.1),layout='constrained')
    vmax=max(np.max(data['confusion'][m]) for m in COLORS)
    for ax,method in zip(axes,COLORS):
        matrix=np.array(data['confusion'][method])
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=vmax)
        for row in range(3):
            for col in range(3):
                ax.text(col,row,f'{matrix[row,col]}\n{matrix[row,col]/matrix[row].sum():.1%}',
                    ha='center',va='center',color='white' if matrix[row,col]>.55*vmax else '#1C2A39',fontsize=10)
        ax.set_xticks(range(3),['负向','中性','正向'])
        ax.set_yticks(range(3),['负向','中性','正向'])
        ax.set_xlabel('预测类别')
        ax.set_title(f'{NAMES[method]}\nACC={data["test"][method]["clean"]["accuracy"]:.2%}')
    axes[0].set_ylabel('真实类别')
    fig.colorbar(im,ax=axes,shrink=.8,label='样本数（统一色标）')
    save(fig,'fig4_test_confusion')

    fig,ax=plt.subplots(figsize=(8.5,3.8),layout='constrained')
    weights=data['weights']
    bars=ax.bar(range(6),weights,color=COLORS['R6'],width=.65)
    ax.axhline(1/6,color=COLORS['R2'],linestyle='--',linewidth=1.5,label='等权参考 1/6')
    ax.set_xticks(range(6),['L / 种子1','L / 种子2','D / 种子1','D / 种子2','H / 种子1','H / 种子2'])
    ax.set(ylim=(0,.67),ylabel='最终平均权重',title='R6：库A全valid重拟合后的权重')
    for bar,value in zip(bars,weights):
        ax.text(bar.get_x()+bar.get_width()/2,value+.012,f'{value:.4f}',ha='center',fontsize=10)
    ax.legend(loc='upper left')
    decorate(ax)
    save(fig,'fig5_ensemble_weights')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='store_true',help='Verify original run and refresh aggregate snapshot')
    args=parser.parse_args()
    ASSETS.mkdir(parents=True,exist_ok=True)
    if args.snapshot or not (ASSETS/'plot_data.json').exists():
        data=snapshot()
    else:
        manifest=read(ASSETS/'source_manifest.json')
        if sha(ASSETS/'plot_data.json') != manifest['plot_data_sha256']:
            raise ValueError('Archived plot data changed')
        data=read(ASSETS/'plot_data.json')
    figures(data)
    with (ASSETS/'plot_values.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle)
        writer.writerow(['figure','group','method','condition','value'])
        for library in ('A','B'):
            for method,row in data['oof'][library].items():
                writer.writerow(['fig1',library,method,'OOF_S',row['oof']['selection_score']])
        for method in COLORS:
            writer.writerow(['fig2','test',method,'mean_64_S',data['test'][method]['mean_S']])
            writer.writerow(['fig2','test',method,'clean_ACC',data['test'][method]['clean']['accuracy']])
            for mode in MODES:
                for rate,value in zip([0,.1,.3,.5],data['curves'][method][mode]):
                    writer.writerow(['fig3','test',method,f'{mode}_{rate}',value])
        for mid,weight in zip(data['model_ids'],data['weights']):
            writer.writerow(['fig5','A','R6',mid,weight])
    write(ASSETS/'render_manifest.json',dict(script_sha256=sha(__file__),matplotlib=matplotlib.__version__,
        numpy=np.__version__,plot_data_sha256=sha(ASSETS/'plot_data.json'),
        outputs={p.name:sha(p) for p in sorted(ASSETS.glob('fig*.*'))}))
    print('Saved 5 figures as PNG/SVG, aggregate snapshot, CSV and provenance.')


if __name__=='__main__':
    main()
