"""Archive completed round-two evidence without retraining or changing selection."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from C.scripts.run_q2_optimization import read, write, digest
from C.src.ensemble_search import metrics


def main():
    out = ROOT / 'C/outputs/q2_round2_ensemble_v1'
    if not (out / 'state/P6.json').exists():
        raise ValueError('Complete the full pipeline first')
    summary = read(out / 'crossfit_summary.json')['results']
    lock = read(out / 'lock.json')
    splits = read(out / 'splits.json')
    # All scenario copies of a sample share one resampling index. Selection is
    # frozen; intervals are descriptive and do not trigger any rerun/refit.
    bootstrap = {}
    for library in ('A', 'B'):
        with np.load(out / 'search' / f'{library}_R0.npz') as f:
            reference = {k: f[k] for k in f.files}
        group_ids = np.array(splits['source_metadata']['valid']['group_ids'])
        members = [np.flatnonzero(group_ids == g) for g in np.unique(group_ids)]
        rng = np.random.default_rng(20260925)
        draws = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(2000)]
        base = np.array([metrics(reference['probs'][:, idx], reference['reg'][:, idx],
            reference['cls'][idx], reference['score'][idx])['selection_score'] for idx in draws])
        bootstrap[library] = {}
        for method in [f'R{i}' for i in range(1, 8)]:
            with np.load(out / 'search' / f'{library}_{method}.npz') as f:
                data = {k: f[k] for k in f.files}
            assert np.array_equal(data['sample_id'], reference['sample_id'])
            assert np.array_equal(data['cls'], reference['cls'])
            values = np.array([metrics(data['probs'][:, idx], data['reg'][:, idx],
                data['cls'][idx], data['score'][idx])['selection_score'] for idx in draws]) - base
            bootstrap[library][method] = dict(delta_ci95=np.quantile(values, [.025, .975]).tolist(), replicates=2000)
        print('bootstrap complete', library, flush=True)
    write(out / 'bootstrap.json', bootstrap)
    lines = (out / 'report.md').read_text(encoding='utf-8').splitlines()
    lines[4:4] = ['可携带的表格与审计依据见[汇总证据](assets/round2/README.md)，不需要复制原始数据。', '']
    position = lines.index('## 锁定后64场景复评') + 1
    lines[position:position] = ['', 'valid网格使用在全部valid重拟合后的集成权重，是拟合集压力测试，不是折外评价；前表的三折拼接结果才是折外指标。test网格只用于锁定后复评。']
    lines += ['', '## 数据、复现与审计', '',
              f"官方train 3395划分为fit {len(splits['fit_indices'])}、stop {len(splits['stop_indices'])}；valid 728三折。",
              f"fit类别计数为{splits['counts']['fit']}，stop为{splits['counts']['stop']}；负/中/正映射为0/1/2。",
              '使用原始行编号关联新生成的fit视图，不复用全train归一化；两个库共享同一划分。',
              '补查原始ID元数据后，采用视频前缀分组：训练内部10折中预设取第0折为stop，valid使用3折StratifiedGroupKFold。原始标签数组通过元数据占位对象隔离，锁定前不还原原始test数值标签。',
              f"官方train/valid共享视频前缀数：{len(splits['official_train_valid_shared_groups'])}。视频分组不等于主体分组。",
              '准备阶段发现B数据接口文件哈希与历史记录不一致，最初预检停止、未训练。随后逐条重编码全部train/valid，要求文本及掩码精确相等才复用缓存；审计记录为source_compatibility.json。', '',
              '训练前修订R7初始化：前7个锚点偏置0，后5个随机点使用独立seed+100000的U(-0.15,0.15)，避免全零偏置在DE中永远不变。预算不变。行级划分准备与默认ONNX多线程准备均在训练前中止，保留superseded目录；最终仅分组版进入正式预算。', '',
              '运行环境见execution_manifest.json；固定设计参数见C/configs/q2_round2_design.json。', '',
              '```powershell', "& 'C:/Users/ken/miniconda3/envs/myenv/python.exe' C/scripts/check_q2_round2.py",
              "& 'C:/Users/ken/miniconda3/envs/myenv/python.exe' -u C/scripts/run_q2_round2.py --phase all",
              "& 'C:/Users/ken/miniconda3/envs/myenv/python.exe' -u C/scripts/report_q2_round2.py", '```', '',
              '分阶段入口为prepare/train/cache/search/lock/evaluate/report。相同命令会核验并跳过完成任务；配置、源码或产物变化时拒绝复用。test仅在P4锁定核验后加载。', '',
              '## 优化器贡献与消融', '', '| 比较 | 库A ΔS | 库B ΔS | 两库平均ΔS |', '|---|---:|---:|---:|']
    for target, ref, title in [('R2','R0','等权集成−固定单模型'), ('R5','R4','DE−随机搜索'),
                             ('R5','R3','DE−贪心'), ('R6','R5','正则DE−DE'), ('R7','R6','偏置−无偏置')]:
        delta = [summary[l][target]['oof']['selection_score']-summary[l][ref]['oof']['selection_score'] for l in ('A','B')]
        lines.append(f'| {title} | {delta[0]:+.6f} | {delta[1]:+.6f} | {np.mean(delta):+.6f} |')
    lines += ['', '集成超过单模型不能单独证明DE优越；需要同时超过等预算随机与贪心基线。', '',
              '## 拟合与折外差距、搜索种子波动', '',
              '折内拟合S取三折拟合集指标的平均；折外S由拼接728条预测计算，二者差值为描述性泛化差距。种子标准差用ddof=1，搜索种子不等于独立训练重复。', '',
              '| 库 | 方法 | 折内拟合S均值 | 拼接折外S | 三搜索种子折外S标准差 | 非零模型数均值 | 有效模型数均值 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for l in ('A','B'):
        for m in [f'R{i}' for i in range(8)]:
            row = read(out / 'search' / f'{l}_{m}.json')
            fit_score = np.mean([x['fit_metrics']['selection_score'] for x in row['folds']])
            seed_scores = [s['selection_score'] for s in row['seed_oof']]
            std = float(np.std(seed_scores, ddof=1)) if len(seed_scores)>1 else 0.
            lines.append(f"| {l} | {m} | {fit_score:.6f} | {row['oof']['selection_score']:.6f} | {std:.6f} | {row['mean_nonzero_models']:.3f} | {row['mean_effective_models']:.3f} |")
    lines += ['', '非零模型数按每折三个搜索解权重的并集计算，再在折及库间平均用于预设平局规则；有效模型数按平均权重计算，仅为权重集中程度描述。R7仍逐搜索解先应用偏置再平均预测。', '',
              '## 配对bootstrap（描述性）', '',
              '每库2000次视频簇级配对抽样，四场景同步；95%百分位区间只反映固定折外预测的样本不确定性，不覆盖重新训练、搜索和方法选型。不能作为全流程无偏显著性证据。', '',
              '| 库 | 方法对R0 | ΔS的95%区间 |', '|---|---|---|']
    for l in ('A','B'):
        for m, result in bootstrap[l].items():
            lo, hi = result['delta_ci95']
            lines.append(f'| {l} | {m} | [{lo:+.6f}, {hi:+.6f}] |')
    lines += ['', '## 选型规则逐项结果', '', '| 方法 | 两库clean限制通过 | 两库平均ΔS | 达到全部门槛 |',
              '|---|---|---:|---|']
    for m, check in lock['checks'].items():
        lines.append(f"| {m} | {check['clean_ok']} | {check['mean_delta']:+.6f} | {check['eligible']} |")
    test_grid = read(out / 'test_grid.json')['scenarios']
    lines += ['', '## 结果解释', '',
              '等权集成相对固定默认单模型在两库均改善；普通DE并未在两库平均S上超过随机搜索，因此本轮不支持“DE本身优于简单搜索”的结论。',
              'R6相对R5的平均S增益约0.0029，两个库方向均为正，但幅度小；这是正则化有益的初步内部验证证据，不是强显著性结论。',
              'R7两库平均S比R6约高0.00083，位于预设0.002平局范围。按实际非零模型数规则锁定R6，没有按test改选。']
    if lock['selected']:
        chosen = lock['selected']
        baseline = test_grid['clean']['metrics']['R0']['raw']['scenarios'][0]
        candidate = test_grid['clean']['metrics'][chosen]['raw']['scenarios'][0]
        delta_grid = np.mean([v['metrics'][chosen]['raw']['selection_score'] - v['metrics']['R0']['raw']['selection_score'] for v in test_grid.values()])
        lines += [f"锁定后clean test：R0 ACC={baseline['accuracy']:.6f}，{chosen} ACC={candidate['accuracy']:.6f}；Macro-F1分别为{baseline['macro_f1']:.6f}/{candidate['macro_f1']:.6f}。",
                  f'全部64条件平均raw S相对R0变化为{delta_grid:+.6f}。clean test退化必须与内部验证收益同时报告，不能声称全面提升。',
                  '可能原因包括对有限valid的集成权重拟合、模型库误差相关性及不同场景的权衡；本轮没有因果消融证实这些原因。不会依据已查看的test调整权重或重新挑选方法。']
    lines += ['', '## 交付边界', '',
              '完整运行产物位于C/outputs/q2_round2_ensemble_v1/。本次交付为实验实现、复现记录及结果，未覆盖现有正式提交包。',
              '若选型未通过，保留第一轮已锁定方案；64条件表中的R0/R2只是第二轮固定参考，不代表替换提交模型。',
              'A的768维文本路线与本轮384维输入一致性路线不同，不能把跨路线准确率差值归因于优化器。', '']
    doc = ROOT / 'docs/C/experiments/第二轮完整优化实验结果.md'
    lines += ['', '实际产物审计见audit.json：逐文件哈希、视频组隔离、216轮、27000次请求、保存解重建折外预测及锁定一致性。']
    doc.write_text('\n'.join(lines), encoding='utf-8')
    write(out / 'report_provenance.json', dict(document=str(doc), document_hash=digest(doc),
         sources={name:digest(out/name) for name in ['crossfit_summary.json','lock.json',
             'valid_grid.json','test_grid.json','bootstrap.json','execution_manifest.json']}))
    print('Archived', doc, flush=True)


if __name__ == '__main__':
    main()
