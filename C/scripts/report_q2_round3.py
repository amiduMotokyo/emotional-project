"""Add paired uncertainty, seed variability and timings to the round-three report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from C.scripts.run_q2_round3 import *


def main():
    run = Run()
    assert read(OUT/'audit.json')['passed']
    run.report()
    original_path = next((ROOT/'data').glob('*/aligned_50.pkl'))
    assert digest(original_path) == read(OLD/'prepared.json')['signature']['source']['aligned_pickle']
    with original_path.open('rb') as f:
        ids = pickle.load(f)['test']['id']
    groups = np.asarray([str(x).split('$_$')[0] for x in ids])
    z = load_npz(OUT/'test/clean.npz')
    delta = (z['selected_prob'].argmax(-1) == z['cls']).astype(float) - (z['baseline_prob'].argmax(-1) == z['cls'])
    unique = np.unique(groups)
    sums = np.asarray([delta[groups == g].sum() for g in unique])
    sizes = np.asarray([np.count_nonzero(groups == g) for g in unique])
    rng = np.random.default_rng(20262010)
    bootstrap = []
    for _ in range(2000):
        drawn = rng.integers(0, len(unique), len(unique))
        bootstrap.append(float(sums[drawn].sum()/sizes[drawn].sum()))
    uncertainty = dict(accuracy_delta=float(delta.mean()), correct_count_delta=int(delta.sum()),
         paired_video_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(), replicates=2000,
         videos=len(unique), note='Conditional on locked fitted models; not full training/selection uncertainty. Reused test.')
    result_files = list((OUT/'models').glob('*/result.json'))
    families = {}
    for path in result_files:
        r = read(path)
        family = path.parent.name.rsplit('_', 1)[0]
        families.setdefault(family, []).append(r)
    variability = {}
    learning_curves = {}
    for family, rows in families.items():
        acc = [r['stop']['scenarios'][0]['accuracy'] for r in rows]
        f1 = [r['stop']['scenarios'][0]['macro_f1'] for r in rows]
        variability[family] = dict(stop_accuracy_mean=float(np.mean(acc)), stop_accuracy_sd=float(np.std(acc, ddof=1)),
            stop_f1_mean=float(np.mean(f1)), stop_f1_sd=float(np.std(f1, ddof=1)),
            best_epochs=[r['best_epoch'] for r in rows])
        curves = []
        for row in rows:
            history = read(OUT/'models'/row['name']/'history.json')
            best, last = history[row['best_epoch']-1], history[-1]
            curves.append([best['loss'], last['loss'], best['clean']['accuracy'], last['clean']['accuracy']])
        learning_curves[family] = dict(zip(('best_loss', 'last_loss', 'best_stop_accuracy', 'last_stop_accuracy'),
                                          np.asarray(curves).mean(0).tolist()))
    times = {p.stem: read(p)['seconds'] for p in (OUT/'state').glob('*.json')}
    train_int8 = sum(times[p.parent.name] for p in result_files if read(p)['kind'] == 'int8')
    train_text = sum(times[p.parent.name] for p in result_files if read(p)['kind'] != 'int8')
    # Validation predictions were completed in the initial select attempt before
    # its JSON serialization failure; the resumed select timer excludes them.
    prediction_cache = sum(v for n, v in times.items() if n.startswith('valid_'))
    total = sum(times[n] for n in ('prepare', 'recipe', 'select', 'evaluate')) + train_int8 + train_text + prediction_cache
    evidence = dict(bootstrap=uncertainty, seed_variability=variability, learning_curves=learning_curves,
                   seconds=dict(int8_train=train_int8, text_train=train_text, valid_prediction_cache=prediction_cache, main_stages=total))
    write(OUT/'analysis.json', evidence)
    path = ROOT/'docs/C/experiments/第三轮准确率优化实验结果.md'
    ci = uncertainty['paired_video_bootstrap_95']
    text = ['\n## 种子波动与复评不确定性', '', '| 配置 | stop Accuracy 均值±标准差 | stop 宏F1 均值±标准差 | 选中 epoch |', '|---|---:|---:|---|']
    for family, r in variability.items():
        text.append(f"| {family} | {r['stop_accuracy_mean']:.4f} ± {r['stop_accuracy_sd']:.4f} | {r['stop_f1_mean']:.4f} ± {r['stop_f1_sd']:.4f} | {r['best_epochs']} |")
    text += ['', f"测试干净准确率相对本轮 baseline 等权集成变化 {uncertainty['accuracy_delta']*100:+.2f} 个百分点（多答对 {uncertainty['correct_count_delta']} 条）。按 {len(unique)} 个视频配对 bootstrap 2000次，95%区间 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] 个百分点。该区间只反映固定模型的测试抽样波动，不包含训练和方法选择的不确定性。", '',
             f"实测五组 int8 训练与 stop 评估 {train_int8/60:.2f} 分钟；全精度冻结/微调训练与 stop 评估 {train_text/60:.2f} 分钟；主阶段合计 {total/60:.2f} 分钟。不含开发、依赖修复、独立审计及报告。", '',
             '审计检查21个模型、378个 epoch、冻结层权重不变、微调层确实更新、视频分组隔离、折外校准重算、64条件逐样本指标及锁定哈希。详细记录见 `audit.json`。', '']
    if read(OUT/'lock.json')['selected'] == 'baseline_mean':
        text += ['本轮最终保留 baseline_mean，selected 与 baseline 是同一组权重和解码。上述零差异、零宽区间由这个身份关系决定，不是对两个不同方法进行等效性验证。', '']
        text += ['## 结果解释与下一轮边界', '',
                 '这套路线未取得可接受的准确率提升。stop 上联合配方的小幅收益没有迁移到 valid；顶部两层微调也没有超过同条件冻结模型的干净准确率。不能据此声称微调普遍无效，只能否定本轮配方、学习率、冻结范围和预算下的收益。', '',
                 '| 三种子等权候选 | valid Accuracy | valid 宏F1 | 三种缺失平均宏F1 |', '|---|---:|---:|---:|']
        candidates = read(OUT/'selection.json')['candidates']
        for name in ('baseline_mean', 'combined_mean', 'frozen_mean', 'tuned_mean', 'frozen_mean_bias'):
            s = candidates[name]['metrics']['scenarios']
            text.append(f"| {name} | {s[0]['accuracy']:.4f} | {s[0]['macro_f1']:.4f} | {np.mean([r['macro_f1'] for r in s[1:]]):.4f} |")
        r = learning_curves['baseline']
        text += ['',
                 '全精度冻结+偏置的准确率61.81%略高于基线61.54%，但干净宏F1及缺失平均宏F1分别降低约1.30、2.86个百分点，超过预先固定的1个百分点限制，因此没有采用。即使只看准确率，这也只是内部验证多答对2条，并不是已证实的泛化提升。', '',
                 f"过拟合迹象明确：baseline 在最佳 epoch 时平均训练损失 {r['best_loss']:.3f}、stop Accuracy {r['best_stop_accuracy']:.2%}；到18 epoch，损失降到 {r['last_loss']:.3f}，stop Accuracy却降到 {r['last_stop_accuracy']:.2%}。本轮已经按stop保留早期权重，因此仅缩短训练只能节省时间，不能把已经保留的最佳模型再变好。", '',
                 '后续如另开实验，优先检验分阶段优化（先稳定融合头，再以更小文本学习率解冻）、更强正则或更稳定的分组模型选择；这些是本轮结果引出的待验证假设，尚未实施。不能把增加 epoch 或继续扩大权重搜索视为已有证据支持的改进。', '',
                 '本轮 baseline_mean 只是此实验的保底候选，不等于优于历史最强模型；正式提交模型和第二轮产物均未替换。', '']
    path.write_text(path.read_text(encoding='utf-8')+'\n'.join(text), encoding='utf-8')
    archive = ROOT/'docs/C/experiments/assets/round3'
    archive.mkdir(parents=True, exist_ok=True)
    import shutil
    for name in ('analysis.json', 'audit.json', 'recipe.json', 'selection.json', 'lock.json', 'test_summary.json', 'manifest.json', 'preparation.json', 'preflight.json', 'serialization_adapter.json'):
        shutil.copy2(OUT/name, archive/name)
    write(archive/'source_manifest.json', {name: digest(OUT/name) for name in
          ('analysis.json','audit.json','recipe.json','selection.json','lock.json','test_summary.json','manifest.json','preparation.json','preflight.json','serialization_adapter.json')})
    print(evidence)


if __name__ == '__main__':
    main()
