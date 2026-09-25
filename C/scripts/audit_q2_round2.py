"""Audit completed real artifacts, identity isolation and saved OOF predictions."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from C.scripts.run_q2_optimization import read, digest, write
from C.src.ensemble_search import predict_mean, metrics, select_method


def main():
    out = ROOT / 'C/outputs/q2_round2_ensemble_v1'
    assert (out / 'state/P6.json').exists()
    run_hash = digest(out / 'execution_manifest.json')
    checked = {}
    for state_path in (out / 'state').glob('*.json'):
        state = read(state_path)
        assert state['run_hash'] == run_hash
        for path, expected in state['artifacts'].items():
            if path not in checked:
                checked[path] = digest(path)
            assert checked[path] == expected, path
    splits = read(out / 'splits.json')
    fit_ids, stop_ids = set(splits['fit_ids']), set(splits['stop_ids'])
    assert not fit_ids & stop_ids and len(fit_ids | stop_ids) == 3395
    train_groups = np.array(splits['source_metadata']['train']['group_ids'])
    assert not set(train_groups[splits['fit_indices']]) & set(train_groups[splits['stop_indices']])
    groups = np.array(splits['source_metadata']['valid']['group_ids'])
    folds = np.array(splits['valid_fold'])
    for fold in range(3):
        assert not set(groups[folds == fold]) & set(groups[folds != fold])
    training = [read(p) for p in (out / 'models').glob('*/result.json')]
    assert len(training) == 12 and sum(r['epochs_completed'] for r in training) == 216
    requests, solutions_count = 0, 0
    for library in ('A', 'B'):
        with np.load(out / f'predictions_{library}.npz') as archive:
            data = {k: archive[k] for k in archive.files}
        np.testing.assert_allclose(data['probs'].sum(-1), 1, atol=2e-7)
        for method in [f'R{i}' for i in range(8)]:
            result = read(out / 'search' / f'{library}_{method}.json')
            with np.load(out / 'search' / f'{library}_{method}.npz') as archive:
                oof = {k: archive[k] for k in archive.files}
            np.testing.assert_array_equal(oof['sample_id'], data['sample_id'])
            seen = np.zeros(728, dtype=int)
            for row in result['folds']:
                idx = np.flatnonzero(folds == row['fold'])
                seen[idx] += 1
                assert set(row['held_ids']) == set(data['sample_id'][idx])
                assert set(row['fit_ids']).isdisjoint(row['held_ids'])
                pp, rr = predict_mean(row['solutions'], data['probs'][:, :, idx], data['reg'][:, :, idx])
                np.testing.assert_allclose(pp, oof['probs'][:, idx], rtol=0, atol=1e-12)
                np.testing.assert_allclose(rr, oof['reg'][:, idx], rtol=0, atol=1e-12)
                if method in ('R3','R4','R5','R6','R7'):
                    assert len(row['solutions']) == 3
                    for solution in row['solutions']:
                        assert solution['requests'] == len(solution['trace']) == 300
                        assert solution['requests'] == solution['actual_objective_calls'] + solution['cache_hits']
                        requests += solution['requests']
                        solutions_count += 1
            np.testing.assert_array_equal(seen, np.ones(728, dtype=int))
            recomputed = metrics(oof['probs'], oof['reg'], data['cls'], data['score'])
            np.testing.assert_allclose(recomputed['selection_score'], result['oof']['selection_score'], rtol=0, atol=1e-12)
    assert requests == 27000 and solutions_count == 90
    locked = read(out / 'lock.json')
    chosen = select_method(read(out / 'crossfit_summary.json')['results'])
    assert chosen['selected'] == locked['selected'] and chosen['checks'] == locked['checks']
    for split in ('valid','test'):
        grid = read(out / f'{split}_grid.json')
        assert grid['complete'] and len(grid['scenarios']) == 64
        assert grid['lock_hash'] == digest(out / 'lock.json')
    write(out / 'audit.json', dict(passed=True, unique_artifacts_checked=len(checked),
         training_runs=12, training_epochs=216, search_runs=90, search_requests=requests,
         grouping='video_disjoint_within_train_and_valid_folds',
         saved_oof_predictions='reconstructed_from_saved_solutions_equal_at_1e-12',
         selection='recomputed_matches_lock', evaluation='64_conditions_each_valid_test'))
    print('PASS real audit:', len(checked), 'artifacts; 216 epochs; 27000 requests; OOF reconstructed; video groups disjoint; lock and 64-condition panels verified.')


if __name__ == '__main__':
    main()
