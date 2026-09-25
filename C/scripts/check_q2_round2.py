"""Meaningful numerical/protocol checks; no experiment data or test labels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from C.src.ensemble_search import fit, predict, predict_mean, metrics, select_method


def main():
    rng = np.random.default_rng(401)
    p = rng.random((6, 4, 39, 3))
    p /= p.sum(-1, keepdims=True)
    r = rng.uniform(-3, 3, (6, 4, 39))
    labels = rng.integers(0, 3, 39)
    scores = rng.uniform(-3, 3, 39)
    ids = np.array([f'synthetic:{i}' for i in range(39)])
    uniform = dict(weights=[1/6]*6, bias=[0]*3)
    up, ur = predict(uniform, p, r)
    np.testing.assert_allclose(up, p.mean(0))
    np.testing.assert_allclose(ur, r.mean(0))
    sp, sr = predict(dict(weights=[0, 0, 1, 0, 0, 0]), p, r)
    np.testing.assert_allclose(sp, p[2])
    np.testing.assert_allclose(sr, r[2])
    result = metrics(up, ur, labels, scores)
    for i, row in enumerate(result['scenarios']):
        np.testing.assert_allclose(row['macro_f1'], f1_score(labels, up[i].argmax(-1), labels=[0,1,2], average='macro', zero_division=0))
        np.testing.assert_allclose(row['mae'], mean_absolute_error(scores, ur[i]))
        np.testing.assert_allclose(row['pearson'], np.corrcoef(scores, ur[i])[0,1])
    absent = metrics(np.ones((1, 10, 3))/3, np.zeros((1, 10)), np.zeros(10, dtype=int), np.zeros(10))
    assert absent['scenarios'][0]['constant_pearson']
    np.testing.assert_allclose(absent['scenarios'][0]['macro_f1'], 1/3)
    fitted = {m: fit(m, p, r, labels, scores, ids, 11) for m in [f'R{i}' for i in range(8)]}
    for method, solution in fitted.items():
        pp, rr = predict(solution, p, r)
        np.testing.assert_allclose(pp.sum(-1), 1, atol=1e-12)
        assert np.all(np.isfinite(rr)) and min(solution['weights']) >= 0
        np.testing.assert_allclose(sum(solution['weights']), 1)
        assert solution['requests'] == (1 if method in ('R0','R2') else 6 if method == 'R1' else 300)
        assert solution['actual_objective_calls'] + solution['cache_hits'] == solution['requests']
    for m in ('R5','R6','R7'):
        for a, b in zip(fitted['R4']['trace'][:12], fitted[m]['trace'][:12]):
            np.testing.assert_array_equal(a['weights'], b['weights'])
    greedy = fit('R3', p, r, labels, scores, ids, 37)
    assert any(np.any(row['bias']) for row in fitted['R7']['trace'][:12])
    assert all(row['bias'] == [0.,0.,0.] for row in fitted['R7']['trace'][:7])
    assert all(max(np.abs(row['bias'])) <= .15 for row in fitted['R7']['trace'])
    assert greedy['weights'] == fitted['R3']['weights']
    # R7 must average transformed probabilities, not average its biases.
    solutions = [dict(weights=[1/6]*6, bias=[-.15,0,.15]), dict(weights=[1,0,0,0,0,0], bias=[.15,0,-.15])]
    mp, mr = predict_mean(solutions, p, r)
    individual = [predict(s,p,r) for s in solutions]
    np.testing.assert_allclose(mp, (individual[0][0]+individual[1][0])/2)
    np.testing.assert_allclose(mr, (individual[0][1]+individual[1][1])/2)
    fake = {l: {m: dict(oof=result, mean_nonzero_models=6) for m in fitted} for l in ('A','B')}
    assert select_method(fake)['selected'] is None
    print('PASS: fixed-label metrics, constants, head/weight equivalence, probabilities, 300 requests, shared initialization, deterministic greedy, seed prediction averaging, fallback.')


if __name__ == '__main__':
    main()
