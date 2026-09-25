"""Prediction-only, fixed-budget ensemble comparisons for C round two.

Inputs: probabilities [M,S,N,3], regression [M,S,N], sample IDs [N].
Class order is negative/neutral/positive. predict never accepts labels.
"""
from __future__ import annotations

import time
import numpy as np


def metrics(prob, reg, labels, scores):
    """Fixed three-class metrics over each scenario; constant Pearson is zero."""
    prob, reg = np.asarray(prob), np.asarray(reg)
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if prob.shape != (*reg.shape, 3) or reg.shape[-1] != len(labels):
        raise ValueError('prediction/label shape mismatch')
    if not np.isfinite(prob).all() or not np.isfinite(reg).all():
        raise ValueError('nonfinite predictions')
    predicted = prob.argmax(-1)
    rows = []
    for p, r in zip(predicted, reg):
        confusion = np.bincount(labels * 3 + p, minlength=9).reshape(3, 3)
        denominator = confusion.sum(0) + confusion.sum(1)
        f1 = np.divide(2 * confusion.diagonal(), denominator,
                       out=np.zeros(3), where=denominator > 0)
        x, y = r - r.mean(), scores - scores.mean()
        constant = bool(np.std(r) <= 1e-8 or np.std(scores) <= 1e-8)
        corr = 0. if constant else float(x @ y / np.sqrt((x @ x) * (y @ y)))
        rows.append(dict(accuracy=float(np.mean(p == labels)), macro_f1=float(f1.mean()),
                         mae=float(np.abs(r - scores).mean()), pearson=corr,
                         constant_pearson=constant, per_class_f1=f1.tolist(), n=len(labels)))
    return dict(scenarios=rows, selection_score=float(np.mean([
        r['macro_f1'] - .15 * r['mae'] + .05 * r['pearson'] for r in rows])))


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / x.sum() if x.sum() > 0 else np.full(len(x), 1 / len(x))


def predict(solution, probs, reg):
    w = np.asarray(solution['weights'])
    p = np.einsum('m,msnc->snc', w, probs, optimize=False)
    r = np.einsum('m,msn->sn', w, reg, optimize=False)
    bias = np.asarray(solution.get('bias', [0., 0., 0.]))
    if np.any(bias):
        logits = np.log(np.maximum(p, 1e-8)) + bias
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
    return p, r


def predict_mean(solutions, probs, reg):
    values = [predict(s, probs, reg) for s in solutions]
    return np.mean([v[0] for v in values], axis=0), np.mean([v[1] for v in values], axis=0)


def fit(method, probs, reg, labels, scores, ids, seed=11):
    """R0..R7; fixed six-model protocol (D/first seed is model index 2).

    Every requested candidate is logged, including duplicates; exact duplicate
    predictions/objectives are cached within a run. DE uses generation snapshots.
    """
    started = time.perf_counter()
    m = probs.shape[0]
    if m != 6 or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError('six aligned models and unique sample IDs required')
    rng = np.random.default_rng(seed)
    cache, records = {}, []

    def evaluate(w, bias=None):
        w = normalize(w)
        bias = np.zeros(3) if bias is None else np.asarray(bias)
        solution = dict(weights=w.tolist(), bias=bias.tolist())
        key = (w.tobytes(), bias.astype(np.float64).tobytes())
        hit = key in cache
        if not hit:
            p, r = predict(solution, probs, reg)
            score = metrics(p, r, labels, scores)['selection_score']
            penalty = float((m * (w @ w) - 1) / (m - 1))
            objective = score - (.01 * penalty if method in ('R6', 'R7') else 0)
            cache[key] = (score, penalty, objective)
        score, penalty, objective = cache[key]
        records.append(dict(request=len(records) + 1, **solution, score=score,
                            penalty=penalty, objective=objective, cache_hit=hit))
        return objective

    eye = np.eye(m)
    if method == 'R0':
        evaluate(eye[2])
    elif method == 'R1':
        for w in eye:
            evaluate(w)
    elif method == 'R2':
        evaluate(np.ones(m))
    elif method == 'R3':
        counts = np.zeros(m)
        for _ in range(50):
            values = [evaluate(counts + w) for w in eye]
            counts[np.argmax(values)] += 1
    elif method in ('R4', 'R5', 'R6', 'R7'):
        initial = np.vstack([np.full(m, 1 / m), eye,
                             np.array([normalize(rng.random(m)) for _ in range(5)])])
        if method == 'R7':
            # A constant bias population cannot move under differential mutation.
            # Keep seven anchors unbiased; seed the five random points separately
            # so all methods retain exactly the same initial weight coordinates.
            bias_coordinates = np.full((12, 2), .5)
            bias_coordinates[7:] = np.random.default_rng(seed + 100000).random((5, 2))
            population = np.column_stack([initial, bias_coordinates])
        else:
            population = initial

        def evaluate_vector(x):
            bias = np.array([.3 * x[6] - .15, 0, .3 * x[7] - .15]) if method == 'R7' else None
            return evaluate(x[:m], bias)

        fitness = np.array([evaluate_vector(x) for x in population])
        for _ in range(24):
            old = population.copy()
            for i in range(12):
                if method == 'R4':
                    child = rng.random(m)
                else:
                    a, b, c = rng.choice([j for j in range(12) if j != i], 3, replace=False)
                    mutant = np.clip(old[a] + .5 * (old[b] - old[c]), 0, 1)
                    cross = rng.random(len(old[i])) < .5
                    cross[rng.integers(len(cross))] = True
                    child = np.where(cross, mutant, old[i])
                value = evaluate_vector(child)
                if value > fitness[i]:
                    population[i], fitness[i] = child, value
    else:
        raise ValueError(method)
    if method in ('R3', 'R4', 'R5', 'R6', 'R7') and len(records) != 300:
        raise AssertionError('budget mismatch')
    best = max(records, key=lambda r: r['objective'])
    return dict(method=method, seed=seed, weights=best['weights'], bias=best['bias'],
                fit_score=best['score'], objective=best['objective'],
                requests=len(records), unique_candidates=len(cache),
                cache_hits=len(records)-len(cache), actual_objective_calls=len(cache),
                seconds=time.perf_counter()-started, trace=records)


def select_method(results):
    """Apply the preregistered two-library guardrails without test access."""
    candidates, checks = [], {}
    for method in [f'R{i}' for i in range(1, 8)]:
        deltas, clean_ok, nonzero = [], [], []
        for library in ('A', 'B'):
            ref, row = results[library]['R0'], results[library][method]
            deltas.append(row['oof']['selection_score'] - ref['oof']['selection_score'])
            a, b = row['oof']['scenarios'][0], ref['oof']['scenarios'][0]
            clean_ok.append(a['macro_f1'] >= b['macro_f1'] - .01 and a['mae'] <= b['mae'] + .02)
            nonzero.append(row['mean_nonzero_models'])
        eligible = all(clean_ok) and min(deltas) >= 0 and np.mean(deltas) >= .005
        mean_score = float(np.mean([results[l][method]['oof']['selection_score'] for l in ('A', 'B')]))
        checks[method] = dict(deltas=deltas, mean_delta=float(np.mean(deltas)),
                              clean_ok=clean_ok, eligible=bool(eligible), mean_score=mean_score,
                              mean_nonzero_models=float(np.mean(nonzero)))
        if eligible:
            candidates.append(method)
    if not candidates:
        return dict(selected=None, decision='retain_round1_locked_submission', checks=checks)
    best = max(checks[m]['mean_score'] for m in candidates)
    tied = [m for m in candidates if best - checks[m]['mean_score'] <= .002]
    selected = min(tied, key=lambda m: (checks[m]['mean_nonzero_models'], int(m[1:])))
    return dict(selected=selected, decision='round2_candidate', checks=checks)
