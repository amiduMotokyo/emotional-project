"""Small, bounded DE and filter-BPSO searches; no model or test-set access."""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import mutual_info_score


BOUNDS = np.array([[-4, -2.5], [-6, -2], [.1, .5],
                   [math.log10(.25), math.log10(2)]])


def decode(vector):
    x = BOUNDS[:, 0] + np.asarray(vector) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    return dict(learning_rate=float(10 ** x[0]), weight_decay=float(10 ** x[1]),
                dropout=float(x[2]), lambda_reg=float(10 ** x[3]))


def parameter_search(evaluate, method, population=6, generations=3, seed=11):
    """Maximize endpoint validation score. Random and DE share initialization."""
    if method not in ('de', 'random') or population < 4 or generations < 0:
        raise ValueError('invalid search settings')
    rng = np.random.default_rng(seed)
    particles = rng.random((population, 4))
    records = []

    def score(x, generation, target, parents=None):
        parameters = decode(x)
        value = float(evaluate(parameters))
        if not np.isfinite(value):
            value = -float('inf')
        records.append(dict(parameters=parameters, score=value,
                            generation=generation, target=target, parents=parents))
        return value

    values = np.array([score(x, 0, i) for i, x in enumerate(particles)])
    for generation in range(1, generations + 1):
        old = particles.copy()
        for i in range(population):
            parents = None
            if method == 'random':
                child = rng.random(4)
            else:
                a, b, c = rng.choice([j for j in range(population) if j != i], 3, replace=False)
                parents = [int(a), int(b), int(c)]
                mutant = np.clip(old[a] + .5 * (old[b] - old[c]), 0, 1)
                cross = rng.random(4) < .5
                cross[rng.integers(4)] = True
                child = np.where(cross, mutant, old[i])
            result = score(child, generation, i, parents)
            if result > values[i]:
                particles[i], values[i] = child, result
    return records


def discretize(values, valid):
    output = np.zeros(len(values), dtype=np.int64)
    if valid.any():
        edges = np.unique(np.quantile(values[valid], [.2, .4, .6, .8]))
        output[valid] = np.searchsorted(edges, values[valid], side='right')
    return output


def su(x, y):
    hx, hy = mutual_info_score(x, x), mutual_info_score(y, y)
    return float(2 * mutual_info_score(x, y) / (hx + hy)) if hx + hy > 0 else 0.


def filter_statistics(train):
    """Fit discretization/relevance/redundancy using training observations only."""
    labels = train['cls']
    target = discretize(train['score'], np.ones(len(labels), bool))
    summaries, validities = [], []
    for key, mask_key in (('audio', 'amask'), ('vision', 'vmask')):
        values, mask = train[key], train[mask_key].astype(bool)
        count = mask.sum(axis=1)
        mean = (values * mask[..., None]).sum(axis=1) / np.maximum(count[:, None], 1)
        variance = (((values - mean[:, None]) ** 2) * mask[..., None]).sum(axis=1)
        std = np.sqrt(variance / np.maximum(count[:, None], 1))
        valid = count > 0
        for j in range(values.shape[-1]):
            summaries.append((discretize(mean[:, j], valid), discretize(std[:, j], valid)))
            validities.append(valid)
    n = len(summaries)
    relevance = np.zeros(n)
    redundancy = np.zeros((n, n))
    for j, (mean, std) in enumerate(summaries):
        ok = validities[j]
        if ok.sum() >= 20:
            relevance[j] = np.mean([su(x[ok], y[ok]) for x in (mean, std)
                                   for y in (labels, target)])
        for k in range(j):
            # Only within-modality redundancy contributes to this experiment.
            if (j < 74) != (k < 74):
                continue
            both = ok & validities[k]
            value = (np.mean([su(x[both], y[both]) for x in summaries[j]
                              for y in summaries[k]]) if both.sum() >= 20 else 1.)
            redundancy[j, k] = redundancy[k, j] = value
    return relevance, redundancy


def subset_score(mask, relevance, redundancy):
    scores = []
    for start, stop in ((0, 74), (74, 109)):
        indices = np.flatnonzero(mask[start:stop]) + start
        if not len(indices):
            return -float('inf')
        pairs = redundancy[np.ix_(indices, indices)][np.triu_indices(len(indices), 1)]
        scores.append(float(relevance[indices].mean() - .2 * (pairs.mean() if len(pairs) else 0)))
    return float(np.mean(scores))


def repair(mask, probabilities, quotas, rng):
    mask = mask.copy()
    for start, stop, quota in ((0, 74, quotas[0]), (74, 109, quotas[1])):
        on = np.flatnonzero(mask[start:stop]) + start
        off = np.flatnonzero(~mask[start:stop]) + start
        if len(on) > quota:
            order = np.lexsort((rng.random(len(on)), probabilities[on]))
            mask[on[order[:len(on) - quota]]] = False
        elif len(on) < quota:
            order = np.lexsort((rng.random(len(off)), -probabilities[off]))
            mask[off[order[:quota - len(on)]]] = True
    return mask


def feature_search(relevance, redundancy, method, ratio, seed=11, population=20, updates=30):
    if method not in ('bpso', 'random', 'topk') or not 0 < ratio <= 1:
        raise ValueError('invalid feature search')
    rng = np.random.default_rng(seed)
    quotas = (math.ceil(74 * ratio), math.ceil(35 * ratio))

    def random_mask():
        mask = np.zeros(109, dtype=bool)
        mask[rng.choice(74, quotas[0], replace=False)] = True
        mask[74 + rng.choice(35, quotas[1], replace=False)] = True
        return mask

    if method == 'topk':
        mask = repair(np.zeros(109, bool), relevance, quotas, rng)
        return {'audio_indices': np.flatnonzero(mask[:74]).tolist(),
                'vision_indices': np.flatnonzero(mask[74:]).tolist(),
                'filter_score': subset_score(mask, relevance, redundancy), 'evaluations': 1}
    particles = np.array([random_mask() for _ in range(population)])
    velocity = np.zeros_like(particles, dtype=float)
    personal = particles.copy()
    fitness = np.array([subset_score(p, relevance, redundancy) for p in personal])
    best = personal[fitness.argmax()].copy()
    trace = [float(fitness.max())]
    for _ in range(updates):
        leader = best.copy()
        for i in range(population):
            if method == 'random':
                candidate = random_mask()
            else:
                velocity[i] = np.clip(.7298 * velocity[i] +
                    1.49618 * rng.random(109) * (personal[i].astype(float) - particles[i]) +
                    1.49618 * rng.random(109) * (leader.astype(float) - particles[i]), -6, 6)
                probability = 1 / (1 + np.exp(-velocity[i]))
                candidate = repair(rng.random(109) < probability, probability, quotas, rng)
            particles[i] = candidate
            value = subset_score(candidate, relevance, redundancy)
            if value > fitness[i]:
                personal[i], fitness[i] = candidate, value
        best = personal[fitness.argmax()].copy()
        trace.append(float(fitness.max()))
    return dict(audio_indices=np.flatnonzero(best[:74]).tolist(),
                vision_indices=np.flatnonzero(best[74:]).tolist(),
                filter_score=float(fitness.max()), evaluations=population * (updates + 1),
                best_trace=trace)
