"""Deterministic class-balanced batches with per-class coverage and video diversity."""
import numpy as np


class ClassBalancedBatch:
    def __init__(self, labels, sources, seed, batch_size=32):
        self.labels = np.asarray(labels); self.sources = np.asarray(sources)
        self.seed = seed; self.batch_size = batch_size
        self.audit = {}

    def __len__(self): return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pools = [rng.permutation(np.flatnonzero(self.labels == c)).tolist() for c in range(3)]
        assert all(len(p) >= self.batch_size for p in pools)
        draws = []
        for b in range(len(self)):
            size = min(self.batch_size, len(self.labels) - b * self.batch_size)
            counts = [size // 3] * 3
            for j in range(size % 3): counts[(b+j) % 3] += 1
            batch = []; videos = set()
            for c, count in enumerate(counts):
                for _ in range(count):
                    if not pools[c]: pools[c] = rng.permutation(np.flatnonzero(self.labels == c)).tolist()
                    candidates = [k for k, idx in enumerate(pools[c]) if idx not in batch]
                    diverse = [k for k in candidates if self.sources[pools[c][k]] not in videos]
                    k = (diverse or candidates)[0]
                    idx = pools[c].pop(k); batch.append(idx); videos.add(self.sources[idx])
            rng.shuffle(batch); draws.extend(batch)
            yield batch
        self.audit = dict(draws=len(draws), unique_samples=len(set(draws)),
                          unique_videos=len(set(self.sources[draws].tolist())),
                          repeat_fraction=1-len(set(draws))/len(draws),
                          class_counts=np.bincount(self.labels[draws], minlength=3).tolist())
