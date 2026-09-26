"""Sample-level supervised contrast with different-source-video pairs only."""
import torch
from torch.nn import functional as F


def supervised_contrastive(z, labels, source, temperature=.1):
    z = F.normalize(z, dim=-1)
    eligible = source[:, None] != source[None, :]
    positive = eligible & (labels[:, None] == labels[None, :])
    valid = positive.any(1)
    if not valid.any(): return z.sum() * 0., 0, len(z)
    logits = (z @ z.T)[valid] / temperature
    allowed = eligible[valid]; pos = positive[valid]
    logp = logits - logits.masked_fill(~allowed, -torch.inf).logsumexp(1, keepdim=True)
    loss = -(logp.masked_fill(~pos, 0.).sum(1) / pos.sum(1)).mean()
    return loss, int(valid.sum()), len(z)
