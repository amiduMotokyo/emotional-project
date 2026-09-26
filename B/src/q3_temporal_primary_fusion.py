"""Masked primary-modality sequence fusion; shared branches, original positions."""
import torch
from torch import nn


class TemporalPrimaryFusion(nn.Module):
    def __init__(self, mode='dynamic', dim=64):
        super().__init__()
        if mode not in ('text', 'uniform', 'dynamic'): raise ValueError(mode)
        self.mode = mode
        self.projections = nn.ModuleList([nn.Linear(d, dim) for d in (384, 74, 35)])
        self.position = nn.Parameter(torch.zeros(1, 50, dim))
        self.modality = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.position, std=.02); nn.init.normal_(self.modality, std=.02)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=.25, batch_first=True)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(.25), nn.Linear(128, dim))
        self.dropout = nn.Dropout(.25)
        self.selector = nn.Sequential(nn.Linear(dim * 3 + 3, 32), nn.GELU(), nn.Linear(32, 3)) if mode == 'dynamic' else None
        self.head = nn.Linear(dim, 4)
        self.last_features = None

    @staticmethod
    def pool(x, mask):
        return (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def forward(self, text, audio, vision, tm, am, vm):
        masks = [m.bool() for m in (tm, am, vm)]
        xs = [p(x.float()) + self.position[:, :x.shape[1]] + self.modality[i]
              for i, (p, x) in enumerate(zip(self.projections, (text, audio, vision)))]
        present = torch.stack([m.any(1) for m in masks], 1)
        hs = []
        for i in range(3):
            kv = torch.cat([xs[j] for j in range(3) if j != i], 1)
            km = torch.cat([masks[j] for j in range(3) if j != i], 1)
            absent = ~km.any(1)
            safe = km.clone(); safe[absent, 0] = True
            kv = kv * km[..., None]
            delta, _ = self.attention(xs[i], kv, kv, key_padding_mask=~safe, need_weights=False)
            delta = delta.masked_fill(absent[:, None, None], 0.)
            h = self.norm1(xs[i] + self.dropout(delta))
            h = self.norm2(h + self.dropout(self.ff(h)))
            hs.append(self.pool(h, masks[i]))
        if self.mode == 'dynamic':
            features = torch.cat([*(self.pool(x, m) for x, m in zip(xs, masks)), present.float()], 1)
            weights = self.selector(features).masked_fill(~present, -1e4).softmax(1) * present
        else:
            weights = present.float()
            if self.mode == 'text':
                weights = torch.where(present[:, :1], torch.tensor([1., 0., 0.], device=text.device), weights)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        self.last_features = sum(weights[:, i:i+1] * hs[i] for i in range(3))
        out = self.head(self.last_features)
        return out[:, :3], 3 * out[:, 3].tanh(), weights
