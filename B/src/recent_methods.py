"""Mask-aware competition adaptations of DPDF-LQ, EBMC, and CMAD concepts.

These use frozen aligned MiniLM features and two prediction heads; they are not
exact reproductions of the authors' end-to-end architectures.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from B.src.fusion import Fusion, MaskedAttentionPool


def _rate(masks):
    return torch.stack([m.float().mean(1) for m in masks], 1)


def _finish(raw):
    return raw[:, :3], 3 * torch.tanh(raw[:, 3])


def _safe_attention(attention, query, key, key_mask):
    safe = key_mask.clone()
    empty = ~safe.any(1)
    if empty.any():
        safe[empty, 0] = True
        key = key.clone()
        key[empty, 0] = 0
    out, _ = attention(query, key, key, key_padding_mask=~safe, need_weights=False)
    return out


class DPDFAligned(nn.Module):
    """DPDF-LQ-style local temporal and dynamic global query paths."""

    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, dim), nn.LayerNorm(dim),
                                  nn.GELU()) for d in (384, 74, 35)])
        self.local_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.GELU(),
            nn.Conv1d(dim, dim, 1)) for _ in range(3)])
        self.local_pool = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.global_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.global_attn = nn.ModuleList([nn.MultiheadAttention(
            dim, 4, dropout=0.1, batch_first=True) for _ in range(3)])
        self.global_gate = nn.Sequential(nn.Linear(dim * 3 + 3, 64),
                                         nn.GELU(), nn.Linear(64, 3))
        self.head = nn.Sequential(nn.Linear(dim * 7 + 3, 128), nn.LayerNorm(128),
                                  nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 64),
                                  nn.GELU(), nn.Linear(64, 4))

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        values = (text.float(), audio.float(), vision.float())
        masks = (text_mask.bool(), audio_mask.bool(), vision_mask.bool())
        projected = [p(x) * m.unsqueeze(-1) for p, x, m in zip(self.proj, values, masks)]
        local, global_features = [], []
        for x, mask, conv, pool, attn in zip(projected, masks, self.local_conv,
                                              self.local_pool, self.global_attn):
            y = (x + conv(x.transpose(1, 2)).transpose(1, 2)) * mask.unsqueeze(-1)
            local.append(pool(y, mask))
            q = self.global_query.expand(x.size(0), -1, -1)
            global_features.append(_safe_attention(attn, q, x, mask).squeeze(1)
                                   * mask.any(1, keepdim=True))
        rates = _rate(masks)
        present = torch.stack([m.any(1) for m in masks], 1)
        gates = torch.softmax(self.global_gate(torch.cat([*global_features, rates], 1))
                              .masked_fill(~present, -1e4), 1) * present.float()
        gates = gates / gates.sum(1, keepdim=True).clamp_min(1e-8)
        fused = sum(gates[:, i:i+1] * global_features[i] for i in range(3))
        raw = self.head(torch.cat([*local, *global_features, fused, rates], 1))
        return (*_finish(raw), gates)


class EBMCAligned(nn.Module):
    """EBMC-style complementary enhancement and reliability-balanced experts."""

    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, dim), nn.LayerNorm(dim),
                                  nn.GELU()) for d in (384, 74, 35)])
        self.pool = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.shared = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.private = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.enhance = nn.ModuleList([nn.Sequential(nn.Linear(dim * 2, dim),
                                     nn.GELU(), nn.Linear(dim, dim)) for _ in range(3)])
        self.experts = nn.ModuleList([nn.Linear(dim, 4) for _ in range(3)])
        self.reliability = nn.Sequential(nn.Linear(dim * 3 + 3, 64),
                                         nn.GELU(), nn.Linear(64, 3))
        self.fusion_head = nn.Sequential(nn.Linear(dim * 4 + 3, 128),
                                         nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25),
                                         nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4))
        self._expert_output = None

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        values = (text.float(), audio.float(), vision.float())
        masks = (text_mask.bool(), audio_mask.bool(), vision_mask.bool())
        pooled = [pool(proj(x) * m.unsqueeze(-1), m) for proj, pool, x, m
                  in zip(self.proj, self.pool, values, masks)]
        present = torch.stack([m.any(1) for m in masks], 1)
        shared = [layer(x) for layer, x in zip(self.shared, pooled)]
        private = [layer(x) for layer, x in zip(self.private, pooled)]
        enhanced = []
        for i in range(3):
            other = sum(shared[j] * present[:, j:j+1] for j in range(3) if j != i)
            count = sum(present[:, j:j+1].float() for j in range(3) if j != i)
            other = other / count.clamp_min(1.0)
            enhanced.append((private[i] + self.enhance[i](torch.cat([shared[i], other], 1)))
                            * present[:, i:i+1])
        rates = _rate(masks)
        gates = torch.softmax(self.reliability(torch.cat([*enhanced, rates], 1))
                              .masked_fill(~present, -1e4), 1) * present.float()
        gates = gates / gates.sum(1, keepdim=True).clamp_min(1e-8)
        fused = sum(gates[:, i:i+1] * enhanced[i] for i in range(3))
        raw = self.fusion_head(torch.cat([*enhanced, fused, rates], 1))
        self._expert_output = [head(x) for head, x in zip(self.experts, enhanced)]
        return (*_finish(raw), gates)

    def auxiliary_loss(self, cls, score, gates):
        terms = []
        for i, raw in enumerate(self._expert_output):
            loss = F.cross_entropy(raw[:, :3], cls, reduction="none")
            loss = loss + 0.8 * F.smooth_l1_loss(3 * torch.tanh(raw[:, 3]), score.float(),
                                                 reduction="none")
            terms.append((loss * gates[:, i].detach()).mean())
        return 0.15 * sum(terms)


class CMADAligned(Fusion):
    """Teacher/student distillation using the existing Fusion backbone."""

    def forward_with_feature(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        inputs = [text.float(), audio.float(), vision.float()]
        pooled = [pool(proj(x), mask) for proj, pool, x, mask in zip(
            (self.tproj, self.aproj, self.vproj), self.pools, inputs, masks)]
        rates = _rate(masks)
        present = torch.stack([mask.any(1) for mask in masks], 1)
        gates = torch.softmax(self.gate(torch.cat([*pooled, rates], 1))
                              .masked_fill(~present, -1e4), 1) * present.float()
        gates = gates / gates.sum(1, keepdim=True).clamp_min(1e-8)
        fused = sum(gates[:, i:i+1] * pooled[i] for i in range(3))
        feature = torch.cat([*pooled, fused, rates], 1)
        return (*_finish(self.head(feature)), gates, feature)

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        logits, score, gates, _ = self.forward_with_feature(
            text, audio, vision, text_mask, audio_mask, vision_mask)
        return logits, score, gates


def cmad_distillation(student, teacher, clean_inputs, masked_inputs):
    """Output, feature, and cross-sample correlation transfer from clean teacher."""
    with torch.no_grad():
        t_logits, t_score, _, t_feature = teacher.forward_with_feature(*clean_inputs)
    s_logits, s_score, _, s_feature = student.forward_with_feature(*masked_inputs)
    temperature = 2.0
    kl = F.kl_div(F.log_softmax(s_logits / temperature, 1),
                  F.softmax(t_logits / temperature, 1), reduction="batchmean")
    feature = F.mse_loss(F.normalize(s_feature, dim=1),
                         F.normalize(t_feature, dim=1))
    s_gram = F.normalize(s_feature, dim=1) @ F.normalize(s_feature, dim=1).T
    t_gram = F.normalize(t_feature, dim=1) @ F.normalize(t_feature, dim=1).T
    correlation = F.mse_loss(s_gram, t_gram)
    return 0.3 * temperature ** 2 * kl + 0.1 * F.smooth_l1_loss(s_score, t_score) + 0.3 * feature + 0.1 * correlation
