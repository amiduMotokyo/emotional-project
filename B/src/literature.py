"""Literature-inspired multimodal fusion baselines for aligned MOSEI features.

LMF, MISA, MulT, and MAG are adapted to the existing cached token-level
MiniLM, audio, and visual features. The MAG variant reuses frozen MiniLM token
states; it is not an end-to-end MAG-BERT fine-tuning run.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from B.src.fusion import MaskedAttentionPool


def _rates(masks):
    return torch.stack([mask.float().mean(dim=1) for mask in masks], dim=1)


def _present(masks):
    return torch.stack([mask.bool().any(dim=1) for mask in masks], dim=1)


def _masked_pool(projection, pool, values, mask):
    result = pool(projection(values.float()), mask.bool())
    return result * mask.bool().any(dim=1, keepdim=True).float()


def _head(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Dropout(0.25), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4),
    )


def _split_output(output):
    return output[:, :3], 3.0 * torch.tanh(output[:, 3])


class LowRankMultimodalFusion(nn.Module):
    """LMF-style factorized tensor fusion after masked modality pooling."""

    def __init__(self, dim: int = 64, rank: int = 8):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(384, dim), nn.LayerNorm(dim), nn.GELU()),
            nn.Sequential(nn.Linear(74, dim), nn.LayerNorm(dim), nn.GELU()),
            nn.Sequential(nn.Linear(35, dim), nn.LayerNorm(dim), nn.GELU()),
        ])
        self.pools = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(rank, dim + 1, dim)) for _ in range(3)
        ])
        for factor in self.factors:
            nn.init.xavier_uniform_(factor)
        self.fusion_bias = nn.Parameter(torch.zeros(dim))
        self.head = _head(dim + 3)

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        values = [text, audio, vision]
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        pooled = [
            _masked_pool(proj, pool, value, mask)
            for proj, pool, value, mask in zip(self.projections, self.pools, values, masks)
        ]
        factors = [
            torch.einsum(
                "bi,rid->brd",
                torch.cat([x, torch.ones_like(x[:, :1])], dim=1),
                weight,
            )
            for x, weight in zip(pooled, self.factors)
        ]
        fused = factors[0] * factors[1] * factors[2]
        fused = fused.sum(dim=1) / self.factors[0].shape[0] + self.fusion_bias
        rates = _rates(masks)
        output = self.head(torch.cat([fused, rates], dim=1))
        presence = _present(masks)
        gates = rates.masked_fill(~presence, 0.0)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (*_split_output(output), gates)


class MISAStyleFusion(nn.Module):
    """Shared/private modality subspaces with CMD, difference, and reconstruction losses."""

    def __init__(self, dim: int = 64, sim_weight: float = 0.3,
                 diff_weight: float = 0.1, recon_weight: float = 1.0):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(384, dim), nn.LayerNorm(dim), nn.GELU()),
            nn.Sequential(nn.Linear(74, dim), nn.LayerNorm(dim), nn.GELU()),
            nn.Sequential(nn.Linear(35, dim), nn.LayerNorm(dim), nn.GELU()),
        ])
        self.pools = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.shared = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
                                     for _ in range(3)])
        self.private = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
                                      for _ in range(3)])
        self.decoders = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(3)])
        self.head = _head(dim * 6 + 3)
        self.sim_weight = sim_weight
        self.diff_weight = diff_weight
        self.recon_weight = recon_weight
        self._auxiliary = None

    @staticmethod
    def _diff_loss(left, right):
        left = left - left.mean(dim=0, keepdim=True)
        right = right - right.mean(dim=0, keepdim=True)
        left = left / (torch.linalg.vector_norm(left, dim=1, keepdim=True).detach() + 1e-6)
        right = right / (torch.linalg.vector_norm(right, dim=1, keepdim=True).detach() + 1e-6)
        return (left.T @ right).square().mean()

    @staticmethod
    def _cmd(left, right, moments: int = 5):
        left_mean = left.mean(dim=0)
        right_mean = right.mean(dim=0)
        loss = torch.linalg.vector_norm(left_mean - right_mean)
        left_centered = left - left_mean
        right_centered = right - right_mean
        for order in range(2, moments + 1):
            lm = left_centered.pow(order).mean(dim=0)
            rm = right_centered.pow(order).mean(dim=0)
            loss = loss + torch.linalg.vector_norm(lm - rm) / (2 ** order)
        return loss

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        values = [text, audio, vision]
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        pooled = [
            _masked_pool(proj, pool, value, mask)
            for proj, pool, value, mask in zip(self.projections, self.pools, values, masks)
        ]
        shared = [encoder(x) for encoder, x in zip(self.shared, pooled)]
        private = [encoder(x) for encoder, x in zip(self.private, pooled)]
        reconstructed = [
            decoder(torch.cat([s, p], dim=1))
            for decoder, s, p in zip(self.decoders, shared, private)
        ]
        sim = sum(self._cmd(shared[i], shared[j]) for i, j in ((0, 1), (0, 2), (1, 2))) / 3.0
        diff_terms = [self._diff_loss(shared[i], private[i]) for i in range(3)]
        diff_terms.extend(self._diff_loss(private[i], private[j])
                          for i, j in ((0, 1), (0, 2), (1, 2)))
        diff = torch.stack(diff_terms).mean()
        recon = torch.stack([
            F.mse_loss(pred, target) for pred, target in zip(reconstructed, pooled)
        ]).mean()
        self._auxiliary = (
            self.sim_weight * sim + self.diff_weight * diff + self.recon_weight * recon
        )
        rates = _rates(masks)
        output = self.head(torch.cat([*shared, *private, rates], dim=1))
        presence = _present(masks)
        gates = rates.masked_fill(~presence, 0.0)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (*_split_output(output), gates)

    def auxiliary_loss(self):
        if self._auxiliary is None:
            raise RuntimeError("forward must be called before auxiliary_loss")
        return self._auxiliary


class _CrossModalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, query_mask, key_mask):
        key_value = key_value * key_mask.unsqueeze(-1).float()
        safe_key_mask = key_mask.clone()
        empty = ~safe_key_mask.any(dim=1)
        if empty.any():
            safe_key_mask[empty, 0] = True
            key_value = key_value.clone()
            key_value[empty, 0] = 0.0
        attended, _ = self.attention(
            query, key_value, key_value,
            key_padding_mask=~safe_key_mask, need_weights=False,
        )
        attended = attended * query_mask.unsqueeze(-1).float()
        return self.norm(query + self.dropout(attended)) * query_mask.unsqueeze(-1).float()


class MulTAligned(nn.Module):
    """MulT-style directional cross-modal attention on the aligned 50-step inputs."""

    def __init__(self, dim: int = 64, heads: int = 4):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(384, dim), nn.LayerNorm(dim), nn.GELU()),
            nn.Sequential(nn.Linear(74, dim), nn.LayerNorm(dim), nn.GELU()),
            nn.Sequential(nn.Linear(35, dim), nn.LayerNorm(dim), nn.GELU()),
        ])
        self.cross = nn.ModuleDict({
            "ta": _CrossModalAttention(dim, heads),
            "tv": _CrossModalAttention(dim, heads),
            "at": _CrossModalAttention(dim, heads),
            "av": _CrossModalAttention(dim, heads),
            "vt": _CrossModalAttention(dim, heads),
            "va": _CrossModalAttention(dim, heads),
        })
        self.self_attention = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=0.15, batch_first=True)
            for _ in range(3)
        ])
        self.self_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.dropout = nn.Dropout(0.15)
        self.pools = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.gate = nn.Linear(dim * 3 + 3, 3)
        self.head = _head(dim * 4 + 3)

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        raw = [text, audio, vision]
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        values = [
            projection(value.float()) * mask.unsqueeze(-1).float()
            for projection, value, mask in zip(self.projections, raw, masks)
        ]
        t, a, v = values
        tm, am, vm = masks
        t_new = self.cross["ta"](t, a, tm, am) + self.cross["tv"](t, v, tm, vm)
        a_new = self.cross["at"](a, t, am, tm) + self.cross["av"](a, v, am, vm)
        v_new = self.cross["vt"](v, t, vm, tm) + self.cross["va"](v, a, vm, am)
        values = [
            t_new * tm.unsqueeze(-1).float(),
            a_new * am.unsqueeze(-1).float(),
            v_new * vm.unsqueeze(-1).float(),
        ]
        contextual = []
        for x, mask, attention, norm in zip(values, masks, self.self_attention, self.self_norms):
            safe_mask = mask.clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
                x = x.clone()
                x[empty, 0] = 0.0
            attended, _ = attention(
                x, x, x, key_padding_mask=~safe_mask, need_weights=False,
            )
            x = norm(x + self.dropout(attended)) * mask.unsqueeze(-1).float()
            contextual.append(x)
        pooled = [
            pool(sequence, mask)
            for pool, sequence, mask in zip(self.pools, contextual, masks)
        ]
        rates = _rates(masks)
        joined = torch.cat([*pooled, rates], dim=1)
        present = _present(masks)
        gates = torch.softmax(self.gate(joined).masked_fill(~present, -1e4), dim=1) * present.float()
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = sum(gates[:, i:i + 1] * pooled[i] for i in range(3))
        output = self.head(torch.cat([*pooled, fused, rates], dim=1))
        return (*_split_output(output), gates)


class MAGMiniLM(nn.Module):
    """MAG-style nonverbal shift applied to frozen MiniLM token-level states."""

    def __init__(self, hidden: int = 384, dim: int = 64):
        super().__init__()
        self.text_norm = nn.LayerNorm(hidden)
        self.audio_shift = nn.Linear(74, hidden)
        self.vision_shift = nn.Linear(35, hidden)
        self.audio_present_projection = nn.Sequential(
            nn.Linear(74, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.vision_present_projection = nn.Sequential(
            nn.Linear(35, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.text_pool = MaskedAttentionPool(hidden)
        self.audio_pool = MaskedAttentionPool(dim)
        self.vision_pool = MaskedAttentionPool(dim)
        self.head = _head(hidden + dim * 2 + 3)

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        tm, am, vm = text_mask.bool(), audio_mask.bool(), vision_mask.bool()
        z = self.text_norm(text.float())
        audio_value = audio.float() * am.unsqueeze(-1).float()
        vision_value = vision.float() * vm.unsqueeze(-1).float()
        h = torch.tanh(self.audio_shift(audio_value) + self.vision_shift(vision_value))
        av_present = (am | vm)
        h = h * av_present.unsqueeze(-1).float()
        h_norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
        z_norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        magnitude = torch.minimum(z_norm, h_norm)
        shifted = z + magnitude * h / h_norm
        shifted = shifted * tm.unsqueeze(-1).float()
        t_pooled = self.text_pool(shifted, tm)
        a_seq = self.audio_present_projection(audio.float()) * am.unsqueeze(-1).float()
        v_seq = self.vision_present_projection(vision.float()) * vm.unsqueeze(-1).float()
        a_pooled = self.audio_pool(a_seq, am)
        v_pooled = self.vision_pool(v_seq, vm)
        rates = _rates([tm, am, vm])
        output = self.head(torch.cat([t_pooled, a_pooled, v_pooled, rates], dim=1))
        presence = _present([tm, am, vm])
        gates = rates.masked_fill(~presence, 0.0)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (*_split_output(output), gates)
