"""Mask-conditioned local reconstruction, cross-modal imputation, and reliability fusion."""
from __future__ import annotations

import torch
from torch import nn

from B.src.fusion import MaskedAttentionPool


class ReliabilityImputationFusion(nn.Module):
    """Reconstruct hidden aligned positions and down-weight uncertain imputations."""

    widths = (384, 74, 35)

    def __init__(self, dim: int = 64, max_positions: int = 50):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(width, dim), nn.LayerNorm(dim), nn.GELU())
            for width in self.widths
        ])
        self.positions = nn.Embedding(max_positions, dim)
        self.mask_tokens = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.local_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0.1,
                                           batch_first=True, norm_first=True),
                num_layers=1,
            ) for _ in self.widths
        ])
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
            for _ in self.widths
        ])
        self.impute_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())
            for _ in self.widths
        ])
        self.reconstruction = nn.ModuleList([
            nn.Linear(dim, width) for width in self.widths
        ])
        self.log_variance = nn.ModuleList([nn.Linear(dim, 1) for _ in self.widths])
        self.modality_pool = nn.ModuleList([MaskedAttentionPool(dim) for _ in self.widths])
        # Rates, longest gap ratios, and mean predicted confidence for each modality.
        self.reliability_gate = nn.Sequential(
            nn.Linear(dim * 3 + 9, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3)
        )
        self.head = nn.Sequential(
            nn.Linear(dim * 4 + 9, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4),
        )

    @staticmethod
    def _longest_gaps(masks: list[torch.Tensor], support: torch.Tensor) -> torch.Tensor:
        features = []
        for mask in masks:
            missing = support & ~mask
            run = torch.zeros(mask.shape[0], device=mask.device)
            longest = torch.zeros_like(run)
            for step in range(mask.shape[1]):
                run = torch.where(missing[:, step], run + 1, 0)
                longest = torch.maximum(longest, run)
            features.append(longest / support.sum(dim=1).clamp_min(1))
        return torch.stack(features, dim=1)

    @staticmethod
    def _safe_keys(keys: torch.Tensor, key_mask: torch.Tensor):
        safe_mask = key_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            keys = keys.clone()
            keys[empty, 0] = 0
        return keys, safe_mask

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        values = [text.float(), audio.float(), vision.float()]
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        batch, steps = masks[0].shape
        if steps > self.positions.num_embeddings:
            raise ValueError(f"sequence length {steps} exceeds configured positions")
        support = masks[0] | masks[1] | masks[2]
        position = self.positions(torch.arange(steps, device=text.device))[None]

        local = []
        for index, (value, mask, project, encoder) in enumerate(zip(
                values, masks, self.projections, self.local_encoders)):
            state = project(value) + position
            mask_token = self.mask_tokens[index].view(1, 1, -1)
            state = torch.where(mask.unsqueeze(-1), state, mask_token + position)
            safe_mask = mask.clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            local.append(encoder(state, src_key_padding_mask=~safe_mask))

        reconstructed, logvars, confidences, summaries = [], [], [], []
        for index in range(3):
            other_indices = [j for j in range(3) if j != index]
            keys = torch.cat([local[j] for j in other_indices], dim=1)
            key_mask = torch.cat([masks[j] for j in other_indices], dim=1)
            keys, safe_keys = self._safe_keys(keys, key_mask)
            cross, _ = self.cross_attention[index](
                local[index], keys, keys, key_padding_mask=~safe_keys, need_weights=False
            )
            candidate = self.impute_layers[index](torch.cat([local[index], cross], dim=-1))
            filled = torch.where(masks[index].unsqueeze(-1), local[index], candidate)
            logvar = self.log_variance[index](candidate).squeeze(-1).clamp(-4.0, 4.0)
            confidence = torch.exp(-torch.nn.functional.softplus(logvar))
            weights = torch.where(masks[index], torch.ones_like(confidence), confidence)
            weights = weights * support.float()
            pool_logits = self.modality_pool[index].score(filled).squeeze(-1)
            pool_logits = pool_logits + torch.log(weights.clamp_min(1e-8))
            pool_logits = pool_logits.masked_fill(~support, -1e4)
            pool_weights = torch.softmax(pool_logits, dim=1) * support.float()
            pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pooled = (filled * pool_weights.unsqueeze(-1)).sum(dim=1)
            confidence_mean = (confidence * support.float()).sum(dim=1) / support.sum(
                dim=1).clamp_min(1)
            summaries.append(pooled)
            reconstructed.append(self.reconstruction[index](candidate))
            logvars.append(logvar)
            confidences.append(confidence_mean)

        rates = torch.stack([
            (mask & support).float().sum(dim=1) / support.sum(dim=1).clamp_min(1)
            for mask in masks
        ], dim=1)
        gaps = self._longest_gaps(masks, support)
        confidence_features = torch.stack(confidences, dim=1)
        reliability = torch.cat([rates, gaps, confidence_features], dim=1)
        modal_summaries = summaries
        # An explicit observed-rate prior prevents a fully imputed stream dominating by default.
        gate_logits = self.reliability_gate(
            torch.cat([*modal_summaries, reliability], dim=1)
        ) + torch.log((0.05 + 0.475 * rates + 0.475 * confidence_features).clamp_min(0.02))
        gates = torch.softmax(gate_logits, dim=1)
        fused = sum(gates[:, i:i + 1] * modal_summaries[i] for i in range(3))
        raw = self.head(torch.cat([*modal_summaries, fused, reliability], dim=1))
        aux = {
            "reconstruction": reconstructed,
            "log_variance": logvars,
            "fusion_representation": torch.cat([*modal_summaries, fused, reliability], dim=1),
        }
        return raw[:, :3], 3 * torch.tanh(raw[:, 3]), gates, aux
