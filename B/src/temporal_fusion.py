"""Position-aware, mask-conditioned fusion for aligned local missingness."""
from __future__ import annotations

import torch
from torch import nn

from .fusion import MaskedAttentionPool


class TemporalFusion(nn.Module):
    """One model for every observed modality combination; no imputed input is trusted."""

    uses_support = True

    def __init__(self, dim: int = 64, max_positions: int = 50):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(width, dim), nn.LayerNorm(dim), nn.GELU())
            for width in (384, 74, 35)
        ])
        self.local = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
            for _ in range(3)
        ])
        self.local_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.modality_pools = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.positions = nn.Embedding(max_positions, dim)
        self.gate = nn.Sequential(nn.Linear(3 * dim + 6, 64), nn.GELU(),
                                  nn.Linear(64, 3))
        self.sequence = nn.GRU(dim + 6, dim, batch_first=True, bidirectional=True)
        self.sequence_pool = MaskedAttentionPool(2 * dim)
        self.head = nn.Sequential(
            nn.Linear(5 * dim + 6, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4),
        )

    @staticmethod
    def _longest_gaps(masks: list[torch.Tensor], support: torch.Tensor) -> torch.Tensor:
        """Unavailable spans, including both ends, divided by original support length."""
        result = []
        for mask in masks:
            missing = support & ~mask
            run = torch.zeros(mask.shape[0], device=mask.device)
            longest = torch.zeros_like(run)
            for time in range(mask.shape[1]):
                run = torch.where(missing[:, time], run + 1, 0)
                longest = torch.maximum(longest, run)
            result.append(longest / support.sum(dim=1).clamp_min(1))
        return torch.stack(result, dim=1)

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask,
                support: torch.Tensor | None = None):
        values = [text.float(), audio.float(), vision.float()]
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        if support is None:
            support = masks[0] | masks[1] | masks[2]
        support = support.bool()
        batch, steps = masks[0].shape
        positions = self.positions(torch.arange(steps, device=text.device))[None]
        hidden = []
        for value, mask, project, conv, norm in zip(
                values, masks, self.projections, self.local, self.local_norm):
            state = (project(value) + positions) * mask.unsqueeze(-1)
            state = norm(state + conv(state.transpose(1, 2)).transpose(1, 2))
            hidden.append(state * mask.unsqueeze(-1))
        available = torch.stack(masks, dim=-1)
        rates = available.float().sum(dim=1) / support.sum(dim=1, keepdim=True).clamp_min(1)
        gaps = self._longest_gaps(masks, support)
        reliability = torch.cat([rates, gaps], dim=1)
        gate_input = torch.cat([*hidden, reliability[:, None].expand(-1, steps, -1)], dim=-1)
        raw_gates = self.gate(gate_input).masked_fill(~available, -1e4)
        gates = torch.softmax(raw_gates, dim=-1) * available.float()
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = sum(gates[:, :, index:index + 1] * hidden[index] for index in range(3))
        sequence_input = torch.cat([
            fused, available.float(), rates[:, None].expand(-1, steps, -1),
        ], dim=-1)
        sequence, _ = self.sequence(sequence_input)
        observed_any = available.any(dim=-1)
        sequence_summary = self.sequence_pool(sequence, observed_any)
        modality_summaries = [pool(value, mask) for pool, value, mask in zip(
            self.modality_pools, hidden, masks)]
        output = self.head(torch.cat([sequence_summary, *modality_summaries,
                                      reliability], dim=1))
        gate_summary = ((gates * observed_any.unsqueeze(-1)).sum(dim=1) /
                        observed_any.sum(dim=1, keepdim=True).clamp_min(1))
        return output[:, :3], 3 * torch.tanh(output[:, 3]), gate_summary
