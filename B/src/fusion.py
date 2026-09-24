"""Shared masked attention and gated multimodal fusion model (M4)."""
from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(values).squeeze(-1).masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (values * weights.unsqueeze(-1)).sum(dim=1)


class Fusion(nn.Module):
    """Return polarity logits, intensity in [-3, 3], and modality gate weights."""
    def __init__(self, dim: int = 64, dropout: float = 0.25,
                 audio_indices=None, vision_indices=None):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        for name, width, indices in (("audio", 74, audio_indices),
                                     ("vision", 35, vision_indices)):
            mask = torch.ones(width)
            if indices is not None:
                if (not len(indices) or len(set(indices)) != len(indices) or
                        any(not isinstance(i, int) or not 0 <= i < width for i in indices)):
                    raise ValueError(f'invalid {name} channel indices')
                mask.zero_()
                mask[indices] = 1
            # Metadata carries masks; old checkpoints still load strictly.
            self.register_buffer(f'{name}_channels', mask, persistent=False)
        self.tproj = nn.Sequential(nn.Linear(384, dim), nn.LayerNorm(dim), nn.GELU())
        self.aproj = nn.Sequential(nn.Linear(74, dim), nn.LayerNorm(dim), nn.GELU())
        self.vproj = nn.Sequential(nn.Linear(35, dim), nn.LayerNorm(dim), nn.GELU())
        self.pools = nn.ModuleList([MaskedAttentionPool(dim) for _ in range(3)])
        self.gate = nn.Linear(dim * 3 + 3, 3)
        self.head = nn.Sequential(
            nn.Linear(dim * 4 + 3, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4),
        )

    def forward(self, text, audio, vision, text_mask, audio_mask, vision_mask):
        masks = [text_mask.bool(), audio_mask.bool(), vision_mask.bool()]
        inputs = [text.float(), audio.float() * self.audio_channels,
                  vision.float() * self.vision_channels]
        projections = [self.tproj, self.aproj, self.vproj]
        pooled = [pool(projection(values), mask)
                  for projection, pool, values, mask in zip(projections, self.pools, inputs, masks)]
        rates = torch.stack([mask.float().mean(dim=1) for mask in masks], dim=1)
        joined = torch.cat([*pooled, rates], dim=1)
        present = torch.stack([mask.any(dim=1) for mask in masks], dim=1)
        gates = torch.softmax(self.gate(joined).masked_fill(~present, -1e4), dim=1) * present.float()
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = sum(gates[:, i:i + 1] * pooled[i] for i in range(3))
        output = self.head(torch.cat([*pooled, fused, rates], dim=1))
        return output[:, :3], 3 * torch.tanh(output[:, 3]), gates


def predict_one(model: Fusion, sample: dict, device: str):
    model.eval()
    keys = ("text", "audio", "vision", "tmask", "amask", "vmask")
    tensors = [torch.from_numpy(sample[key].copy()).to(device) for key in keys]
    with torch.inference_mode():
        logits, intensity, gates = model(*tensors)
        return (torch.softmax(logits, dim=1).cpu().numpy()[0],
                float(intensity.cpu().numpy()[0]), gates.cpu().numpy()[0])
