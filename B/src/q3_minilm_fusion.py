"""Online MiniLM with controlled parameter updates and shared Fusion."""
import torch
from torch import nn
from transformers import AutoModel
from B.src.fusion import Fusion


class MiniLMFusion(nn.Module):
    def __init__(self, encoder, dim=192, scope='all'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(encoder), local_files_only=True, attn_implementation='eager')
        self.encoder.requires_grad_(False)
        if scope == 'top2': self.encoder.encoder.layer[-2:].requires_grad_(True)
        elif scope == 'all':
            self.encoder.encoder.requires_grad_(True)
            self.encoder.embeddings.requires_grad_(True)
        elif scope != 'frozen': raise ValueError(scope)
        self.fusion = Fusion(dim=dim)
        # Uniform module.train() dropout behavior for every experimental arm.
        self.scope = scope

    def forward(self, ids, attention, audio, vision, tm, am, vm):
        text = self.encoder(input_ids=ids.long(), attention_mask=attention.long(),
                            token_type_ids=torch.zeros_like(ids).long()).last_hidden_state
        return self.fusion(text, audio, vision, tm, am, vm)

    def optimizer_groups(self):
        groups = {1e-6: [], 5e-6: [], 2e-5: []}
        for name, p in self.encoder.named_parameters():
            if p.requires_grad:
                lr = 1e-6 if name.startswith('embeddings.') else 2e-5 if any(f'layer.{i}.' in name for i in [4, 5]) else 5e-6
                groups[lr].append(p)
        return [{'params': ps, 'lr': lr} for lr, ps in groups.items() if ps] + [{'params': self.fusion.parameters(), 'lr': 2e-4}]
