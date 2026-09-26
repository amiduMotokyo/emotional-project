"""Round-seven online model. Projection is training-only; prediction API is unchanged."""
from torch import nn
from B.src.q3_minilm_fusion import MiniLMFusion
from B.src.q3_temporal_primary_fusion import TemporalPrimaryFusion


class Round7Model(MiniLMFusion):
    def __init__(self, encoder, dim=64, scope='top2', fusion='original', contrastive=False, balanced=False):
        super().__init__(encoder, dim=dim, scope=scope)
        self.last_features = None
        if fusion != 'original': self.fusion = TemporalPrimaryFusion(mode=fusion, dim=dim)
        else: self.fusion.head[-1].register_forward_pre_hook(self._capture)
        self.projector = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64)) if contrastive else None

    def _capture(self, module, inputs): self.last_features = inputs[0]

    def forward(self, *args):
        result = super().forward(*args)
        if isinstance(self.fusion, TemporalPrimaryFusion): self.last_features = self.fusion.last_features
        return result

    def optimizer_groups(self):
        groups = super().optimizer_groups()
        if self.projector is not None: groups.append({'params': self.projector.parameters(), 'lr': 2e-4})
        return groups
