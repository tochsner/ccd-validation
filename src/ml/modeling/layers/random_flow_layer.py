import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class RandomFlowLayer(FlowLayer):

    def __init__(self, dim: int):
        super().__init__()
        # we generate a random invertible matrix

        m = torch.randn((dim, dim)) / 100
        mx = torch.sum(torch.abs(m), dim=1)
        m[range(len(m)), range(len(m))] = mx

        # make determinant 1

        m = torch.exp(-m.trace() / dim) * torch.matrix_exp(m)

        self.register_buffer("m", m)
        self.register_buffer("m_inv", torch.inverse(m))

    def forward(self, z, **kwargs):
        return {
            "z": torch.einsum('ij,bj->bi', self.m, z),
            "log_dj": torch.zeros_like(z),
        }

    def inverse(self, z, **kwargs):
        return {
            "z": torch.einsum('ij,bj->bi', self.m, z),
            "log_dj": torch.zeros_like(z),
        }
