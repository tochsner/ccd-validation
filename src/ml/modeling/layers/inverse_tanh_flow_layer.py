from typing import Optional
import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class InverseTanhFlowLayer(FlowLayer):

    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, z, **kwargs):
        return {
            "z": torch.atanh(z * 2 - 1),
            "log_dj": -torch.log(torch.abs(2 * z * (z + 1))),
        }

    def inverse(self, z, **kwargs):
        return {
            "z": (1 + torch.tanh(z)) / 2,
            "log_dj": torch.log(torch.abs(0.5 / torch.cosh(z)**2)),
        }
