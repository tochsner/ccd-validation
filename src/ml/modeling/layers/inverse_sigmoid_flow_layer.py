from typing import Optional
import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class InverseSigmoidFlowLayer(FlowLayer):

    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, z, **kwargs):
        return {
            "z": torch.log(z / (1 - z)),
            "log_dj": -torch.log(torch.abs(z - z*z)),
        }

    def inverse(self, z, **kwargs):
        return {
            "z": torch.sigmoid(z),
            "log_dj": torch.log(torch.abs(torch.sigmoid(z) * (1 - torch.sigmoid(z)))),
        }
