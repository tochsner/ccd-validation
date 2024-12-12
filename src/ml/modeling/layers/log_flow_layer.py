import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class LogFlowLayer(FlowLayer):

    def forward(self, z, **kwargs):
        return {
            "z": torch.log(z),
            "log_dj": -torch.sum(torch.log(z), dim=list(range(1, z.dim()))),
        }

    def inverse(self, z, **kwargs):
        return {
            "z": torch.exp(z),
            "log_dj": -torch.sum(z, dim=list(range(1, z.dim()))),
        }
