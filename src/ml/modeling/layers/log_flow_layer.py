import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class LogFlowLayer(FlowLayer):

    def forward(self, z, **kwargs):
        return {
            "z": torch.log(z),
            "log_dj": -torch.log(torch.abs(z))
        }

    def inverse(self, z, **kwargs):
        return {"z": torch.exp(z), "log_dj": -z}
