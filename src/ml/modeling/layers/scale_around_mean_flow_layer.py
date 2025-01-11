import torch
from torch import nn
from src.ml.modeling.layers.flow_layers import FlowLayer


class ScaleAroundMeanFlowLayer(FlowLayer):

    def __init__(self, dim: int, momentum: float = 0.8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

        self.register_buffer("running_mean", torch.zeros((1, dim)))

        self.momentum = momentum

    def forward(self, z, additional_mask, **kwargs):
        if self.training:
            mean = torch.sum(additional_mask * z, dim=0, keepdim=True) / torch.sum(
                additional_mask, dim=0, keepdim=True
            )
        else:
            mean = self.running_mean
        
        z = (z - mean) * self.scale + mean

        # (z - mean) * self.scale + mean = z * scale - mean * scale + mean = z * scale + mean * (1 - scale)

        log_dj = torch.log(
            torch.abs(
                self.scale
                + (torch.ones_like(z) - self.scale)
                / torch.sum(additional_mask, dim=0, keepdim=True)
            )
        )

        if self.training:
            with torch.no_grad():
                self.running_mean.data = (
                    self.momentum * self.running_mean + (1 - self.momentum) * mean
                )

        return {
            "z": z,
            "log_dj": log_dj,
        }

    def inverse(self, z, additional_mask, **kwargs):
        mean = torch.sum(additional_mask * z, dim=0, keepdim=True) / torch.sum(
            additional_mask, dim=0, keepdim=True
        )
        z = (z - mean) / self.scale + mean
        return {
            "z": z,
        }
