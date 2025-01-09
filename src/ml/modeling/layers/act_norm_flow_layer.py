from torch import Tensor, nn
import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class ActNormFlowLayer(FlowLayer):

    def __init__(self, mask: Tensor, dim: int):
        super().__init__()

        self.register_buffer("mask", mask)

        self.mean = nn.Parameter(torch.zeros(dim))
        self.std = nn.Parameter(torch.ones(dim))

    def forward(self, z, **kwargs):
        # transforms a flow into latent space

        z = (z - self.mean) / self.std
        log_det = -self.std.log()

        return {
            "z": z,
            "log_dj": log_det,
        }

    def inverse(self, z, **kwargs):
        # transforms latent space into flow space

        z = z * self.running_std + self.running_mean
        log_det = self.running_std.log()

        return {
            "z": z,
            "log_dj": log_det,
        }
