from torch import Tensor
from torch import nn
import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class UnconditionalMaskedAffineFlowLayer(FlowLayer):

    def __init__(
        self,
        mask: Tensor,
        translate: nn.Module,
        scale: nn.Module,
    ):
        super().__init__()

        self.translate = translate
        self.scale = scale
        
        self.register_buffer("mask", mask)
        self.scaling = nn.Parameter(torch.ones(mask.shape))

    def forward(self, z, **kwargs):
        z_masked = z * self.mask

        scale = self.scale(z_masked)
        translation = self.translate(z_masked)

        scale = scale * (1 - self.mask)
        scale = self.scaling * torch.tanh(scale)
        translation = translation * (1 - self.mask)

        z = (z * torch.exp(-scale)) - translation

        log_det = -scale

        return {
            "z": z,
            "log_dj": log_det
        }

    def inverse(self, z, **kwargs):
        z_masked = z * self.mask

        scale = self.scale(z_masked)
        scale = self.scaling * torch.tanh(scale)
        translation = self.translate(z_masked)

        scale = scale * (1 - self.mask)
        translation = translation * (1 - self.mask)

        z = (z + translation) * torch.exp(scale)
        log_det = scale

        return {
            "z": z,
            "log_dj": log_det,
        }
