from torch import Tensor
from torch import nn
import torch
import numpy as np
from src.ml.modeling.flow_layers import ConditionalFlowLayer


class MaskedAffineFlowLayer(ConditionalFlowLayer):

    def __init__(self, mask: Tensor, translate: nn.Module, scale: nn.Module, context_embedding: nn.Module):
        super().__init__()

        self.translate = translate
        self.scale = scale
        self.mask = mask
        self.context_embedding = context_embedding

    def forward(self, z, context, **kwargs):
        z_masked = self.mask * z

        context = self.context_embedding(context)

        scale = self.scale(z_masked, context)
        translation = self.translate(z_masked, context)

        z = z_masked + (1 - self.mask) * (z * torch.exp(scale) + translation)
        log_det = torch.sum((1 - self.mask) * scale, dim=list(range(1, self.mask.dim())))

        return {
            "z": z,
            "log_dj": log_det,
        }

    def inverse(self, z, context, **kwargs):
        z_masked = self.mask * z

        context = self.context_embedding(context)

        scale = self.scale(z_masked, context)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)

        trans = self.translate(z_masked, context)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        z = z_masked + (1 - self.mask) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.mask) * scale, dim=list(range(1, self.mask.dim())))

        return {
            "z": z,
            "log_dj": log_det,
        }
