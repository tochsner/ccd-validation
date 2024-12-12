from torch import Tensor
from torch import nn
import torch
import numpy as np
from src.ml.modeling.layers.flow_layers import ConditionalFlowLayer


class MaskedAffineFlowLayer(ConditionalFlowLayer):

    def __init__(
        self,
        mask: Tensor,
        translate: nn.Module,
        scale: nn.Module,
        context_embedding: nn.Module,
    ):
        super().__init__()

        self.translate = translate
        self.scale = scale
        self.mask = mask
        self.context_embedding = context_embedding

    def forward(self, z, context, **kwargs):
        z_masked = self.mask * z

        embedded_context = self.context_embedding(context)

        scale = self.scale(z_masked, embedded_context)
        translation = self.translate(z_masked, embedded_context)

        scale = scale * (1 - self.mask)
        translation = translation * (1 - self.mask)

        z = (z + translation) * torch.exp(scale)
        log_det = torch.sum(scale, dim=list(range(1, self.mask.dim())))

        return {
            "z": z,
            "context": context,
            "log_dj": log_det,
        }

    def inverse(self, z, context, **kwargs):
        z_masked = self.mask * z

        embedded_context = self.context_embedding(context)

        scale = self.scale(z_masked, embedded_context)
        translation = self.translate(z_masked, embedded_context)

        scale = scale * (1 - self.mask)
        translation = translation * (1 - self.mask)

        z = (z * torch.exp(-scale)) - translation
        log_det = -torch.sum(scale, dim=list(range(1, self.mask.dim())))

        return {
            "z": z,
            "context": context,
            "log_dj": log_det,
        }
