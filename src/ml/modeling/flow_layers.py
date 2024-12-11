from abc import ABC
import torch.nn as nn


class FlowLayer(ABC, nn.Module):
    def forward(self, z, **kwargs):
        raise NotImplementedError

    def inverse(self, z, **kwargs):
        raise NotImplementedError

    def log_det_jacobian(self, z, **kwargs):
        raise NotImplementedError


class ConditionalFlowLayer(ABC, nn.Module):
    def forward(self, z, context, **kwargs):
        raise NotImplementedError

    def inverse(self, z, context, **kwargs):
        raise NotImplementedError

    def log_det_jacobian(self, z, context, **kwargs):
        raise NotImplementedError
