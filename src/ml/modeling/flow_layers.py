from abc import ABC
import torch.nn as nn


class FlowLayer(ABC, nn.Module):
    def forward(self, z):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError

    def log_det_jacobian(self, z):
        raise NotImplementedError


class ConditionalFlowLayer(ABC, nn.Module):
    def forward(self, z, y):
        raise NotImplementedError

    def inverse(self, z, y):
        raise NotImplementedError

    def log_det_jacobian(self, z, y):
        raise NotImplementedError
