from typing import Optional
from torch import Tensor, nn
import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class BatchNormFlowLayer(FlowLayer):

    def __init__(self, mask: Tensor, dim: int, momentum: float = 0.95):
        super().__init__()

        self.register_buffer("mask", mask)

        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_std", torch.ones(dim))

        self.momentum = momentum

    def forward(self, z, batch_mask=None, **kwargs):
        # transforms a flow into latent space

        if batch_mask is None:
            batch_mask = torch.ones_like(z)

        if self.training:
            batch_mean = torch.nan_to_num(
                (self.mask * batch_mask * z).sum(dim=0)
                / (self.mask * batch_mask).sum(dim=0)
            )
            batch_std = torch.nan_to_num(
                (self.mask * batch_mask * (z - batch_mean) ** 2).sum(dim=0)
                / (self.mask * batch_mask).sum(dim=0)
            ).sqrt()

            self.running_mean.data = batch_mean + self.running_mean * (
                1 - self.momentum
            )
            self.running_std.data = (
                batch_std * self.momentum * self.mask
                + self.running_std * (1 - self.momentum)
            )

            if (self.running_mean == torch.inf).any():
                raise ValueError("NaN")

            if self.running_std.min() < 0:
                raise ValueError("NaN")

            if (self.running_std == torch.inf).any():
                raise ValueError("NaN")

        z = (z - self.running_mean) / self.running_std
        log_det = 0.0 * -self.running_std.log()

        if (z == torch.inf).any():
            raise ValueError("NaN")

        if (log_det == torch.inf).any():
            raise ValueError("NaN")

        return {
            "z": z,
            "log_dj": log_det,
        }

    def inverse(self, z, **kwargs):
        # transforms latent space into flow space

        if self.training:
            self.running_std.data = 1 / z.std(
                dim=0
            ) * self.momentum + self.running_std * (1 - self.momentum)
            self.running_mean.data = (
                -z.mean(dim=0) * self.running_std
            ) * self.momentum + self.running_mean * (1 - self.momentum)

        z = z * self.running_std + self.running_mean
        log_det = self.running_std.log()

        return {
            "z": z,
            "log_dj": log_det,
        }
