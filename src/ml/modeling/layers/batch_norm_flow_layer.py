from typing import Optional
from torch import Tensor, nn
import torch
from src.ml.modeling.layers.flow_layers import FlowLayer


class BatchNormFlowLayer(FlowLayer):

    def __init__(self, dim: int, mask, momentum: float = 0.8):
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
            with torch.no_grad():
                batch_mean = (
                    self.mask
                    * torch.nan_to_num(
                        (batch_mask * z).sum(dim=0) / batch_mask.sum(dim=0)
                    )
                    + (1 - self.mask) * self.running_mean
                )
                batch_std = (
                    self.mask
                    * torch.nan_to_num(
                        (batch_mask * (z - batch_mean) ** 2).sum(dim=0)
                        / batch_mask.sum(dim=0)
                    ).sqrt()
                    + (1 - self.mask) * self.running_std
                )

                # we sometimes have costant features
                batch_std = torch.where(batch_std > 0, batch_std, 1.0)

                self.running_mean.data = (
                    self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
                )
                self.running_std.data = (
                    self.momentum * batch_std + (1 - self.momentum) * self.running_std
                )
        else:
            batch_mean = self.running_mean
            batch_std = self.running_std

        z = (z - batch_mean) / batch_std
        log_det = -batch_std.log().unsqueeze(0).repeat(len(z), 1)

        if (self.running_mean == torch.inf).any():
            raise ValueError("NaN")

        if self.running_std.min() < 0:
            raise ValueError("NaN")

        if (self.running_std == torch.inf).any():
            raise ValueError("NaN")

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

        z = z * self.running_std + self.running_mean
        log_det = self.running_std.log()

        return {
            "z": z,
            "log_dj": log_det,
        }
