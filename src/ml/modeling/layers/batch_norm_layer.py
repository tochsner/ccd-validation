import torch
import torch.nn as nn
from src.ml.modeling.layers.flow_layers import FlowLayer


class BatchNormFlow(FlowLayer):

    def __init__(self, dim, momentum=0.95, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        # Running batch statistics
        self.r_mean = torch.zeros(dim)
        self.r_var = torch.ones(dim)

        self.momentum = momentum
        self.eps = eps

        # Trainable scale and shift (cf. original paper)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z, **kwargs):
        if self.training:
            # Current batch stats
            self.b_mean = z.mean(0)
            self.b_var = (z - self.b_mean).pow(2).mean(0) + self.eps

            # Running mean and var
            self.r_mean = self.momentum * self.r_mean + (
                (1 - self.momentum) * self.b_mean
            )
            self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)

            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var

        x_hat = (z - mean) / var.sqrt()
        y = self.gamma * x_hat + self.beta

        return {
            "z": y,
            "log_dj": self.log_abs_det_jacobian(z),
        }

    def inverse(self, z, **kwargs):
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var

        x_hat = (z - self.beta) / self.gamma
        y = x_hat * var.sqrt() + mean

        return {
            "z": y,
        }

    def log_abs_det_jacobian(self, z):
        # Here we only need the variance
        mean = z.mean(0)
        var = (z - mean).pow(2).mean(0) + self.eps
        log_det = torch.log(self.gamma) - 0.5 * torch.log(var + self.eps)
        return log_det.sum(dim=list(range(1, log_det.dim())))
