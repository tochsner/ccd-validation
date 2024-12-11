import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ml.modeling.flow_layers import ConditionalFlowLayer


class PlanarFlowLayer(ConditionalFlowLayer):

    def __init__(self, dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        
        self.h = torch.tanh
        self.hp = lambda x: 1 - torch.tanh(x) ** 2

    def forward(self, z, y, **kwargs):
        f_z = F.linear(z, self.weight, self.bias)
        return {
            "z": z + self.scale * self.h(f_z),
            "y": y,
            "log_dj": self.log_det_jacobian(z, y),
        }

    def log_det_jacobian(self, z, y, **kwargs):
        f_z = F.linear(z, self.weight, self.bias)
        psi = self.hp(f_z) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)
