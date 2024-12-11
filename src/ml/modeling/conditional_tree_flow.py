from typing import Callable, Iterator
import torch

import torch.nn.functional as F
from torch import Tensor, nn, optim
from src.ml.modeling.affine_coupling_flow_layer import MaskedAffineFlowLayer
from src.ml.modeling.normalizing_flow import NormalizingFlow


class ContextEmmbedding(nn.Module):
    def __init__(self, context_dim: int, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(context_dim, embedding_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class ScalingModule(nn.Module):
    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.linear = nn.Linear(dim + context_dim, dim)

    def forward(self, z, y):
        z = self.linear(torch.cat([z, y], dim=1))
        return z


class TranslationModule(nn.Module):
    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.linear = nn.Linear(dim + context_dim, dim)

    def forward(self, z, y):
        z = self.linear(torch.cat([z, y], dim=1))
        return z


class ConditionalTreeFlow(NormalizingFlow):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        context_embedding_size: int,
        mask_fraction: float,
        num_blocks: int,
        loss: Callable[[torch.Tensor, torch.Tensor], Tensor],
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    ):
        flow_layers = [
            MaskedAffineFlowLayer(
                mask=(torch.FloatTensor(dim).uniform_() > mask_fraction).float(),
                context_embedding=ContextEmmbedding(
                    context_dim, context_embedding_size
                ),
                translate=TranslationModule(dim, context_embedding_size),
                scale=ScalingModule(dim, context_embedding_size),
            )
            for _ in range(num_blocks)
        ]

        super().__init__(
            loss,
            optimizer,
            flow_layers,
        )

    def encode(self, batch) -> dict:
        return {
            "z": batch["branch_lengths"],
            "context": batch["clades_one_hot"],
            "log_dj": torch.zeros(len(batch["branch_lengths"]), device=self.device),
        }
