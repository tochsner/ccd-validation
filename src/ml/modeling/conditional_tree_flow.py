from typing import Callable, Iterator
import torch

import torch.nn.functional as F
from torch import nn, optim
from src.ml.modeling.layers.affine_coupling_flow_layer import MaskedAffineFlowLayer
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
    def __init__(self, dim: int, context_dim: int, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim + context_dim, embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, dim)

    def forward(self, z, y):
        z = self.linear_1(torch.cat([z, y], dim=1))
        z = F.relu(z)
        z = self.linear_2(z)
        return z


class TranslationModule(nn.Module):
    def __init__(self, dim: int, context_dim: int, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim + context_dim, embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, dim)

    def forward(self, z, y):
        z = self.linear_1(torch.cat([z, y], dim=1))
        z = F.relu(z)
        z = self.linear_2(z)
        return z


class ConditionalTreeFlow(NormalizingFlow):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        context_embedding_size: int,
        mask_fraction: float,
        num_blocks: int,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    ):
        flow_layers: list[nn.Module] = []

        for _ in range(num_blocks):
            flow_layers.append(
                MaskedAffineFlowLayer(
                    mask=(torch.FloatTensor(dim).uniform_() < mask_fraction).float(),
                    context_embedding=ContextEmmbedding(
                        context_dim, context_embedding_size
                    ),
                    translate=TranslationModule(dim, context_embedding_size, dim),
                    scale=ScalingModule(dim, context_embedding_size, dim),
                )
            )

        super().__init__(
            optimizer,
            flow_layers,
        )

    def encode(self, batch) -> dict:
        return {
            "z": batch["branch_lengths"],
            "context": batch["clades_one_hot"],
            "log_dj": torch.zeros(len(batch["branch_lengths"])),
        }

    def decode(self, batch) -> dict:
        return {
            "branch_lengths": batch["z"],
            "clades_one_hot": batch["context"],
            "log_dj": batch["log_dj"],
        }
