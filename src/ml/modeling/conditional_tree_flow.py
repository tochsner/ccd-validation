from typing import Callable, Iterator
import torch

import torch.nn.functional as F
from torch import nn, optim
from src.ml.modeling.layers.log_flow_layer import LogFlowLayer
from src.ml.modeling.layers.affine_coupling_flow_layer import MaskedAffineFlowLayer
from src.ml.modeling.layers.batch_norm_layer import BatchNormFlow
from src.ml.modeling.normalizing_flow import NormalizingFlow


class ContextEmmbedding(nn.Module):
    def __init__(self, context_dim: int, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(context_dim, embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x


class Conditioner(nn.Module):
    def __init__(self, dim: int, embedding_dim: int, context_embedding_dim: int):
        super().__init__()
        self.dim = dim

        self.linear_1 = nn.Linear(embedding_dim + context_embedding_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, z, context):
        res = self.linear_1(torch.cat([z, context], dim=1))
        res = F.relu(res)
        res = self.linear_2(res)
        return z[:, :self.dim] + res


class ConditionalTreeFlow(NormalizingFlow):

    def __init__(
        self,
        input_example,
        context_embedding_dim: int,
        mask_fraction: float,
        num_blocks: int,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    ):
        self.input_example = input_example

        dim = len(input_example["branch_lengths"])
        context_dim = len(input_example["taxa_names"]) * (
            len(input_example["taxa_names"]) - 1
        )

        context_embedding = ContextEmmbedding(context_dim, context_embedding_dim)

        flow_layers: list[nn.Module] = []
        flow_layers.append(LogFlowLayer())

        for _ in range(num_blocks):
            flow_layers.append(
                MaskedAffineFlowLayer(
                    mask=(torch.FloatTensor(dim).uniform_() < mask_fraction).float(),
                    translate=Conditioner(dim, dim, context_embedding_dim),
                    scale=Conditioner(dim, dim, context_embedding_dim),
                    context_embedding=context_embedding,
                )
            )

        super().__init__(
            optimizer,
            flow_layers,
        )

    def encode(self, batch) -> dict:
        # binary encode clades

        clade_bitstrings = torch.transpose(torch.stack(batch["clades"]), 0, 1)
        num_taxa = len(batch["taxa_names"])

        mask = 2 ** torch.arange(num_taxa)
        clade_encoding = (
            clade_bitstrings.unsqueeze(-1)
            .bitwise_and(mask)
            .ne(0)
            .byte()
            .flatten(1)
            .float()
        )

        return {
            **batch,
            "z": batch["branch_lengths"],
            "context": clade_encoding,
            "log_dj": torch.zeros(len(batch["branch_lengths"])),
        }

    def decode(self, batch) -> dict:
        return {
            **batch,
            "branch_lengths": batch["z"],
            "log_dj": batch["log_dj"],
        }
