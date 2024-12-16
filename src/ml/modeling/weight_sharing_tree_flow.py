from typing import Callable, Iterator
import torch

import torch.nn.functional as F
from torch import nn, optim
from src.ml.modeling.layers.unconditional_affine_coupling_flow_layer import (
    UnconditionalMaskedAffineFlowLayer,
)
from src.ml.modeling.layers.log_flow_layer import LogFlowLayer
from src.ml.modeling.normalizing_flow import NormalizingFlow


class Conditioner(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, z):
        res = self.linear_1(z)
        res = F.relu(res)
        res = self.linear_2(res)
        return z + res


class WeightSharingTreeFlow(NormalizingFlow):

    def __init__(
        self,
        input_example,
        mask_fraction: float,
        num_blocks: int,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    ):
        self.input_example = input_example

        dim = len(input_example["all_observed_clades"])

        flow_layers: list[nn.Module] = []
        flow_layers.append(LogFlowLayer())

        for _ in range(num_blocks):
            flow_layers.append(
                UnconditionalMaskedAffineFlowLayer(
                    mask=(torch.FloatTensor(dim).uniform_() < mask_fraction).float(),
                    translate=Conditioner(dim),
                    scale=Conditioner(dim),
                )
            )

        super().__init__(
            optimizer,
            flow_layers,
        )

        self.register_buffer(
            "all_observed_clades",
            torch.tensor(input_example["all_observed_clades"]),
        )

    def forward(self, batch):
        # transforms an input into latent space
        batch_mask = self.get_batch_mask(batch)

        transformed = self.encode(batch)

        for flow in self.flows:
            transformed["z"] = transformed["z"] * batch_mask

            result = flow.forward(**transformed)

            transformed["z"] = torch.nan_to_num(result["z"] * batch_mask)
            transformed["log_dj"] += result["log_dj"]

        return {**batch, **transformed}

    def inverse(self, batch):
        # transforms latent space into flow space
        batch_mask = self.get_batch_mask(batch)

        transformed = batch

        for flow in self.flows[::-1]:
            transformed["z"] = transformed["z"] * batch_mask

            result = flow.inverse(**transformed)

            transformed["z"] = torch.nan_to_num(result["z"] * batch_mask)

        return self.decode({**batch, **transformed})

    def get_batch_mask(self, batch):
        batch_size = len(batch["branch_lengths"])

        num_branch_lengths = len(batch["branch_lengths"][0])
        num_clades = len(self.all_observed_clades)

        all_observed_clades_list = sorted(self.all_observed_clades.tolist())

        batch_mask = torch.zeros(batch_size, num_clades).float()
        for i in range(batch_size):
            for j in range(num_branch_lengths):
                clade_index = all_observed_clades_list.index(batch["clades"][j][i])
                batch_mask[i, clade_index] = 1.0

        return batch_mask


    def encode(self, batch) -> dict:
        # binary encode clades

        batch_size = len(batch["branch_lengths"])
        num_branch_lengths = len(batch["branch_lengths"][0])
        num_clades = len(self.all_observed_clades)

        all_observed_clades_list = sorted(self.all_observed_clades.tolist())

        encoded_branch_lengths = torch.zeros(batch_size, num_clades)
        for i in range(batch_size):
            for j in range(num_branch_lengths):
                clade_index = all_observed_clades_list.index(batch["clades"][j][i])
                encoded_branch_lengths[i, clade_index] = batch["branch_lengths"][i][j]

        return {
            **batch,
            "z": encoded_branch_lengths,
            "log_dj": torch.zeros(len(batch["branch_lengths"])),
        }

    def decode(self, batch) -> dict:
        return {
            **batch,
            "branch_lengths": batch["z"],
            "log_dj": batch["log_dj"],
        }
