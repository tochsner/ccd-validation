from typing import Callable, Iterator, Literal, Optional
import torch

import torch.nn.functional as F
from torch import nn, optim, tensor
from src.ml.modeling.layers.unconditional_affine_coupling_flow_layer import (
    UnconditionalMaskedAffineFlowLayer,
)
from src.ml.modeling.layers.log_flow_layer import LogFlowLayer
from src.ml.modeling.layers.inverse_sigmoid_flow_layer import InverseSigmoidFlowLayer
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


class UnimodalGammaHeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.concentration = nn.Parameter(tensor(0.0))
        self.rate = nn.Parameter(tensor(2.0))

    def get_log_likelihood(self, tree_height, **kwargs):
        # we ensure that we have a mode by forcing the concentration to be greater than 1
        concentration = 1.0 + torch.exp(self.concentration)
        return (
            torch.xlogy(concentration, self.rate)
            + torch.xlogy(concentration - 1, tree_height)
            - self.rate * tree_height
            - torch.lgamma(concentration)
        )

    def sample(self, sample_shape):
        concentration = 1.0 + torch.exp(self.concentration)
        return torch.distributions.Gamma(concentration, self.rate).sample(sample_shape)

    def mode(self):
        concentration = 1.0 + torch.exp(self.concentration)
        return float((concentration - 1.0) / self.rate)


class WeightSharingTreeFlow(NormalizingFlow):

    def __init__(
        self,
        input_example,
        mask_fraction: float,
        num_blocks: int,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
        height_model_name: Optional[Literal["gamma"]] = None,
        encoding: Literal["fractions", "absolute_positive"] = "fractions",
    ):
        self.input_example = input_example

        dim = len(input_example["all_observed_clades"])

        flow_layers: list[nn.Module] = []

        match encoding:
            case "fractions":
                flow_layers.append(InverseSigmoidFlowLayer())
            case "absolute_positive":
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

        self.observed_clade_indices = {
            int(clade): i for i, clade in enumerate(sorted(self.all_observed_clades))
        }

        match height_model_name:
            case "unimodal_gamma":
                self.height_model = UnimodalGammaHeightModel()
            case _:
                self.height_model = None

    def forward(self, batch):
        # transforms an input into latent space
        batch_mask = self.get_batch_mask(batch)

        transformed = self.encode(batch)

        for flow in self.flows:
            transformed["z"] = transformed["z"] * batch_mask

            result = flow.forward(**transformed)

            transformed["z"] = torch.nan_to_num(result["z"] * batch_mask)
            transformed["log_dj"] += torch.sum(
                torch.nan_to_num(result["log_dj"] * batch_mask),
                dim=list(range(1, result["log_dj"].dim())),
            )

        if self.height_model:
            height_log_prob = self.height_model.get_log_likelihood(**transformed)
            transformed["log_dj"] += height_log_prob

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
        num_clades = len(self.all_observed_clades)

        indices_to_mask = torch.stack(batch["clades"]).T.apply_(
            lambda x: self.observed_clade_indices.get(x, 0)
        )
        batch_mask = torch.zeros(batch_size, num_clades).scatter_(
            1, indices_to_mask, 1.0
        )

        return batch_mask

    def encode(self, batch) -> dict:
        batch_size = len(batch["branch_lengths"])
        num_clades = len(self.all_observed_clades)

        indices_to_populate = torch.stack(batch["clades"]).T.apply_(
            lambda x: self.observed_clade_indices.get(x, 0)
        )
        encoded_branch_lengths = torch.zeros(batch_size, num_clades).scatter_(
            1, indices_to_populate, batch["branch_lengths"]
        )

        return {
            **batch,
            "z": encoded_branch_lengths,
            "log_dj": torch.zeros(len(batch["branch_lengths"])),
        }

    def decode(self, batch) -> dict:
        populate_indices = torch.stack(batch["clades"]).T.apply_(
            lambda x: self.observed_clade_indices.get(x, 0)
        )
        branch_lengths = torch.gather(batch["z"], 1, populate_indices)

        return {
            **batch,
            "branch_lengths": branch_lengths,
            "log_dj": batch["log_dj"],
        }

    def get_log_likelihood(self, batch):
        log_likelihood = super().get_log_likelihood(batch)

        if self.height_model:
            height_log_prob = self.height_model.get_log_likelihood(**batch)
            log_likelihood += height_log_prob

        return log_likelihood

    def sample(self, batch):
        sample = super().sample(batch)

        if self.height_model:
            sample["tree_height"] = self.height_model.sample((batch["branch_lengths"].shape[0],))

        return sample
