from typing import Callable, Iterator, Literal, Optional
import torch

from torch import nn, optim, tensor
from src.ml.modeling.layers.random_flow_layer import RandomFlowLayer
from src.ml.modeling.layers.scale_around_mean_flow_layer import ScaleAroundMeanFlowLayer
from src.ml.modeling.layers.unconditional_affine_coupling_flow_layer import (
    UnconditionalMaskedAffineFlowLayer,
)
from src.ml.modeling.layers.log_flow_layer import LogFlowLayer
from src.ml.modeling.layers.inverse_sigmoid_flow_layer import InverseSigmoidFlowLayer
from src.ml.modeling.normalizing_flow import NormalizingFlow


class Conditioner(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.layers = nn.Sequential()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(dim, dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                )
            elif i < num_layers - 1:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(dim // 2, dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                )
            else:
                # we don't apply relu or dropout to the last layer
                self.layers.append(
                    nn.Linear(dim // 2, dim),
                )

                # we initialize the last layer with 0 weights such that it
                # is the identity at the beginning (see the glow paper)
                self.layers[-1].weight.data.fill_(0)
                self.layers[-1].bias.data.fill_(0)

    def forward(self, z):
        return z + self.layers(z)


class LogNormalHeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(tensor(0.0))
        self.log_scale = nn.Parameter(tensor(0.0))

        self.register_buffer("has_been_initialized", tensor(0.0))

    def get_log_likelihood(self, tree_height, **kwargs):
        if self.training and not self.has_been_initialized:
            self.mean.data = tree_height.log().mean(dim=0)
            self.log_scale.data = torch.log(tree_height.log().std(dim=0))
            self.has_been_initialized.data = tensor(1.0)

        return torch.distributions.LogNormal(self.mean, self.log_scale.exp()).log_prob(
            tree_height
        )

    def sample(self, z, **kwargs):
        return torch.distributions.LogNormal(self.mean, self.log_scale.exp()).sample(
            (len(z),)
        )

    def mode(self, **kwargs):
        return torch.exp(self.mean)


class WeightSharingTreeFlow(NormalizingFlow):

    def __init__(
        self,
        input_example,
        mask_fraction: float,
        num_blocks: int,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
        height_model_name: Optional[Literal["gamma"]] = None,
        encoding: Literal["fractions", "absolute_positive"] = "fractions",
        conditioner_num_layers: int = 2,
        conditioner_dropout: float = 0.25,
    ):
        self.input_example = input_example

        dim = len(input_example["all_observed_clades"])

        flow_layers: list[nn.Module] = []

        match encoding:
            case "fractions":
                flow_layers.append(InverseSigmoidFlowLayer())
                flow_layers.append(ScaleAroundMeanFlowLayer(dim))
            case "absolute_positive":
                flow_layers.append(LogFlowLayer())

        for _ in range(num_blocks):
            mask = (torch.FloatTensor(dim).uniform_() < mask_fraction).float()
            flow_layers.append(
                UnconditionalMaskedAffineFlowLayer(
                    mask,
                    translate=Conditioner(
                        dim, conditioner_num_layers, conditioner_dropout
                    ),
                    scale=Conditioner(dim, conditioner_num_layers, conditioner_dropout),
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
        self.register_buffer(
            "taxa_names",
            torch.tensor(
                [
                    sum(ord(x_) * 10**i for i, x_ in enumerate(x))
                    for x in input_example["taxa_names"]
                ]
            ),
        )

        self.sorted_observed_clades = torch.tensor(sorted(self.all_observed_clades))

        match height_model_name:
            case "lognormal":
                self.height_model = LogNormalHeightModel()
            case _:
                self.height_model = None

        self.log_alpha = nn.Parameter(torch.randn(dim))
        self.log_beta = nn.Parameter(torch.randn(dim))

    def _repace_with_clade_indices(self, clades):
        return (
            (clades.unsqueeze(-1) == self.sorted_observed_clades)
            .to(torch.long)
            .argmax(dim=-1, keepdim=True)
            .squeeze(-1)
        )

    def forward(self, batch):
        # transforms an input into latent space
        batch_mask = self.get_batch_mask(batch)

        transformed = self.encode(batch)

        for flow in self.flows:
            transformed["z"] = transformed["z"] * batch_mask

            result = flow.forward(**transformed, additional_mask=batch_mask)

            transformed["z"] = torch.nan_to_num(result["z"] * batch_mask)
            transformed["log_dj"] += torch.sum(
                torch.nan_to_num(result["log_dj"] * batch_mask),
                dim=list(range(1, result["log_dj"].dim())),
            )

            if (transformed["z"] == torch.inf).any():
                raise ValueError("NaN")

        return {**batch, **transformed}

    def inverse(self, batch):
        # transforms latent space into flow space
        batch_mask = self.get_batch_mask(batch)

        transformed = batch

        for flow in self.flows[::-1]:
            transformed["z"] = transformed["z"] * batch_mask

            result = flow.inverse(**transformed, additional_mask=batch_mask)

            transformed["z"] = torch.nan_to_num(result["z"] * batch_mask)

        return self.decode({**batch, **transformed})

    def get_batch_mask(self, batch):
        batch_size = len(batch["branch_lengths"])
        num_clades = len(self.all_observed_clades)

        indices_to_mask = self._repace_with_clade_indices(
            torch.stack(batch["clades"]).T
        )
        batch_mask = torch.zeros(batch_size, num_clades).scatter_(
            1, indices_to_mask, 1.0
        )

        return batch_mask

    def encode(self, batch) -> dict:
        self._verify_taxa_names(batch)

        batch_size = len(batch["branch_lengths"])
        num_clades = len(self.all_observed_clades)

        indices_to_populate = self._repace_with_clade_indices(
            torch.stack(batch["clades"]).T
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
        populate_indices = self._repace_with_clade_indices(
            torch.stack(batch["clades"]).T
        )
        branch_lengths = torch.gather(batch["z"], 1, populate_indices)

        return {
            **batch,
            "branch_lengths": branch_lengths,
            "log_dj": batch["log_dj"],
        }

    def get_base_log_likelihood(self, batch):
        complete_log_likelihood = self.base_distribution.log_prob(batch["z"])

        masked_log_likelihood = torch.nan_to_num(
            complete_log_likelihood * self.get_batch_mask(batch)
        )
        log_likelihood_per_batch = masked_log_likelihood.sum(
            dim=list(range(1, batch["z"].dim()))
        )

        return log_likelihood_per_batch

    def get_log_likelihood(self, batch):
        log_likelihood = super().get_log_likelihood(batch)

        if self.height_model:
            encoded = self.encode(batch)
            height_log_prob = self.height_model.get_log_likelihood(**encoded)
            log_likelihood += height_log_prob

        return log_likelihood

    def sample(self, batch):
        sample = super().sample(batch)

        if self.height_model:
            encoded = self.encode(batch)
            sample["tree_height"] = self.height_model.sample(**encoded)

        return sample

    def _verify_taxa_names(self, batch):
        # encoded_batch_taxa_names = torch.tensor(
        #     [[ord(t) for t in ts] for ts in batch["taxa_names"]]
        # ).T
        # if not torch.all(encoded_batch_taxa_names == self.taxa_names):
            ...
            # raise ValueError("Taxa names do not match.")
