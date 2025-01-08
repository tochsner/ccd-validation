from abc import ABC
from typing import Callable, Iterator, Literal, Optional
import torch
import lightning as pl
import torch.nn as nn
from torch import nn, optim



class NormalizingFlow(ABC, pl.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
        flows: list,
    ):
        super().__init__()

        self.optimizer = optimizer

        self.flows = nn.ModuleList(flows)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        self.save_hyperparameters()

    def encode(self, batch) -> dict:
        raise NotImplementedError

    def decode(self, batch) -> dict:
        raise NotImplementedError

    def forward(self, batch):
        # transforms an input into latent space
        transformed = self.encode(batch)

        for flow in self.flows:
            result = flow.forward(**transformed)
            transformed["z"] = result["z"]
            transformed["log_dj"] += torch.sum(
                result["log_dj"], dim=list(range(1, result["log_dj"].dim()))
            )

        if self.height_model:
            height_log_prob = self.height_model.forward(**transformed)
            transformed["log_dj"] += height_log_prob

        return {**batch, **transformed}

    def inverse(self, batch):
        # transforms latent space into flow space
        transformed = batch

        for flow in self.flows[::-1]:
            result = flow.inverse(**transformed)
            transformed["z"] = result["z"]

        return self.decode({**batch, **transformed})

    def get_log_likelihood(self, batch):
        transformed = self.forward(batch)

        z = transformed["z"]
        log_dj = transformed["log_dj"]

        log_pz = self.prior.log_prob(z).sum(dim=list(range(1, z.dim())))
        log_px = log_dj + log_pz

        return log_px

    def get_loss(self, batch):
        return -self.get_log_likelihood(batch).mean()

    def sample(self, batch):
        transformed = self.encode(batch)

        prior_sample = self.prior.sample(transformed["z"].shape)
        transformed["z"] = prior_sample

        return self.inverse({**batch, **transformed})

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f"val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f"test_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
