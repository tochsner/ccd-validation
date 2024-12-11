from abc import ABC
from typing import Callable, Iterator
import torch
import lightning as pl
import torch.nn as nn
from torch import Tensor, nn, optim


class NormalizingFlow(ABC, pl.LightningModule):
    def __init__(
        self,
        loss: Callable[[torch.Tensor, Tensor], Tensor],
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
        flows: list,
    ):
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer

        self.flows = nn.ModuleList(flows)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def encode(self, batch) -> dict:
        raise NotImplementedError

    def forward(self, batch):
        encoded = self.encode(batch)

        for flow in self.flows:
            encoded = flow.forward(**encoded)

        return encoded

    def get_loss(self, batch):
        encoded = self.encode(batch)
        
        z = encoded["z"]
        log_dj = encoded["log_dj"]

        log_pz = self.prior.log_prob(z).sum()
        log_px = log_dj + log_pz

        return -log_px.mean()

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
