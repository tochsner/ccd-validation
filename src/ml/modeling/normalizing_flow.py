from abc import ABC
from typing import Callable, Iterator
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

        self.scale = nn.Parameter(torch.Tensor(1, 1))

        self.save_hyperparameters()

    def encode(self, batch) -> dict:
        raise NotImplementedError
    
    def decode(self, batch) -> dict:
        raise NotImplementedError

    def forward(self, batch):
        transformed = self.encode(batch)

        for flow in self.flows:
            result = flow.forward(**transformed)
            transformed["z"] = result["z"]
            transformed["log_dj"] += result["log_dj"]

        return transformed
    
    def inverse(self, batch):
        transformed = batch

        for flow in self.flows[::-1]:
            result = flow.inverse(**transformed)
            transformed["z"] = result["z"]

        return self.decode(transformed)

    def get_loss(self, batch):
        transformed = self.forward(batch)
        
        z = transformed["z"]
        log_dj = transformed["log_dj"]

        log_pz = self.prior.log_prob(z).sum()
        log_px = log_dj + log_pz

        return -log_pz.mean()
    
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
