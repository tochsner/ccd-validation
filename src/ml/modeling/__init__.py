from functools import partial
from typing import Callable, Iterator

import lightning.pytorch as pl
from torch import Tensor, nn, optim

def model_factory(
    name: str,
    loss: Callable[[Tensor, Tensor], Tensor],
    optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    **kwargs,
) -> pl.LightningModule:
    """Factory function for models."""
    match name:
        case _:
            raise ValueError(f"Unknown model {name}.")


def loss_factory(name: str, **kwargs) -> Callable[[Tensor, Tensor], Tensor]:
    """Factory function for loss functions."""
    match name:
        case "mse":
            return partial(nn.functional.mse_loss, **kwargs)
        case _:
            raise ValueError(f"Unknown loss {name}.")


def optimizer_factory(
    name: str, **kwargs
) -> Callable[[Iterator[nn.Parameter]], optim.Optimizer]:
    """Factory function for optimizers."""
    match name:
        case "adam":
            return partial(optim.Adam, **kwargs)
        case _:
            raise ValueError(f"Unknown optimizer {name}.")
