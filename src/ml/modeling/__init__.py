from functools import partial
from typing import Callable, Iterator

import lightning.pytorch as pl
from torch import nn, optim

from src.ml.modeling.conditional_tree_flow import ConditionalTreeFlow


def model_factory(
    name: str,
    optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    **kwargs,
) -> pl.LightningModule:
    """Factory function for models."""
    match name:
        case "conditional_tree_flow":
            return ConditionalTreeFlow(optimizer=optimizer, **kwargs)
        case _:
            raise ValueError(f"Unknown model {name}.")


def optimizer_factory(
    name: str, **kwargs
) -> Callable[[Iterator[nn.Parameter]], optim.Optimizer]:
    """Factory function for optimizers."""
    match name:
        case "adam":
            return partial(optim.Adam, **kwargs)
        case _:
            raise ValueError(f"Unknown optimizer {name}.")
