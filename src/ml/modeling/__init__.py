from functools import partial
from typing import Callable, Iterator

import lightning.pytorch as pl
from torch import Tensor, nn, optim

from src.ml.modeling.planar_flow_layer import PlanarFlowLayer
from src.ml.modeling.conditional_tree_flow import ConditionalTreeFlow


def model_factory(
    name: str,
    loss: Callable[[Tensor, Tensor], Tensor],
    optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    flows: list,
    **kwargs,
) -> pl.LightningModule:
    """Factory function for models."""
    match name:
        case "conditional_tree_flow":
            return ConditionalTreeFlow(loss, optimizer, flows, **kwargs)
        case _:
            raise ValueError(f"Unknown model {name}.")


def flow_layer_factory(
    name: str,
    **kwargs,
) -> nn.Module:
    """Factory function for flow modules."""
    match name:
        case "planar_flow_layer":
            return PlanarFlowLayer(**kwargs)
        case _:
            raise ValueError(f"Unknown flow {name}.")


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
