from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import mlflow
from torch.utils.data import DataLoader, Dataset

from src.ml.data.splitting import create_data_splits
from src.ml.modeling import (
    model_factory,
    optimizer_factory,
)
from src.ml.utils.set_seed import set_seed

torch.set_default_device(torch.device("cpu"))


def train_neural_network(
    dataset: Dataset,
    comet_project_name: str,
    splitting_config: dict[str, Any],
    dataloader_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
):
    """Trains a neural network."""
    set_seed()

    mlflow.pytorch.autolog()

    torch.set_default_dtype(torch.float32)

    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, **splitting_config
    )

    train_loader = DataLoader(train_dataset, **dataloader_config)
    test_loader = DataLoader(test_dataset, **dataloader_config)
    val_loader = DataLoader(val_dataset, **dataloader_config)

    optimizer = optimizer_factory(**optimizer_config)
    model = model_factory(
        optimizer=optimizer,
        dim=len(train_dataset[0]["branch_lengths"]),
        **model_config,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"ml_data/models/{comet_project_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
            ),
        ],
        **trainer_config,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)
