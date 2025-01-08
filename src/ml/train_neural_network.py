from typing import Any, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
from torch.utils.data import DataLoader, Dataset

from src.ml.data.splitting import create_data_splits
from src.ml.modeling import (
    model_factory,
    optimizer_factory,
)
from src.ml.utils.set_seed import set_seed

from torch.utils.data._utils.collate import default_collate_fn_map


def default_collate_fn(batch, **kwargs):
    return torch.tensor(batch, dtype=torch.float32)


default_collate_fn_map[float] = default_collate_fn

torch.set_default_device(torch.device("cpu"))
torch.set_default_dtype(torch.float32)


def train_neural_network(
    dataset: Dataset,
    run_name: str,
    splitting_config: dict[str, Any],
    dataloader_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
    mlflow_experiment_name: Optional[str] = None,
):
    """Trains a neural network."""
    set_seed()

    mlflow.pytorch.autolog()
    if mlflow_experiment_name:
        mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_params(
        {
            **splitting_config,
            **dataloader_config,
            **optimizer_config,
            "optimizer_name": optimizer_config["name"],
            **model_config,
            "model_name": model_config["name"],
            **trainer_config,
        }
    )

    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, **splitting_config
    )

    train_loader = DataLoader(
        train_dataset, generator=torch.Generator("cpu"), **dataloader_config
    )
    test_loader = DataLoader(
        test_dataset, generator=torch.Generator("cpu"), **dataloader_config
    )
    val_loader = DataLoader(
        val_dataset, generator=torch.Generator("cpu"), **dataloader_config
    )

    optimizer = optimizer_factory(**optimizer_config)
    model = model_factory(
        optimizer=optimizer,
        input_example=train_dataset[0],
        **model_config,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"ml_data/models/{run_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_loss", patience=4),
        ],
        **trainer_config,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(model=model, dataloaders=test_loader)

    mlflow.end_run()

    return test_result[0]["test_loss"]
