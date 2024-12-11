from typing import Any

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from torch.utils.data import DataLoader, Dataset

from src.ml.data.splitting import create_data_splits
from src.ml.modeling import flow_factory, loss_factory, model_factory, optimizer_factory
from src.ml.utils.set_seed import set_seed

torch.set_default_device(torch.device("cpu"))


def _get_comet_api_key():
    with open("comet_api_key.yaml", "r") as f:
        return yaml.safe_load(f)["key"]


def train_neural_network(
    dataset: Dataset,
    comet_project_name: str,
    splitting_config: dict[str, Any],
    dataloader_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    flow_configs: list[dict[str, Any]],
    loss_config: dict[str, Any],
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
):
    """Trains a neural network."""
    set_seed()

    torch.set_default_dtype(torch.float32)

    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, **splitting_config
    )

    train_loader = DataLoader(train_dataset, **dataloader_config)
    test_loader = DataLoader(test_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    flows = []
    for flow_config in flow_configs:
        flow = flow_factory(**flow_config)
        flows.append(flow)

    loss = loss_factory(**loss_config)
    optimizer = optimizer_factory(**optimizer_config)
    model = model_factory(loss=loss, optimizer=optimizer, flows=flows, **model_config)

    logger = CometLogger(api_key=_get_comet_api_key(), project_name=comet_project_name)
    logger.log_hyperparams(
        {
            **splitting_config,
            **dataloader_config,
            **optimizer_config,
            **loss_config,
            **model_config,
            **trainer_config,
        }
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator="cpu",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"data/models/{comet_project_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
            ),
        ],
        **trainer_config,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)
