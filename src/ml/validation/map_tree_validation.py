from loguru import logger
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

from src.ml.data.tree_dataset import TreeDataset
from src.ml.modeling import model_factory, optimizer_factory
from src.ml.modeling.conditional_tree_flow import ConditionalTreeFlow
from src.ml.preprocessing import preprocessing_factory

from src.datasets.load_trees import write_trees_to_file

OUTPUT_DIR = Path("data/map_data")

MODEL_NAME = "nf-conditioned"
MODELS_PATH = Path("ml_data/models/yule_10_simple_2024_12_16_16_41_10")
CONFIG_PATH = Path("ml_data/output/config.yaml")


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_data(trees_file: Path):
    logger.info("Loading data.")
    data_sets = [TreeDataset(trees_file)]
    return data_sets


def _preprocess_data(config, data_sets):
    logger.info("Start preprocessing.")

    for preprocessing_step in config["preprocessing"]:
        logger.info("Perform {} preprocessing.", preprocessing_step["name"])

        transform = preprocessing_factory(**preprocessing_step)
        data_sets = [transform(data_set) for data_set in data_sets]

    data_loader = DataLoader(data_sets[0], shuffle=False)

    return data_loader, data_sets[0][0]


def _load_model(config, input_example, data_set_name):
    optimizer = optimizer_factory(**config["training"]["optimizer_config"])
    model = model_factory(
        optimizer=optimizer,
        input_example=input_example,
        **config["training"]["model_config"],
    )

    model = ConditionalTreeFlow.load_from_checkpoint(
        next((MODELS_PATH / data_set_name).glob("*.ckpt"))
    )
    model = model.eval()

    return model


def true_tree_density_validation():
    config = _load_config()

    for map_tree_file in OUTPUT_DIR.glob("*_10_mrca.trees"):
        dataset, run, num_samples, method = map_tree_file.stem.split("_")

        if int(num_samples) <= 1000:
            # this is not the full run
            continue

        if not list((MODELS_PATH / f"{dataset}_{run}").glob("*.ckpt")):
            continue

        logger.info(f"Start validation for {dataset}_{run}.")

        data_sets = _load_data(map_tree_file)
        data_loader, input_example = _preprocess_data(config, data_sets)

        model = _load_model(config, input_example, f"{dataset}_{run}")

        logger.info("Start validation.")

        first_and_only_batch = next(iter(data_loader))
        samples = [model.sample(first_and_only_batch) for _ in range(10)]
        samples = torch.cat(samples, dim=0)  # type: ignore
        mean_sample = torch.mean(samples, dim=0)

        # TODO apply branch lengths to the correct clades and store the tree again


if __name__ == "__main__":
    true_tree_density_validation()
