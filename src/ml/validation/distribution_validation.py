from loguru import logger
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

import yaml

from src.ml.preprocessing.add_relative_clade_information import AddRelativeCladeInformation
from src.datasets.load_trees import load_trees_from_file, write_trees_to_file
from src.ml.data.tree_dataset import TreeDataset
from src.ml.modeling import model_factory, optimizer_factory
from src.ml.modeling.conditional_tree_flow import ConditionalTreeFlow
from src.ml.modeling.weight_sharing_tree_flow import WeightSharingTreeFlow
from src.ml.preprocessing import preprocessing_factory

SAMPLES_DIR = Path("data/distribution_data")
OUTPUT_DIR = Path("data/distribution_data")

MODEL_NAME = "nf-ws-fraction"
MODELS_PATH = Path("ml_data/models/debug_2025_01_12_12_07_06")
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

    data_loader = DataLoader(data_sets[0], batch_size=16, shuffle=False)

    return data_loader, data_sets[0][0]


def _load_model(config, input_example, data_loader, data_set_name):
    optimizer = optimizer_factory(**config["training"]["optimizer_config"])
    model = model_factory(
        optimizer=optimizer,
        input_example=input_example,
        **config["training"]["model_config"],
    )

    model = WeightSharingTreeFlow.load_from_checkpoint(
        next((MODELS_PATH / data_set_name).glob("*.ckpt"))
    )
    model = model.eval()

    return model


def distribution_validation():
    config = _load_config()

    for tree_file in SAMPLES_DIR.glob("*dirichlet.trees"):
        dataset_name, run, *_ = tree_file.stem.split("_")

        dataset_name = f"{dataset_name}_{run}"

        if not list((MODELS_PATH / dataset_name).glob("*.ckpt")):
            continue

        logger.info(f"Start validation for {tree_file}.")

        data_sets = _load_data(tree_file)
        data_loader, input_example = _preprocess_data(config, data_sets)

        trees = load_trees_from_file(tree_file)

        model = _load_model(config, input_example, data_loader, dataset_name)

        logger.info("Start validation.")

        sampled_trees = []

        for tree_batch in iter(data_loader):
            sample_batch = model.sample(tree_batch)

            for i, tidx in enumerate(sample_batch["tree_index"]):
                AddRelativeCladeInformation.set_branch_lengths(
                    trees[tidx],
                    [float(x.detach()) for x in sample_batch["branch_lengths"][i]],
                    [int(x[i].detach()) for x in sample_batch["clades"]],
                    float(sample_batch["tree_height"][i].detach()),
                )
                sampled_trees.append(trees[tidx])

        write_trees_to_file(
            sampled_trees, OUTPUT_DIR / f"{dataset_name}_sampled-trees_{MODEL_NAME}.trees"
        )

if __name__ == "__main__":
    distribution_validation()
