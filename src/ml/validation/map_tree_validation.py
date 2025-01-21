from loguru import logger
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import yaml

from src.ml.preprocessing.add_relative_clade_information import (
    AddRelativeCladeInformation,
)
from src.ml.data.tree_dataset import TreeDataset
from src.ml.modeling import model_factory, optimizer_factory
from src.ml.modeling.weight_sharing_tree_flow import WeightSharingTreeFlow
from src.ml.preprocessing import preprocessing_factory

from src.datasets.load_trees import write_trees_to_file, load_trees_from_file

OUTPUT_DIR = Path("data/map_data")

MODEL_NAME = "nf-ws-fraction"
MODELS_PATH = Path("ml_data/models/tuned_weight_sharing_fraction_height_scaling_yule_10_2025_01_12_12_28_16")
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

    model = WeightSharingTreeFlow.load_from_checkpoint(
        next((MODELS_PATH / data_set_name).glob("*.ckpt"))
    )
    model = model.eval()

    return model


def map_tree_validation():
    config = _load_config()

    for map_tree_file in OUTPUT_DIR.glob("*_mrca.trees"):
        dataset, run, num_samples, _ = map_tree_file.stem.split("_")

        if int(num_samples) <= 1000:
            # this is not the full run
            continue

        if not list((MODELS_PATH / f"{dataset}_{run}").glob("*.ckpt")):
            continue

        logger.info(f"Start validation for {dataset}_{run}.")

        data_sets = _load_data(map_tree_file)
        data_loader, input_example = _preprocess_data(config, data_sets)

        model = _load_model(config, input_example, f"{dataset}_{run}")

        logger.info("Find convex hull.")

        mrca_batch = next(iter(data_loader))
        samples = [model.sample(mrca_batch) for _ in range(500)]

        sampled_branch_lengths = torch.cat([sample["branch_lengths"] for sample in samples]) 

        approximate_hull = []

        for dim in range(sampled_branch_lengths.shape[1]):
            sorted_by_dim = torch.argsort(sampled_branch_lengths[:, dim])
            approximate_hull.append(sampled_branch_lengths[sorted_by_dim[0]])
            approximate_hull.append(sampled_branch_lengths[sorted_by_dim[-1]])

        approximate_hull = torch.stack(approximate_hull)

        class Module(nn.Module):
            def __init__(self, model, approximate_hull, clades, tree_height, **kwargs):
                super().__init__()
                model.freeze()
                
                self.convex_factors = torch.nn.Parameter(torch.rand(len(approximate_hull)), requires_grad=True)
                
                self.approximate_hull = approximate_hull.T.detach()

                self.wrapped_model = model
                self.clades = clades
                self.tree_height = tree_height

            def loss(self):
                normalized_convex_factors = torch.softmax(self.convex_factors, 0)
                branch_lenghts = self.approximate_hull @ normalized_convex_factors

                batch = {
                    "branch_lengths": branch_lenghts.unsqueeze(0),
                    "clades": self.clades,
                    "tree_height": self.tree_height,
                }

                return -self.wrapped_model.get_log_likelihood(batch)

        logger.info("Run Adam to find map.")

        wrapped_model = Module(model, approximate_hull, **mrca_batch)
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=0.01)

        num_steps = 2000
        for _ in range(num_steps):
            loss = wrapped_model.loss()
            loss.backward()
            optim.step()
            optim.zero_grad()

        normalized_convex_factors = torch.softmax(wrapped_model.convex_factors, 0)
        map_branch_lenghts = wrapped_model.approximate_hull @ normalized_convex_factors

        logger.info("Found map.")

        mrca_batch["branch_lengths"] = map_branch_lenghts.unsqueeze(0)
        encoded_sample = model.encode(mrca_batch)
        tree_height = float(model.height_model.mode(**encoded_sample))

        tree = load_trees_from_file(map_tree_file)[0]

        AddRelativeCladeInformation.set_branch_lengths(
            tree,
            map_branch_lenghts.detach().numpy().tolist(),
            [int(x.detach()) for x in mrca_batch["clades"]],
            tree_height,
        )

        write_trees_to_file(
            [tree], OUTPUT_DIR / f"{dataset}_{run}_{num_samples}_{MODEL_NAME}.trees"
        )


if __name__ == "__main__":
    map_tree_validation()
