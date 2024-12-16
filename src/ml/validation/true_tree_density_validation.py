from loguru import logger
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

import yaml

from src.ml.data.tree_dataset import TreeDataset
from src.ml.modeling import model_factory, optimizer_factory
from src.ml.modeling.conditional_tree_flow import ConditionalTreeFlow
from src.ml.preprocessing import preprocessing_factory

CCD1_SAMPLES_DIR = Path("data/ccd1_sample_data")
OUTPUT_DIR = Path("data/true_tree_density_data")

MODEL_NAME = "nf-conditioned"
MODELS_PATH = Path("ml_data/models/yule_10_simple_2024_12_16_16_41_10")
CONFIG_PATH = Path("ml_data/output/config.yaml")


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_ccd1_likelihood_data(tree_file: Path):
    logger.info("Loading CCD1 likelihood data.")

    likelihood_file = CCD1_SAMPLES_DIR / f"{tree_file.stem}.log"

    with open(likelihood_file, "r") as f:
        likelihoods = list(map(lambda s: float(s.split(",")[1]), f.readlines()[1:]))

    return likelihoods


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

    model = ConditionalTreeFlow.load_from_checkpoint(
        next((MODELS_PATH / data_set_name).glob("*.ckpt"))
    )
    model = model.eval()

    return model


def true_tree_density_validation():
    config = _load_config()

    for tree_file in CCD1_SAMPLES_DIR.glob("*.trees"):
        if not list((MODELS_PATH / tree_file.stem).glob("*.ckpt")):
            continue

        logger.info(f"Start validation for {tree_file}.")

        ccd1_likelihoods = _load_ccd1_likelihood_data(tree_file)
        data_sets = _load_data(tree_file)
        data_loader, input_example = _preprocess_data(config, data_sets)

        model = _load_model(config, input_example, data_loader, tree_file.stem)

        logger.info("Start validation.")

        log_likelihoods = []

        for tree_batch in iter(data_loader):
            sample_batch = model.sample(tree_batch)

            if len(log_likelihoods) == 0:
                # this is the first tree, i.e. the true tree
                sample_batch["branch_lengths"][0] = model.encode(tree_batch)["z"][0]

            model_log_likelihood_batch = (
                model.get_log_likelihood(sample_batch).detach().numpy()
            )

            for i, model_log_likelihood in enumerate(model_log_likelihood_batch):
                log_likelihood = (
                    ccd1_likelihoods[len(log_likelihoods)] + model_log_likelihood
                )
                log_likelihoods.append(log_likelihood)

        with open(OUTPUT_DIR / f"{tree_file.stem}_{MODEL_NAME}.log", "w") as f:
            f.write("tree,log_posterior\n")
            for i, log_likelihood in enumerate(log_likelihoods):
                f.write(f"{'true' if i == 0 else (i-1)},{log_likelihood}\n")


if __name__ == "__main__":
    true_tree_density_validation()
