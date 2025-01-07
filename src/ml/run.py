from datetime import datetime
from loguru import logger
from pathlib import Path
from shutil import copy, copytree, rmtree

import numpy as np
import yaml

from src.ml.train_neural_network import train_neural_network
from src.ml.data import data_sets_factory
from src.ml.preprocessing import preprocessing_factory
from src.ml.utils.set_seed import set_seed


OUTPUT_PATH = Path("ml_data/output")
HISTORY_PATH = Path("ml_data/output_history")
CONFIG_FILE = Path("src/ml/config.yaml")


def run(config: dict):
    run_name = f"{config['run_name']}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    # prepare output directory

    rmtree(OUTPUT_PATH, ignore_errors=True)
    OUTPUT_PATH.mkdir(exist_ok=True)

    copy(CONFIG_FILE, OUTPUT_PATH)
    logger.add(OUTPUT_PATH / "logs.log")

    logger.info("Start run {}.", run_name)

    set_seed()

    # load data

    logger.info("Loading data.")
    data_sets = data_sets_factory(**config["data_set"])

    scores = []

    for data_set_name, data_set in data_sets:
        logger.info("Processing data set {}.", data_set_name)

        # preprocess data

        logger.info("Start preprocessing.")

        for preprocessing_step in config["preprocessing"]:
            logger.info("Perform {} preprocessing.", preprocessing_step["name"])

            transform = preprocessing_factory(**preprocessing_step)
            data_set = transform(data_set)

        # train models

        logger.info("Start training.")

        score = train_neural_network(
            dataset=data_set,
            run_name=f"{run_name}/{data_set_name}",
            **config["training"],
        )
        scores.append(score)

    # copy to history

    copytree(OUTPUT_PATH, HISTORY_PATH / run_name)

    return float(np.mean(scores))


if __name__ == "__main__":
    # load config file

    logger.info("Loading config file.")

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    run(config)
