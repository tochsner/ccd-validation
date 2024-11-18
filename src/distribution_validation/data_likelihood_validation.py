from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from random import sample


logging.getLogger().setLevel(logging.INFO)
sns.set_style("darkgrid")


SAMPLES_DIR = Path("data/distribution_validation")
REF_DIR = Path("data/mcmc_runs")
GRAPHS_DIR = Path("data/distribution_validation_analysis")


def _calculate_data_likelihood(analysis_name: str, log_file: Path):
    logs = pd.read_csv(log_file)
    return np.sum(np.log(logs.posterior / np.sum(logs.posterior)))


def _plot_data_likelihood(dataset_name: str, likelihood_per_model: dict[str, list]):
    sns.barplot(likelihood_per_model)

    plt.xlabel(f"Data likelihood ({dataset_name})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(GRAPHS_DIR / f"{dataset_name}_data-likelihood.png", dpi=300)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    sample_tree_files = list(SAMPLES_DIR.glob("*.trees"))

    dataset_indices: dict[str, list[int]] = defaultdict(list)

    for i, sample_tree_file in enumerate(sample_tree_files):
        file_name_wo_ext = sample_tree_file.name.removesuffix(".trees")
        dataset_name, *_ = file_name_wo_ext.split("_")
        dataset_indices[dataset_name].append(i)

    logging.info(f"Start per-dataset comparison...")

    for dataset_name, indices in dataset_indices.items():
        likelihood_per_model = defaultdict(list)

        for i in indices:
            sample_tree_file = sample_tree_files[i]

            file_name_wo_ext = sample_tree_file.name.removesuffix(".trees")
            dataset_name, sample_type, model_name = file_name_wo_ext.split("_")

            log_file = SAMPLES_DIR / f"{dataset_name}_logs_{model_name}.log"

            likelihood_per_model[model_name] = _calculate_data_likelihood(
                file_name_wo_ext, log_file
            )

        _plot_data_likelihood(dataset_name, likelihood_per_model)
