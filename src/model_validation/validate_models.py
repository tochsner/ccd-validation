from pathlib import Path
from Bio.Phylo.BaseTree import Tree
import numpy as np
from src.distribution_analysis.process_tree import (
    get_observed_nodes,
    get_clade_split_df,
)
from src.utils.tree_utils import get_tree_height, get_taxa_names
from src.datasets.load_trees import load_trees_from_file
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from random import sample


logging.getLogger().setLevel(logging.INFO)


SAMPLES_DIR = Path("data/validation")
REF_DIR = Path("data/beast")
GRAPHS_DIR = Path("data/validation_analysis")

NUM_PAIRS = 10_000
SAMPLE_SIZE = 10_000

def _create_height_distribution_plot(
    model_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    logging.info(f"Create height distribution plots for {model_name}...")

    ref_tree_heights = [
        get_tree_height(tree) for tree in sample(reference_trees, SAMPLE_SIZE)
    ]
    sample_tree_heights = [
        get_tree_height(tree) for tree in sample(sample_trees, SAMPLE_SIZE)
    ]

    sns.histplot(
        ref_tree_heights,
        stat="density",
        label="Reference",
    )
    sns.histplot(
        sample_tree_heights,
        stat="density",
        label="Sample",
    )

    plt.xlabel("Tree height")

    max_displayed_height = np.percentile(
        sample_tree_heights + sample_tree_heights, 99.9
    )
    plt.xlim(0, max_displayed_height)
    plt.legend(loc="upper right")

    plt.savefig(GRAPHS_DIR / f"{model_name}-height-distribution.png", dpi=300)
    plt.close()


def _create_branch_length_distribution_plot(
    model_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    logging.info(f"Create branch length distribution plots for {model_name}...")

    taxa_names = get_taxa_names(reference_trees[0])

    _, ref_clade_splits = get_observed_nodes(
        sample(reference_trees, SAMPLE_SIZE), taxa_names
    )
    _, sample_clade_splits = get_observed_nodes(
        sample(sample_trees, SAMPLE_SIZE), taxa_names
    )

    df_ref_clade_splits = get_clade_split_df(ref_clade_splits)
    df_sample_clade_splits = get_clade_split_df(sample_clade_splits)

    ref_branch_lengths = list(
        df_ref_clade_splits["left_branch"] + df_ref_clade_splits["right_branch"]
    )
    sample_branch_lengths = list(
        df_sample_clade_splits["left_branch"] + df_sample_clade_splits["right_branch"]
    )

    sns.histplot(
        ref_branch_lengths,
        stat="density",
        label="Reference",
    )
    sns.histplot(
        sample_branch_lengths,
        stat="density",
        label="Sample",
    )

    plt.xlabel("Branch length")

    max_displayed_length = np.percentile(
        ref_branch_lengths + sample_branch_lengths, 99.9
    )
    plt.xlim(0, max_displayed_length)
    plt.legend(loc="upper right")

    plt.savefig(GRAPHS_DIR / f"{model_name}-branch-length-distribution.png", dpi=300)
    plt.close()


def _create_posterior_error_plot(sample_log_file: Path, reference_log_file: Path):
    logging.info(f"Create pairwise posterior-ratio error plots for {model_name}...")

    sample_logs = pd.read_csv(sample_log_file)
    reference_logs = pd.read_csv(reference_log_file, delimiter="\t")

    sample_log_posteriors = dict(
        zip(
            sample_logs.state,
            np.log(sample_logs.posterior / np.sum(sample_logs.posterior)),
        )
    )
    reference_log_posteriors = dict(
        zip(reference_logs.Sample.map(lambda x: f"STATE_{x}"), reference_logs.posterior)
    )

    states = list(reference_log_posteriors.keys())

    log_errors = []
    for _ in range(NUM_PAIRS):
        state_a, state_b = sample(states, 2)
        ref_log_posterior_diff = (
            reference_log_posteriors[state_a] - reference_log_posteriors[state_b]
        )
        sample_log_posterior_diff = (
            sample_log_posteriors[state_a] - sample_log_posteriors[state_b]
        )

        error = ref_log_posterior_diff - sample_log_posterior_diff
        log_errors.append(error)

    sns.histplot(log_errors, stat="density")

    plt.xlabel("Pairwise posterior ratio error")

    plt.savefig(GRAPHS_DIR / f"{model_name}-approximation-error.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    for sample_tree_file in SAMPLES_DIR.glob("*.trees"):
        model_name = sample_tree_file.name.removesuffix(".trees")

        reference_tree_file = REF_DIR / (
            "-".join(model_name.split("-")[:-1]) + ".trees"
        )

        sample_log_file = SAMPLES_DIR / (model_name + ".log")
        reference_log_file = REF_DIR / ("-".join(model_name.split("-")[:-1]) + ".log")

        logging.info(f"Load trees for {model_name}...")

        sample_trees = load_trees_from_file(sample_tree_file)
        reference_trees = load_trees_from_file(reference_tree_file)

        logging.info(f"Start validating {model_name}...")

        _create_height_distribution_plot(model_name, reference_trees, sample_trees)
        _create_branch_length_distribution_plot(
            model_name, reference_trees, sample_trees
        )
        _create_posterior_error_plot(sample_log_file, reference_log_file)
