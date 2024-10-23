from pathlib import Path
from Bio.Phylo.BaseTree import Tree
import numpy as np
from src.distribution_analysis.process_tree import get_observed_nodes, get_clade_split_df
from src.utils.tree_utils import get_tree_height, get_taxa_names
from src.datasets.load_trees import load_trees_from_file
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logging.getLogger().setLevel(logging.INFO)


SAMPLES_DIR = Path("data/sampled_bccd")
REF_DIR = Path("data/beast")
GRAPHS_DIR = Path("data/validation")


def _create_height_distribution_plots(
    model_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    logging.info(f"Create height distribution plots for {model_name}...")

    ref_tree_heights = [get_tree_height(tree) for tree in reference_trees]
    sample_tree_heights = [get_tree_height(tree) for tree in sample_trees]

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

    max_displayed_height = np.percentile(sample_tree_heights + sample_tree_heights, 99)
    plt.xlim(0, max_displayed_height)
    plt.legend(loc="upper right")

    plt.savefig(GRAPHS_DIR / f"{model_name}-height-distribution.png", dpi=300)
    plt.close()


def _create_branch_length_distribution_plots(
    model_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    logging.info(f"Create branch length distribution plots for {model_name}...")

    taxa_names = get_taxa_names(reference_trees[0])

    _, ref_clade_splits = get_observed_nodes(reference_trees, taxa_names)
    _, sample_clade_splits = get_observed_nodes(sample_trees, taxa_names)

    df_ref_clade_splits = get_clade_split_df(ref_clade_splits)
    df_sample_clade_splits = get_clade_split_df(sample_clade_splits)

    ref_branch_lengths = list(df_ref_clade_splits["left_branch"] + df_ref_clade_splits["right_branch"])
    sample_branch_lengths = list(df_sample_clade_splits["left_branch"] + df_sample_clade_splits["right_branch"])

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

    max_displayed_length = np.percentile(ref_branch_lengths + sample_branch_lengths, 99)
    plt.xlim(0, max_displayed_length)
    plt.legend(loc="upper right")

    plt.savefig(GRAPHS_DIR / f"{model_name}-branch-length-distribution.png", dpi=300)
    plt.close()


def create_basic_distribution_plots(
    model_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    _create_height_distribution_plots(model_name, reference_trees, sample_trees)
    _create_branch_length_distribution_plots(model_name, reference_trees, sample_trees)


if __name__ == "__main__":
    for sample_tree_file in SAMPLES_DIR.glob("*.trees"):
        model_name = sample_tree_file.name.removesuffix(".trees")

        reference_tree_file = REF_DIR / sample_tree_file.name

        logging.info(f"Load trees for {model_name}...")

        sample_trees = load_trees_from_file(sample_tree_file)
        reference_trees = load_trees_from_file(reference_tree_file)

        logging.info(f"Start validating {model_name}...")

        create_basic_distribution_plots(model_name, reference_trees, sample_trees)
