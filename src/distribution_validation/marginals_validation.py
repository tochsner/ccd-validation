from collections import defaultdict
import os
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
import numpy as np
import seaborn as sns
from random import sample
from multiprocessing import Pool


logging.getLogger().setLevel(logging.INFO)
sns.set_style("darkgrid")


MCMC_DIR = Path("data/thinned_mcmc_runs")
SAMPLES_DIR = Path("data/distribution_data")
GRAPHS_DIR = Path("plots/marginals_plots")

NUM_PAIRS = 5_000
SAMPLE_SIZE = 5_000
NUM_BINS = 25


def _load_sample_trees(sample_tree_file: Path):
    return load_trees_from_file(sample_tree_file, SAMPLE_SIZE)


def _load_ref_trees(sample_tree_file: Path):
    file_name_wo_ext = sample_tree_file.name.removesuffix(".trees")
    dataset_name, run, *_ = file_name_wo_ext.split("_")
    return load_trees_from_file(MCMC_DIR / f"{dataset_name}_{run}.trees", SAMPLE_SIZE)


def _get_bins(items: list[float]):
    min_item = np.percentile(items, 0.001)
    max_item = np.percentile(items, 99.999)
    return np.linspace(min_item, max_item, NUM_BINS)


def _create_height_distribution_plot(
    analysis_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    logging.info(f"Create height distribution plots for {analysis_name}...")

    ref_tree_heights = [
        get_tree_height(tree)
        for tree in sample(reference_trees, min(SAMPLE_SIZE, len(reference_trees)))
    ]
    sample_tree_heights = [
        get_tree_height(tree)
        for tree in sample(sample_trees, min(SAMPLE_SIZE, len(sample_trees)))
    ]

    bins = _get_bins(ref_tree_heights)

    sns.histplot(ref_tree_heights, stat="density", label="Reference", bins=bins)
    sns.histplot(sample_tree_heights, stat="density", label="Sample", bins=bins)

    plt.title(f"Tree Height Distribution ({analysis_name})")
    plt.xlabel("Tree height")
    plt.legend(loc="upper right")

    plt.savefig(GRAPHS_DIR / f"{analysis_name}_height-distribution.png", dpi=200)
    plt.clf()
    plt.close()


def _create_branch_length_distribution_plot(
    analysis_name: str, reference_trees: list[Tree], sample_trees: list[Tree]
):
    logging.info(f"Create branch length distribution plots for {analysis_name}...")

    taxa_names = get_taxa_names(reference_trees[0])

    _, ref_clade_splits = get_observed_nodes(
        sample(reference_trees, min(SAMPLE_SIZE, len(reference_trees))), taxa_names
    )
    _, sample_clade_splits = get_observed_nodes(
        sample(sample_trees, min(SAMPLE_SIZE, len(sample_trees))), taxa_names
    )

    df_ref_clade_splits = get_clade_split_df(ref_clade_splits).sample(SAMPLE_SIZE)
    df_sample_clade_splits = get_clade_split_df(sample_clade_splits).sample(SAMPLE_SIZE)

    ref_branch_lengths = list(
        df_ref_clade_splits["left_branch"] + df_ref_clade_splits["right_branch"]
    )
    sample_branch_lengths = list(
        df_sample_clade_splits["left_branch"] + df_sample_clade_splits["right_branch"]
    )

    bins = _get_bins(ref_branch_lengths)

    sns.histplot(ref_branch_lengths, stat="density", label="Reference", bins=bins)
    sns.histplot(sample_branch_lengths, stat="density", label="Sample", bins=bins)

    plt.title(f"Branch Length Distribution ({analysis_name})")
    plt.xlabel("Branch length")
    plt.legend(loc="upper right")

    plt.savefig(GRAPHS_DIR / f"{analysis_name}_branch-length-distribution.png", dpi=200)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    sample_tree_files = list(SAMPLES_DIR.glob("*.trees"))

    # sample_tree_files = [
    #     Path("data/distribution_data/yule-10_241_sampled-trees_gamma-mu-sigma-beta-old-old.trees"),
    #     Path("data/distribution_data/yule-10_241_sampled-trees_mu-sigma-beta-old-old.trees"),
    #     Path("data/distribution_data/yule-20_197_sampled-trees_mu-sigma-beta-old-old.trees"),
    #     Path("data/distribution_data/yule-20_197_sampled-trees_gamma-mu-sigma-beta-old-old.trees"), 
    # ]

    logging.info(f"Load {len(sample_tree_files)} trees...")

    with Pool(os.cpu_count()) as pool:
        trees_per_sample = pool.map(_load_sample_trees, sample_tree_files)
        ref_trees_per_sample = pool.map(_load_ref_trees, sample_tree_files)

    logging.info(f"Start per-sample validation...")

    for i, (sample_tree_file, sample_trees, reference_trees) in enumerate(
        zip(sample_tree_files, trees_per_sample, ref_trees_per_sample)
    ):
        file_name_wo_ext = sample_tree_file.name.removesuffix(".trees")
        *_, sample_type, model_name = file_name_wo_ext.split("_")

        logging.info(f"Start validating {model_name} ({sample_type})...")

        _create_height_distribution_plot(
            file_name_wo_ext, reference_trees, sample_trees
        )
        _create_branch_length_distribution_plot(
            file_name_wo_ext, reference_trees, sample_trees
        )
