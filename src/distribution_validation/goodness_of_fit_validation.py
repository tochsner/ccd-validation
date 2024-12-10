import warnings
from scipy.stats import ks_2samp
import logging
from multiprocessing import Pool
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.distribution_analysis.process_tree import (
    get_clade_split_df,
    get_observed_nodes,
)
from src.datasets.load_trees import load_trees_from_file
from src.utils.tree_utils import get_taxa_names

sns.set_style("darkgrid")
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")

MCMC_DIR = Path("data/thinned_mcmc_runs")
SAMPLES_DIR = Path("data/distribution_data")
PLOTS_DIR = Path("plots/goodness_of_fit_plots")


def _load_sample_trees(sample_tree_file: Path):
    return load_trees_from_file(sample_tree_file)


loaded_ref_trees = {}


def _load_ref_trees(sample_tree_file: Path):
    file_name_wo_ext = sample_tree_file.name.removesuffix(".trees")
    dataset_name, run, *_ = file_name_wo_ext.split("_")
    ref_tree_file = MCMC_DIR / f"{dataset_name}_{run}.trees"

    if ref_tree_file not in loaded_ref_trees:
        loaded_ref_trees[ref_tree_file] = load_trees_from_file(ref_tree_file)

    return loaded_ref_trees[ref_tree_file]


if __name__ == "__main__":
    sample_tree_files = list(SAMPLES_DIR.glob("*.trees"))

    logging.info(f"Start per-sample validation...")

    dfs = []

    for i, sample_tree_file in tqdm(list(enumerate(sample_tree_files))):
        sample_trees = _load_sample_trees(sample_tree_file)
        reference_trees = _load_ref_trees(sample_tree_file)

        file_name_wo_ext = sample_tree_file.name.removesuffix(".trees")
        data_set, run, sample_type, model_name = file_name_wo_ext.split("_")

        taxa_names = get_taxa_names(sample_trees[0])

        _, sample_clade_splits = get_observed_nodes(sample_trees, taxa_names)
        sample_branches = get_clade_split_df(sample_clade_splits)

        _, ref_clade_splits = get_observed_nodes(reference_trees, taxa_names)
        ref_branches = get_clade_split_df(ref_clade_splits)

        df_gof = ref_branches.groupby("clade_split").apply(
            lambda r: ks_2samp(
                sample_branches[
                    sample_branches.clade_split == r.clade_split.iloc[0]
                ].min_branch,
                r.min_branch,
            ).statistic  # type: ignore
        )

        df_gof = df_gof.to_frame("gof").rename_axis("clade_split").reset_index()
        df_gof["run"] = run
        df_gof["model_name"] = model_name
        df_gof["data_set"] = data_set

        dfs.append(df_gof)

    logging.info(f"Create plots...")

    df = pd.concat(dfs)

    for data_set in df.data_set.unique():
        df_wins = (
            df[df.data_set == data_set]
            .sort_values("gof")
            .drop_duplicates(["clade_split", "run"])
            .sort_values("model_name")
        )

        sns.countplot(df_wins, x="model_name")

        plt.title(f"GoF Wins per Clade Split ({data_set}) ↑")
        plt.ylabel("Number of Clade Splits")
        plt.xlabel("Model")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"wins_{data_set}.png", dpi=300)
        plt.close()
