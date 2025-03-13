from collections import defaultdict
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.map_validation.tree_scores import (
    heights_error,
    squared_rooted_branch_score,
)
from src.datasets.load_trees import load_trees_from_file

logging.getLogger().setLevel(logging.INFO)
sns.set_style("darkgrid")

TRUE_TREE_DIR = Path("data/mcmc_config")
MAP_TREE_DIR = Path("data/map_data")
GRAPHS_DIR = Path("plots/map_plots")

SCORES = {
    "Squared Rooted Branch Score": squared_rooted_branch_score,
    "Height Score": heights_error,
}
NUM_SCORES = len(SCORES)


def _create_scores_plot(df_scores: pd.DataFrame):
    logging.info(f"Create score plot...")

    for dataset, df in df_scores.groupby("dataset"):
        # plot data likelihoods on full dataset

        fig, axs = plt.subplots(ncols=NUM_SCORES, figsize=(NUM_SCORES * 4, 4))

        for i, score in enumerate(SCORES):
            sns.barplot(
                x="model",
                y=score,
                data=df[df.sample_size == "all"],
                ax=axs[i],
                errorbar=None,
                estimator="median",
            )

            axs[i].set_xlabel("Model")
            axs[i].set_xticks(
                axs[i].get_xticks(), axs[i].get_xticklabels(), rotation=30, ha="right"
            )

        fig.suptitle(
            f"Median Scores for Different MAP Estimators ({dataset}) on All Data ↓"
        )
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_full_scores.png", dpi=200)
        plt.clf()
        plt.close()

        # plot data likelihoods for different sample sizes

        fig, axs = plt.subplots(ncols=NUM_SCORES, figsize=(NUM_SCORES * 4, 4))

        for i, score in enumerate(SCORES):
            sns.lineplot(
                x="sample_size",
                hue="model",
                y=score,
                data=df,
                ax=axs[i],
                errorbar=None,
                estimator="median",
            )

            axs[i].set_xlabel("Sample Size")
            axs[i].set_xticks(
                axs[i].get_xticks(), axs[i].get_xticklabels(), rotation=30, ha="right"
            )

        fig.suptitle(f"Median Scores for Different MAP Estimators ({dataset}) ↓")
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_scores.png", dpi=200)
        plt.clf()
        plt.close()


def _create_wins_plot(df_scores: pd.DataFrame):
    df_scores = df_scores[
        df_scores["model"].isin(
            ["mrca", "sb-dirichlet", "sb-clade-beta", "dirichlet", "sb-beta-per-clade", "logit-multivariate-gaussian"]
        )
    ].replace(
        {
            "model": {
                "mrca": "MRCA",
                "sb-dirichlet": "Dirichlet",
                "sb-clade-beta": "Beta",
                "dirichlet": "Dirichlet",
                "sb-beta-per-clade": "Beta",
                "logit-multivariate-gaussian": "Logit"
            }
        }
    )

    logging.info(f"Create wins plot...")

    for dataset, df in df_scores.groupby("dataset"):
        num_sample_sizes = df["sample_size"].nunique()

        # plot wins on full dataset

        fig, axs = plt.subplots(
            ncols=NUM_SCORES,
            nrows=1,
            figsize=(4 * NUM_SCORES, 4),
        )

        for i, score in enumerate(SCORES):
            df_score_wins = (
                df[df["sample_size"] == "all"]
                .sort_values(score)
                .drop_duplicates(["run"])
                .sort_values("model")
            )

            sns.countplot(data=df_score_wins, x="model", ax=axs[i])

            axs[i].set_xlabel("Model")
            axs[i].set_ylabel(f"Number of Wins ({score})")
            axs[i].set_xticks(
                axs[i].get_xticks(),
                axs[i].get_xticklabels(),
                rotation=30,
                ha="right",
            )

        fig.suptitle(
            f"Number of Wins for Different MAP Estimators ({dataset}) ↑", y=0.99
        )
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_full_wins.png", dpi=200)
        plt.clf()
        plt.close()

        # plot wins on subsets

        fig, axs = plt.subplots(
            ncols=NUM_SCORES,
            nrows=num_sample_sizes,
            figsize=(4 * NUM_SCORES, 4 * num_sample_sizes),
        )

        for i, score in enumerate(SCORES):
            for j, sample_size in enumerate(df["sample_size"].unique()):
                df_score_wins = (
                    df[df["sample_size"] == sample_size]
                    .sort_values(score)
                    .drop_duplicates(["run"])
                    .sort_values("model")
                )

                sns.countplot(data=df_score_wins, x="model", ax=axs[j, i])

                axs[j, i].set_xlabel(f"Model trained on {sample_size} trees")
                axs[j, i].set_ylabel(f"Number of Wins ({score})")
                axs[j, i].set_xticks(
                    axs[j, i].get_xticks(),
                    axs[j, i].get_xticklabels(),
                    rotation=30,
                    ha="right",
                )

        fig.suptitle(
            f"Number of Wins for Different MAP Estimators ({dataset}) ↑", y=0.99
        )
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_wins.png", dpi=200)
        plt.clf()
        plt.close()


def map_validation():
    logging.info(f"Load trees...")

    map_trees_per_dataset = defaultdict(list)

    for map_tree in MAP_TREE_DIR.glob("*.trees"):
        file_name_wo_ext = map_tree.name.removesuffix(".trees")
        dataset_name, run, sample_size, model_name = file_name_wo_ext.split("_")

        if int(sample_size) > 1000:
            sample_size = "all"

        map_trees_per_dataset[(dataset_name, run, sample_size)].append(map_tree)

    logging.info(f"Calculate scores...")

    scores_dict = {
        "model": [],
        "dataset": [],
        "run": [],
        "sample_size": [],
        **{score: [] for score in SCORES.keys()},
    }

    for (dataset, run, sample_size), map_files in tqdm(
        list(map_trees_per_dataset.items())
    ):
        reference_tree = load_trees_from_file(TRUE_TREE_DIR / f"{dataset}_{run}.trees")[
            0
        ]

        for map_file in map_files:
            file_name_wo_ext = map_file.name.removesuffix(".trees")
            *_, model_name = file_name_wo_ext.split("_")

            map_tree = load_trees_from_file(map_file)[0]

            scores_dict["model"].append(model_name)
            scores_dict["dataset"].append(dataset)
            scores_dict["sample_size"].append(sample_size)
            scores_dict["run"].append(run)

            for score, score_func in SCORES.items():
                scores_dict[score].append(score_func(map_tree, reference_tree))

    df_scores = pd.DataFrame(scores_dict)

    df_scores = df_scores.sort_values("sample_size")

    _create_scores_plot(df_scores)
    _create_wins_plot(df_scores)


if __name__ == "__main__":
    map_validation()
