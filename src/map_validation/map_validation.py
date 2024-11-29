from collections import defaultdict
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.map_validation.tree_scores import (
    heights_error,
    rooted_branch_score,
    squared_rooted_branch_score,
)
from src.datasets.load_trees import load_trees_from_file

logging.getLogger().setLevel(logging.INFO)
sns.set_style("darkgrid")

TRUE_TREE_DIR = Path("data/mcmc_config")
MAP_TREE_DIR = Path("data/map_data")
GRAPHS_DIR = Path("plots/map_plots")

SCORES = {
    "Rooted Branch Score": rooted_branch_score,
    "Squared RBS": squared_rooted_branch_score,
    "Height Score": heights_error,
}
NUM_SCORES = len(SCORES)


def _create_scores_plot(df_scores: pd.DataFrame):
    logging.info(f"Create score plot...")

    for dataset, df in df_scores.groupby("dataset"):
        fig, axs = plt.subplots(ncols=NUM_SCORES, figsize=(NUM_SCORES * 4, 5))

        for i, score in enumerate(SCORES):
            sns.boxplot(x="model", y=score, data=df, ax=axs[i])

            axs[i].set_xlabel("Model")
            axs[i].set_xticks(
                axs[i].get_xticks(), axs[i].get_xticklabels(), rotation=30, ha="right"
            )

        fig.suptitle("Scores for Different MAP Estimators")
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_scores.png", dpi=200)
        plt.clf()
        plt.close()


def _create_wins_plot(df_scores: pd.DataFrame):
    logging.info(f"Create wins plot...")

    for dataset, df in df_scores.groupby("dataset"):
        fig, axs = plt.subplots(ncols=NUM_SCORES, figsize=(4 * NUM_SCORES, 4))

        for i, score in enumerate(SCORES):
            df_score_wins = df.sort_values(score).drop_duplicates(["run"])

            sns.countplot(data=df_score_wins, x="model", ax=axs[i])

            axs[i].set_xlabel("Model")
            axs[i].set_ylabel(f"Number of Wins ({score})")
            axs[i].set_xticks(
                axs[i].get_xticks(), axs[i].get_xticklabels(), rotation=30, ha="right"
            )

        fig.suptitle("Number of Wins for Different MAP Estimators")
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_wins.png", dpi=200)
        plt.clf()
        plt.close()


def map_validation():
    logging.info(f"Load trees...")

    map_trees_per_dataset = defaultdict(list)

    for map_tree in MAP_TREE_DIR.glob("*.trees"):
        file_name_wo_ext = map_tree.name.removesuffix(".trees")
        dataset_name, run, model_name = file_name_wo_ext.split("_")
        map_trees_per_dataset[(dataset_name, run)].append(map_tree)

    logging.info(f"Calculate scores...")

    scores_dict = {
        "model": [],
        "dataset": [],
        "run": [],
        **{score: [] for score in SCORES.keys()},
    }

    for (dataset, run), map_files in map_trees_per_dataset.items():
        reference_tree = load_trees_from_file(TRUE_TREE_DIR / f"{dataset}_{run}.trees")[
            0
        ]

        for map_file in map_files:
            file_name_wo_ext = map_file.name.removesuffix(".trees")
            *_, model_name = file_name_wo_ext.split("_")

            map_tree = load_trees_from_file(map_file)[0]

            scores_dict["model"].append(model_name)
            scores_dict["dataset"].append(dataset)
            scores_dict["run"].append(run)

            for score, score_func in SCORES.items():
                scores_dict[score].append(score_func(map_tree, reference_tree))

    df_scores = pd.DataFrame(scores_dict)

    _create_scores_plot(df_scores)
    _create_wins_plot(df_scores)


if __name__ == "__main__":
    map_validation()
