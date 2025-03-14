from scipy.stats import binom
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import stats
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats

import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

POSTERIOR_DIR = Path("data/true_tree_density_data")
GRAPHS_DIR = Path("plots/true_tree_density_plots")


def _get_ecdf_error(samples):
    ecfd = stats.ecdf(samples.fillna(0)).cdf

    error = 0

    for i in range(len(ecfd.quantiles) - 1):
        quantile_size = (ecfd.quantiles[i + 1] - ecfd.quantiles[i]) / 100
        error += quantile_size * abs(ecfd.quantiles[i] / 100 - ecfd.probabilities[i])

    quantile_size = (100.0 - ecfd.quantiles[-1]) / 100
    error += quantile_size * abs(ecfd.quantiles[-1] / 100 - ecfd.probabilities[-1])

    return error


def generate_empirical_cdf_plots():
    dict_true_tree_percentiles = {
        "dataset_name": [],
        "model_name": [],
        "true_tree_percentile": [],
    }

    for posterior_file in tqdm(list(POSTERIOR_DIR.glob("*.log"))):
        file_name_wo_ext = posterior_file.name.removesuffix(".log")
        dataset_name, _, model_name = file_name_wo_ext.split("_")

        posterior_df = pd.read_csv(posterior_file)

        assert posterior_df.iloc[0]["tree"] == "true"
        true_posterior = posterior_df.iloc[0]["log_posterior"]
        posterior_df = posterior_df.drop(posterior_df.index[0]).dropna()

        dict_true_tree_percentiles["dataset_name"].append(dataset_name)
        dict_true_tree_percentiles["model_name"].append(model_name)
        dict_true_tree_percentiles["true_tree_percentile"].append(
            stats.percentileofscore(posterior_df.log_posterior, true_posterior)
        )

    df_true_tree_percentiles = pd.DataFrame(dict_true_tree_percentiles)
    df_true_tree_percentiles["rounded_true_tree_percentile"] = (
        df_true_tree_percentiles.true_tree_percentile // 10 * 10
    )
    df_true_tree_percentiles = df_true_tree_percentiles.dropna()

    for dataset, df_dataset in df_true_tree_percentiles.groupby("dataset_name"):
        # df_dataset = df_dataset[
        #     df_dataset.model_name.isin(["dirichlet", "sb-beta-per-clade"])
        # ].replace(
        #     {
        #         "model_name": {
        #             "dirichlet": "Dirichlet",
        #             "sb-beta-per-clade": "Beta",
        #         }
        #     }
        # )

        # plot histogram of true tree percentiles
        df_percentile_counts = (
            df_dataset.drop(columns="true_tree_percentile", inplace=False)
            .groupby("model_name")
            .value_counts()
            .reset_index()
        )

        fig = sns.barplot(
            df_percentile_counts,
            x="rounded_true_tree_percentile",
            y="count",
            hue="model_name",
            errorbar=None,
        )

        plt.title(f"Number of Runs per True Tree Percentile ({dataset})")
        plt.xlabel("True Tree Percentile")
        plt.ylabel("Number of Runs")
        fig.get_legend().set_title("")

        runs_per_model = 100
        expected_counts = runs_per_model / 10
        lower_bound, upper_bound = binom.interval(0.95, runs_per_model, 1 / 10)

        plt.axhline(expected_counts, color="gray", linestyle="solid")
        plt.axhspan(lower_bound, upper_bound, color="gray", alpha=0.2)

        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / f"{dataset}_histogram.png", dpi=300)
        plt.close()

        # plot empirical cumulative distribution function

        fig = sns.ecdfplot(
            df_dataset, x="true_tree_percentile", hue="model_name"
        )

        plt.plot([0, 100], [0, 1], color="black", linestyle="solid")

        for r in range(100):
            lower_bound, upper_bound = binom.interval(0.95, runs_per_model, r / 100)
            rect = plt.Rectangle(  # type: ignore
                (r, lower_bound / 100),
                1,
                upper_bound / 100 - lower_bound / 100,
                facecolor="gray",
                alpha=0.2,
            )
            fig.add_patch(rect)

        plt.title(f"Empirical CDF ({dataset})")
        plt.xlabel("True Tree Percentile")
        plt.ylabel("Empirical Cumulative Probability")
        fig.get_legend().set_title("")

        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / f"{dataset}_ecfd.png", dpi=300)
        plt.close()

        # plot errors per model

        sns.barplot(
            df_dataset.groupby("model_name")
            .apply(lambda x: _get_ecdf_error(x.true_tree_percentile))
            .sort_values(),  # type: ignore
        )

        plt.title(f"Empirical CDF Error ({dataset}) ↓")
        plt.xlabel("Model")
        plt.ylabel("ECDF Error")
        plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / f"{dataset}_ecfd-error.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    generate_empirical_cdf_plots()
