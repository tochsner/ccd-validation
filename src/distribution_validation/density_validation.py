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
sns.set_style("whitegrid")

POSTERIOR_DIR = Path("data/hpd_validation")
GRAPHS_DIR = Path("data/hpd_validation_analysis")
GRAPHS_DIR = Path("data/distribution_validation_analysis")


def _get_ecdf_error(samples):
    ecfd = stats.ecdf(samples).cdf

    error = 0

    for i in range(len(ecfd.quantiles) - 1):
        quantile_size = (ecfd.quantiles[i + 1] - ecfd.quantiles[i]) / 100
        error += quantile_size * abs(ecfd.quantiles[i] / 100 - ecfd.probabilities[i])

    return error


def plot_empirical_cdf():
    dict_true_tree_percentiles = {
        "dataset_name": [],
        "model_name": [],
        "true_tree_percentile": [],
    }

    for posterior_file in tqdm(list(POSTERIOR_DIR.glob("*.log"))):
        file_name_wo_ext = posterior_file.name.removesuffix(".log")
        run_name, model_name = file_name_wo_ext.split("_")
        dataset_name = "-".join(run_name.split("-")[:-1])

        posterior_df = pd.read_csv(posterior_file)

        assert posterior_df.iloc[0]["tree"] == "true"
        true_posterior = posterior_df.iloc[0]["posterior"]
        posterior_df = posterior_df.drop(posterior_df.index[0])

        dict_true_tree_percentiles["dataset_name"].append(dataset_name)
        dict_true_tree_percentiles["model_name"].append(model_name)
        dict_true_tree_percentiles["true_tree_percentile"].append(
            stats.percentileofscore(posterior_df.posterior, true_posterior)
        )

    df_true_tree_percentiles = pd.DataFrame(dict_true_tree_percentiles)

    for dataset in df_true_tree_percentiles.dataset_name.unique():
        fig = sns.ecdfplot(
            df_true_tree_percentiles[df_true_tree_percentiles.dataset_name == dataset],
            x="true_tree_percentile",
            hue="model_name",
        )

        plt.plot([0, 100], [0, 1], color="black", linestyle="solid")

        plt.title("Empirical CDF")
        plt.xlabel("True Tree Percentile")
        plt.ylabel("Empirical Cumulative Probability")
        fig.get_legend().set_title("")

        plt.savefig(GRAPHS_DIR / f"{dataset}_ecfd.png", dpi=300)
        plt.close()

        sns.barplot(
            df_true_tree_percentiles.groupby("model_name")
            .apply(lambda x: _get_ecdf_error(x.true_tree_percentile))
            .sort_values(),  # type: ignore
        )

        plt.title("Empirical CDF Error")
        plt.xlabel("Model")
        plt.ylabel("ECDF Error")
        plt.xticks(rotation=30, ha="right")

        plt.savefig(GRAPHS_DIR / f"{dataset}_ecfd-error.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    plot_empirical_cdf()
