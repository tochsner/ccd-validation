from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("darkgrid")


POSTERIOR_DIR = Path("data/likelihood_data")
PLOTS_DIR = Path("plots/likelihood_plots")


def generate_data_likelihood_plots():
    dict_likelihoods = {
        "dataset_name": [],
        "run": [],
        "sample_size": [],
        "model_name": [],
        "data_likelihood": [],
        "zero_likelihood_fraction": [],
    }

    for posterior_file in list(POSTERIOR_DIR.glob("*.log")):
        file_name_wo_ext = posterior_file.name.removesuffix(".log")
        dataset_name, run, sample_size, _, model_name = file_name_wo_ext.split("_")

        if int(sample_size) > 10000:
            sample_size = "full"

        likelihoods = pd.read_csv(posterior_file)

        dict_likelihoods["dataset_name"].append(dataset_name)
        dict_likelihoods["run"].append(run)
        dict_likelihoods["sample_size"].append(sample_size)
        dict_likelihoods["model_name"].append(model_name)
        dict_likelihoods["data_likelihood"].append(
            np.sum(np.log(likelihoods.posterior[likelihoods.posterior > 0]))
        )
        dict_likelihoods["zero_likelihood_fraction"].append(
            np.mean(likelihoods.posterior == 0)
        )

    df_likelihoods = pd.DataFrame(dict_likelihoods)

    for dataset_name, df_likelihoods_per_dataset in df_likelihoods.groupby(
        "dataset_name"
    ):
        df_likelihoods_per_dataset = df_likelihoods_per_dataset.sort_values(
            "data_likelihood"
        )

        # plot data likelihoods

        sns.lineplot(
            df_likelihoods_per_dataset,
            x="sample_size",
            y="data_likelihood",
            hue="model_name",
            ci=None,
        )

        plt.title(f"Data Likelihood ({dataset_name})")
        plt.xlabel("Model")
        plt.ylabel("Data Likelihood")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(PLOTS_DIR / f"{dataset_name}_data-likelihood.png", dpi=300)
        plt.close()

        # plot zero likelihood fraction

        sns.lineplot(
            df_likelihoods_per_dataset,
            x="sample_size",
            y="zero_likelihood_fraction",
            hue="model_name",
            errorbar=None,
        )

        plt.title(f"Fraction of Trees with Zero Likelihood ({dataset_name})")
        plt.xlabel("Model")
        plt.ylabel("Fraction")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(PLOTS_DIR / f"{dataset_name}_zero-likelihood-fraction.png", dpi=300)
        plt.close()

        # plot wins

        df_wins = (
            df_likelihoods_per_dataset.sort_values("data_likelihood", ascending=False)
            .drop_duplicates(["run", "sample_size"])
            .sort_values("sample_size")
        )

        grid = sns.FacetGrid(df_wins, col="sample_size")
        grid.map_dataframe(sns.countplot, x="model_name")

        grid.figure.subplots_adjust(top=0.9)
        grid.figure.suptitle(f"Data Likelihood Wins ({dataset_name})")

        grid.set_xlabels("Model")
        grid.set_ylabels("Number of Runs")

        grid.set_xticklabels(rotation=30, ha="right")
        grid.tight_layout()

        plt.savefig(PLOTS_DIR / f"{dataset_name}_wins.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    generate_data_likelihood_plots()
