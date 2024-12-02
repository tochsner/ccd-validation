from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("darkgrid")


POSTERIOR_DIR = Path("data/likelihood_data")
PLOTS_DIR = Path("plots/num_parameters_plots")


def generate_num_parameters_plots():
    dict_num_parameters = {
        "dataset_name": [],
        "model_name": [],
        "num_parameters": [],
    }

    for posterior_file in list(POSTERIOR_DIR.glob("*.log")):
        file_name_wo_ext = posterior_file.name.removesuffix(".log")
        dataset_name, _, sample_size, _, model_name = file_name_wo_ext.split("_")

        if int(sample_size) <= 1000:
            # this is not the full dataset
            continue

        likelihoods = pd.read_csv(posterior_file)
        valid_likelihoods = likelihoods[likelihoods.log_posterior != -np.inf]

        dict_num_parameters["dataset_name"].append(dataset_name)
        dict_num_parameters["model_name"].append(model_name)
        dict_num_parameters["num_parameters"].append(
            valid_likelihoods.num_parameters.mean()
        )

    df_num_parameters = pd.DataFrame(dict_num_parameters)
    df_num_parameters = df_num_parameters.sort_values("num_parameters")

    for dataset_name, df_likelihoods_per_dataset in df_num_parameters.groupby(
        "dataset_name"
    ):
        # plot number of paramters

        sns.barplot(
            df_likelihoods_per_dataset,
            x="model_name",
            y="num_parameters",
        )

        plt.title(f"Number of Parameters ({dataset_name})")
        plt.xlabel("Model")
        plt.ylabel("Number of Parameters")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(PLOTS_DIR / f"{dataset_name}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    generate_num_parameters_plots()
