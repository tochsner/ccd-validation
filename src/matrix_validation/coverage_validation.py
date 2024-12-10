from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


COVERAGE_DIR = Path("data/matrix_coverage_data")
GRAPHS_DIR = Path("plots/matrix_coverage_plots")


def create_coverage_plots():
    df_dict = {
        "dataset": [],
        "run": [],
        "model": [],
        "coverage": [],
    }

    for file in COVERAGE_DIR.glob("*.log"):
        file_name_wo_ext = file.name.removesuffix(".log")
        dataset_name, run, model_name, _ = file_name_wo_ext.split("_")

        with open(file, "r") as handle:
            coverage = float(handle.readline().strip())

        df_dict["dataset"].append(dataset_name)
        df_dict["run"].append(run)
        df_dict["model"].append(model_name)
        df_dict["coverage"].append(coverage)

    df = pd.DataFrame(df_dict)

    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset]
        df_dataset.sort_values(by="coverage", ascending=False)

        sns.barplot(
            data=df_dataset,
            x="model",
            y="coverage",
        )
        plt.title(f"Coverage ({dataset})")
        plt.xlabel("Model")
        plt.ylabel("Coverage")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(GRAPHS_DIR / f"{dataset}_coverage.png", dpi=300)
        plt.clf()
        plt.close()


if __name__ == "__main__":
    create_coverage_plots()
