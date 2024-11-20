from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from random import sample


logging.getLogger().setLevel(logging.INFO)
sns.set_style("darkgrid")


MCMC_DIR = Path("data/thinned_mcmc_runs")
SAMPLES_DIR = Path("data/distribution_data")
GRAPHS_DIR = Path("plots/posterior_ratio_plots")

NUM_PAIRS = 2_000_000


def _calculate_posterior_errors(
    analysis_name: str, sample_log_file: Path, reference_log_file: Path
):
    logging.info(f"Calculate pairwise posterior-ratio error for {analysis_name}...")

    sample_logs = pd.read_csv(sample_log_file)
    reference_logs = pd.read_csv(reference_log_file, delimiter="\t", comment="#")

    sample_log_posteriors = dict(
        zip(
            sample_logs.state,
            np.log(sample_logs.posterior / np.sum(sample_logs.posterior)),
        )
    )
    reference_log_posteriors = dict(
        zip(reference_logs.Sample.map(lambda x: f"STATE_{x}"), reference_logs.posterior)
    )

    states = list(reference_log_posteriors.keys())

    log_errors = []
    for _ in range(NUM_PAIRS):
        state_a, state_b = sample(states, 2)
        ref_log_posterior_diff = (
            reference_log_posteriors[state_a] - reference_log_posteriors[state_b]
        )
        sample_log_posterior_diff = (
            sample_log_posteriors[state_a] - sample_log_posteriors[state_b]
        )

        error = ref_log_posterior_diff - sample_log_posterior_diff
        log_errors.append(error)

    return np.ma.masked_invalid(log_errors)


def _plot_posterior_error(dataset_name: str, error_per_model: dict[str, list]):
    logging.info(f"Create pairwise posterior-ratio error plots for {dataset_name}...")

    mean_abs_error_per_model = {
        model_name: np.mean(np.abs(error))
        for model_name, error in error_per_model.items()
    }
    mean_abs_error_per_model = dict(
        sorted(mean_abs_error_per_model.items(), key=lambda x: x[1])
    )

    sns.barplot(mean_abs_error_per_model)

    plt.ylim([0, max(mean_abs_error_per_model.values()) + 0.1])

    plt.xlabel(f"Abs. Posterior Ratio Error ({dataset_name})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(GRAPHS_DIR / f"{dataset_name}_posterior-ratio-error.png", dpi=300)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    sample_tree_files = list(SAMPLES_DIR.glob("*.log"))

    dataset_indices: dict[str, list[int]] = defaultdict(list)

    for i, sample_tree_file in enumerate(sample_tree_files):
        file_name_wo_ext = sample_tree_file.name.removesuffix(".log")
        dataset_name, *_ = file_name_wo_ext.split("_")
        dataset_indices[dataset_name].append(i)

    logging.info(f"Start per-dataset comparison...")

    for dataset_name, indices in dataset_indices.items():
        error_per_model = defaultdict(list)

        for i in indices:
            sample_tree_file = sample_tree_files[i]

            file_name_wo_ext = sample_tree_file.name.removesuffix(".log")
            dataset_name, run, sample_type, model_name = file_name_wo_ext.split("_")

            sample_log_file = SAMPLES_DIR / f"{dataset_name}_{run}_logs_{model_name}.log"
            reference_log_file = MCMC_DIR / f"{dataset_name}_{run}.log"

            error_per_model[model_name] = _calculate_posterior_errors(
                file_name_wo_ext, sample_log_file, reference_log_file
            )

        _plot_posterior_error(dataset_name, error_per_model)
