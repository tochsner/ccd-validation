from pathlib import Path
from typing import Optional

from loguru import logger
from optuna import Trial, create_study
import yaml

from src.ml.run import run


CONFIG_FILE = Path("src/ml/optuna_config.yaml")


def _build_optuna_config(trial: Trial, config, name: Optional[str] = None):
    if isinstance(config, list):
        return [_build_optuna_config(trial, item) for item in config]

    if isinstance(config, dict) and "sampler" not in config:
        return {k: _build_optuna_config(trial, v, k) for k, v in config.items()}

    if name and isinstance(config, dict) and "sampler" in config:
        match config["sampler"]:
            case "uniform":
                return trial.suggest_float(name, config["min"], config["max"])
            case "log_uniform":
                return trial.suggest_loguniform(name, config["min"], config["max"])
            case "choice":
                return trial.suggest_categorical(name, config["options"])
            case "randint":
                return trial.suggest_int(name, config["min"], config["max"])
            case _:
                raise ValueError(f"Unknown sampler {config['sampler']}")

    return config


def hyperparameter_tuning(
    num_trials: int,
    **hyperparameter_config: dict,
):
    def run_training(trial: Trial) -> float:
        run_config: dict = _build_optuna_config(trial, hyperparameter_config)  # type: ignore
        run_config["run_name"] = f"{run_config['run_name']}_{trial.number}"
        score = run(run_config)
        return score

    study = create_study(direction="minimize")
    study.optimize(run_training, num_trials)


if __name__ == "__main__":
    logger.info("Loading config file.")

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Start hyperparameter tuning.")

    hyperparameter_tuning(**config)
