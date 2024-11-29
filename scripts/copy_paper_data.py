from pathlib import Path
import shutil
from tqdm import tqdm

RAW_DATA_DIR = Path("/Users/tobiaochsner/Downloads/Coal40")
MCMC_RUNS_DIR = Path("data/mcmc_runs")
MCMC_CONFIG_DIR = Path("data/mcmc_config")

dataset_name = "coal-40"

for replication_dir in tqdm(list(RAW_DATA_DIR.glob("*"))):
    if not replication_dir.is_dir():
        continue

    run_name = replication_dir.name.replace("rep", "")

    true_tree_file = next((replication_dir).glob("*.trees"))
    shutil.copyfile(true_tree_file, MCMC_CONFIG_DIR / f"{dataset_name}_{run_name}.trees")
    
    beast_xml_file = next((replication_dir).glob("*.xml"))
    shutil.copyfile(beast_xml_file, MCMC_CONFIG_DIR / f"{dataset_name}_{run_name}.xml")

    log_file = next((replication_dir / "run1").glob("*.log"))
    shutil.copyfile(log_file, MCMC_RUNS_DIR / f"{dataset_name}_{run_name}.log")
    
    trees_file = next((replication_dir / "run1").glob("*.trees"))
    shutil.copyfile(trees_file, MCMC_RUNS_DIR / f"{dataset_name}_{run_name}.trees")
