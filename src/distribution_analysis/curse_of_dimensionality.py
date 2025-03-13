import pandas as pd
import seaborn as sns

from pathlib import Path

sns.set_style("darkgrid")

DISTANCES_FILE = Path("data/distances_data/rf_distances.csv")

def distance_experiment():
    distances = pd.read_csv(DISTANCES_FILE)
    
    distances["dataset"] = distances.tree.apply(lambda x: x.split("_")[0])
    distances["ratio"] = distances["avg_max_pairwise_distance"] / distances["avg_min_pairwise_distance"]

    sns.lineplot(data=distances, x="dataset", y="distance", hue="model")
    


if __name__ == "__main__":
    distance_experiment()