from pathlib import Path
from nexus import NexusReader
from nexus.handlers.tree import TreeHandler
from random import sample
from tqdm import tqdm


DATASET_SIZES = [3, 10, 30, 100, 300, 1_000, 3_000, 10_000]
INPUT_DATASETS_DIR = Path("data/beast")
SUBSAMPLED_DATASETS_DIR = Path("data/subsampled")


def subsample_trees(input_tree_file: Path, output_tree_file: Path, num_trees: int):
    nexus_reader = NexusReader(input_tree_file)

    tree_hanlder: TreeHandler = nexus_reader.trees  # type: ignore
    tree_hanlder.trees = sample(tree_hanlder.trees, num_trees)

    nexus_reader.write_to_file(output_tree_file)


def create_subsampled_datasets():
    for tree_file in tqdm(list(INPUT_DATASETS_DIR.glob("*.trees"))):
        for dataset_size in DATASET_SIZES:
            output_tree_file = (
                SUBSAMPLED_DATASETS_DIR / f"{tree_file.stem}-{dataset_size}.trees"
            )
            subsample_trees(tree_file, output_tree_file, dataset_size)


if __name__ == "__main__":
    create_subsampled_datasets()
