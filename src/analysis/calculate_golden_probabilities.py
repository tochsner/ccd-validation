from collections import Counter
from pathlib import Path
from nexus import NexusReader
from nexus.handlers.tree import TreeHandler, Tree
from random import sample
from tqdm import tqdm
from Bio.Nexus.Trees import Tree as NexusTree


INPUT_DATASETS_DIR = Path("data/beast")
PROBABILITIES_DIR = Path("data/probabilities")


def to_pruned_newick(tree: Tree, keep_name: bool = True) -> str:
    if keep_name:
        name_of_pruned_tree = tree.name or ""
    else:
        name_of_pruned_tree = ""

    nexus_tree = NexusTree(tree.newick_string, name=name_of_pruned_tree)
    return nexus_tree.to_string()


def calculate_golden_probabilities(input_tree_file: Path, output_csv_file: Path):
    nexus_reader = NexusReader(input_tree_file)

    tree_handler: TreeHandler = nexus_reader.trees  # type: ignore

    pruned_trees = [
        to_pruned_newick(tree, keep_name=False) for tree in tree_handler.trees
    ]

    tree_counter = Counter(pruned_trees)

    outpout_str = ""

    for tree, pruned_tree in zip(tree_handler.trees, pruned_trees):
        count = tree_counter[pruned_tree]
        probability = count / len(tree_handler.trees)
        outpout_str += f"{tree.name},{probability}\n"

    output_csv_file.write_text(outpout_str)


def calculate_golden_probabilities_for_all_datasets():
    for tree_file in tqdm(list(INPUT_DATASETS_DIR.glob("*.trees"))):
        output_csv_file = PROBABILITIES_DIR / f"{tree_file.stem}-golden.csv"
        calculate_golden_probabilities(tree_file, output_csv_file)


if __name__ == "__main__":
    calculate_golden_probabilities_for_all_datasets()
