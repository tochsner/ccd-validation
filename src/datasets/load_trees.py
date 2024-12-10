from enum import Enum
import itertools
from pathlib import Path
from random import sample
from typing import Optional
from Bio.Phylo.BaseTree import Tree
from Bio.Phylo._io import parse
from tqdm import tqdm


class TreeDataset(Enum):
    YULE_10 = "Yule-10"
    DS_1 = "DS1"
    DS_4 = "DS4"


def load_trees(
    dataset: TreeDataset,
    max_trees: Optional[int] = None,
    max_files: Optional[int] = None,
) -> list[Tree]:
    """Loads trees for the given datasets."""
    match dataset:
        case TreeDataset.YULE_10:
            files = Path("data/mcmc_runs").glob("yule-10-*.trees")
        case TreeDataset.DS_1:
            files = Path("data/ds/DS1_HKY/config7").glob("**/*.trees")
        case TreeDataset.DS_4:
            files = Path("data/ds/DS4_HKY/config7").glob("**/*.trees")

    trees: list[Tree] = []
    for i, file in tqdm(enumerate(list(files))):
        if max_trees and max_trees <= len(trees):
            break
        if max_files and max_files <= i:
            print("break")
            break

        print(file)

        trees += parse(file, "nexus")

    if max_trees and max_trees < len(trees):
        trees = sample(trees, max_trees)

    return trees


def load_trees_from_file(
    tree_file: Path, max_trees: Optional[int] = None
) -> list[Tree]:
    """Loads trees for the given tree file."""
    return list(itertools.islice(parse(tree_file, "nexus"), max_trees))
