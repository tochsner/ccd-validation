from pathlib import Path
from typing import Optional
from src.ml.data.transformed_dataset import TransformedDataset
from src.datasets.load_trees import load_trees_from_file


class TreeDataset(TransformedDataset):
    """A dataset of trees."""

    def __init__(self, trees_file: Path):
        super().__init__()
        self.trees_file = trees_file
        self.trees = load_trees_from_file(trees_file)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        raw_item = {"tree": self.trees[index]}
        transformed_item = self._apply_transform(raw_item, self.transform)
        return transformed_item

    def apply_initial_transform(self):
        self.trees = [
            self._apply_transform(tree, self.initial_transform) for tree in self.trees
        ]


def tree_datasets(
    directory: str,
    glob: str,
    max_files: Optional[int] = None,
) -> list[TreeDataset]:
    """Builds tree datasets for the tree files in a directory."""
    files = list(Path(directory).glob(glob))

    if max_files:
        files = files[:max_files]

    return [TreeDataset(file) for file in files]
