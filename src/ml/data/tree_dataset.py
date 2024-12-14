from pathlib import Path
from typing import Iterable, Optional
from torch.utils.data import Dataset
from src.datasets.load_trees import load_trees_from_file


class TreeDataset(Dataset):
    """A dataset of trees."""

    def __init__(self, trees_file: Path):
        super().__init__()
        self.trees_file = trees_file
        self.trees = load_trees_from_file(trees_file)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        return {
            "tree": self.trees[index],
            "trees_file": str(self.trees_file),
            "tree_index": index,
        }


def tree_datasets(
    directory: str,
    glob: str,
    max_files: Optional[int] = None,
) -> Iterable[tuple[str, TreeDataset]]:
    """Builds tree datasets for the tree files in a directory."""
    files = list(Path(directory).glob(glob))

    if max_files:
        files = files[:max_files]

    yield from ((file.stem, TreeDataset(file)) for file in files)
