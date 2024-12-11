from typing import Sequence

from src.ml.data.tree_dataset import tree_datasets
from torch.utils.data import Dataset


def data_sets_factory(name: str, **kwargs) -> Sequence[Dataset]:
    """Factory function for data sets."""
    match name:
        case "tree_datasets":
            return tree_datasets(**kwargs)

        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
