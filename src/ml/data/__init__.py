from typing import Iterable

from src.ml.data.tree_dataset import tree_datasets
from src.ml.data.dummy_dataset import dummy_datasets
from torch.utils.data import Dataset


def data_sets_factory(name: str, **kwargs) -> Iterable[tuple[str, Dataset]]:
    """Factory function for data sets."""
    match name:
        case "dummy_datasets":
            return dummy_datasets()

        case "tree_datasets":
            return tree_datasets(**kwargs)

        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
