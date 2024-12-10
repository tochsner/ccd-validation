from typing import Sequence

from src.ml.data.transformed_dataset import TransformedDataset
from src.ml.data.tree_dataset import tree_datasets


def data_sets_factory(name: str, **kwargs) -> Sequence[TransformedDataset]:
    """Factory function for data sets."""
    match name:
        case "tree_datasets":
            return tree_datasets(**kwargs)

        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
