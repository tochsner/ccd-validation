from typing import Callable
from torch.utils.data import Dataset
from src.ml.preprocessing.add_absolute_clade_information import AddAbsoluteCladeInformation
from src.ml.preprocessing.add_relative_clade_information import AddRelativeCladeInformation
from src.ml.preprocessing.add_taxa_names import AddTaxaNames
from src.ml.preprocessing.remove_tree import RemoveTree


def preprocessing_factory(name: str, **kwargs) -> Callable[[Dataset], Dataset]:
    """Factory function for preprocessing functions."""
    match name:
        case "add_taxa_names":
            return lambda dataset: AddTaxaNames(dataset, **kwargs)
        case "add_absolute_clade_information":
            return lambda dataset: AddAbsoluteCladeInformation(dataset, **kwargs)
        case "add_relative_clade_information":
            return lambda dataset: AddRelativeCladeInformation(dataset, **kwargs)
        case "remove_tree":
            return lambda dataset: RemoveTree(dataset, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
