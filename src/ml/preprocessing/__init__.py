from typing import Callable
from torch.utils.data import Dataset
from ml.preprocessing.add_clade_information import AddCladeInformation
from src.ml.preprocessing.add_taxa_names import AddTaxaNames


def preprocessing_factory(name: str, **kwargs) -> Callable[[Dataset], Dataset]:
    """Factory function for preprocessing functions."""
    match name:
        case "add_taxa_names":
            return lambda dataset: AddTaxaNames(dataset, **kwargs)
        case "add_clade_information":
            return lambda dataset: AddCladeInformation(dataset, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
