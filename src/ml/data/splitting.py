
from src.ml.utils.set_seed import set_seed
from random import shuffle

from torch.utils.data import Dataset, Subset


def create_data_splits(
    dataset: Dataset,
    train_fraction: float,
    test_fraction: float,
    seed: int = 0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Splits a dataset into train, validation and test set."""
    set_seed(seed)

    num_samples = len(dataset)  # type: ignore
    num_train_samples = int(train_fraction * num_samples)
    num_test_samples = int(test_fraction * num_samples)

    all_indices = list(range(num_samples))
    shuffle(all_indices)

    train_indices = all_indices[:num_train_samples]
    test_indices = all_indices[
        num_train_samples : num_train_samples + num_test_samples
    ]
    val_indices = all_indices[num_train_samples + num_test_samples :]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )
