from torch.utils.data import Dataset
from abc import ABC


class TransformedDataset(ABC, Dataset):
    """A dataset that allows to transform the data on the fly or at once."""

    def __init__(self, source_dataset: Dataset):
        self.source_dataset = source_dataset
        self.transformed_data = None
        self.apply_initial_transform()

    def apply_initial_transform(self):
        """Applies the initial transform to the data."""
        self.transformed_data = []

        for item in self.source_dataset:
            transformed_item = self.initial_transform(item)
            self.transformed_data.append(transformed_item)

    def initial_transform(self, item):
        """Applies the initial transform an item."""
        return item

    def transform(self, item):
        """Applies the transform to an item."""
        return item

    def __getitem__(self, index):
        if self.transformed_data is not None:
            item = self.transformed_data[index]
        else:
            item = self.source_dataset[index]

        return self.transform(item)

    def __len__(self):
        if self.transformed_data is not None:
            return len(self.transformed_data)
        else:
            return len(self.source_dataset)  # type: ignore
