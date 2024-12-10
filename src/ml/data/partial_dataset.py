from torch.utils.data import Dataset


class PartialDataset(Dataset):
    """A dataset that only contains a subset of the data."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
