import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.len = 5000

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {
            "branch_lengths": 5 + torch.randn(9).pow(2), 
            "clades": list(range(9)),
            "taxa_names": [str(i) for i in range(10)],
        }


def dummy_datasets() -> list[tuple[str, DummyDataset]]:
    return [("dummy", DummyDataset())]
