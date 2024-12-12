from pathlib import Path
from typing import Optional
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
            "branch_lengths": torch.randn(10) * 0.001, 
            "clades_one_hot": torch.randint(0, 2, (20,)).float(),
        }


def dummy_datasets() -> list[DummyDataset]:
    return [DummyDataset()]
