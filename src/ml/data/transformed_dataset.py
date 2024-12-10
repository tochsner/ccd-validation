from typing import Callable
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class TransformedDataset(ABC, Dataset):
    """A dataset that allows to transform the data on the fly or at once."""

    def __init__(self):
        self.initial_transform: list[Callable] = []
        self.transform: list[Callable] = []

    @abstractmethod
    def apply_initial_transform(self):
        raise NotImplementedError

    def _apply_transform(self, item, transforms: list[Callable]):
        for transform in transforms:
            item = transform(item)
        return item
