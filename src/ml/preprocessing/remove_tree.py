from src.ml.preprocessing.transformed_dataset import TransformedDataset


class RemoveTree(TransformedDataset):
    def initial_transform(self, item):
        del item["tree"]
        return item
