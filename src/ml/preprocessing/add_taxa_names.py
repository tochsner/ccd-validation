from src.ml.preprocessing.transformed_dataset import TransformedDataset

from src.utils.tree_utils import get_taxa_names


class AddTaxaNames(TransformedDataset):
    def initial_transform(self, item):
        item["taxa_names"] = get_taxa_names(item["tree"])
        return item
