from src.ml.preprocessing.transformed_dataset import TransformedDataset

from torch.utils.data import Dataset
from Bio.Phylo.BaseTree import Tree, Clade
import torch

from src.utils.tree_utils import get_taxa_names


class AddAbsoluteCladeInformation(TransformedDataset):
    def __init__(self, source_dataset: Dataset):
        super().__init__(source_dataset)

    def _process_tree(
        self,
        bio_clade: Clade,
        taxa_names: list[str],
        observed_clades: list[int],
        min_branch_lengths: dict[int, float],
    ) -> int:
        """Recursively collects the observed clades in the tree."""
        if bio_clade.is_terminal():
            if not bio_clade.name:
                raise ValueError("BioClade has no name.")

            node_bitstring = 1
            node_bitstring = node_bitstring << taxa_names.index(bio_clade.name)

            return node_bitstring

        left_clade = self._process_tree(
            bio_clade.clades[0], taxa_names, observed_clades, min_branch_lengths
        )
        right_clade = self._process_tree(
            bio_clade.clades[1], taxa_names, observed_clades, min_branch_lengths
        )

        clade_bitstring = left_clade | right_clade
        observed_clades.append(clade_bitstring)

        min_branch_lengths[clade_bitstring] = min(
            bio_clade.clades[0].branch_length, bio_clade.clades[1].branch_length
        )

        return clade_bitstring

    def apply_initial_transform(self):
        taxa_names = self.source_dataset[0]["taxa_names"]

        all_observed_clades: set[int] = set()
        for item in self.source_dataset:
            tree: Tree = item["tree"]

            observed_clades: list[int] = []
            min_branch_lengths: dict[int, float] = {}
            self._process_tree(
                tree.root, taxa_names, observed_clades, min_branch_lengths
            )

            item["clades"] = sorted(observed_clades)
            item["branch_lengths"] = torch.tensor(
                [min_branch_lengths[clade] for clade in item["clades"]]
            )
            all_observed_clades.update(observed_clades)

        all_observed_clades_list = sorted(list(all_observed_clades))
        for item in self.source_dataset:
            item["all_observed_clades"] = all_observed_clades_list

        self.transformed_data = [item for item in self.source_dataset]

    def _set_branch_lengths(
        self,
        bio_clade: Clade,
        taxa_names: list[str],
        clade_to_branch_lengths: dict[int, float],
    ) -> tuple[int, float]:
        """Recursively sets the branch lengths of the tree."""
        if bio_clade.is_terminal():
            if not bio_clade.name:
                raise ValueError("BioClade has no name.")

            node_bitstring = 1
            node_bitstring = node_bitstring << taxa_names.index(bio_clade.name)

            return node_bitstring, 0.0

        left_clade, left_height = self._set_branch_lengths(
            bio_clade.clades[0], taxa_names, clade_to_branch_lengths
        )
        right_clade, right_height = self._set_branch_lengths(
            bio_clade.clades[1], taxa_names, clade_to_branch_lengths
        )

        clade_bitstring = left_clade | right_clade

        min_branch_length = clade_to_branch_lengths[clade_bitstring]

        if left_height <= right_height:
            bio_clade.clades[0].branch_length = min_branch_length + (
                right_height - left_height
            )
            bio_clade.clades[1].branch_length = min_branch_length

            return clade_bitstring, min_branch_length + right_height

        else:
            bio_clade.clades[0].branch_length = min_branch_length
            bio_clade.clades[1].branch_length = min_branch_length + (
                left_height - right_height
            )

            return clade_bitstring, min_branch_length + left_height

    def set_branch_lengths(
        self, tree: Tree, branch_lengths: list[float], clades: list[int]
    ):
        clade_to_branch_lengths = dict(zip(clades, branch_lengths))
        self._set_branch_lengths(tree.root, get_taxa_names(tree), clade_to_branch_lengths)
