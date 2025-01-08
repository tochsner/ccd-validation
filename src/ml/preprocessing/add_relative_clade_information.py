from src.ml.preprocessing.transformed_dataset import TransformedDataset

from torch.utils.data import Dataset
from Bio.Phylo.BaseTree import Tree, Clade
import torch

from src.utils.tree_utils import get_taxa_names


class AddRelativeCladeInformation(TransformedDataset):
    def __init__(self, source_dataset: Dataset):
        super().__init__(source_dataset)

    def _process_tree(
        self,
        bio_clade: Clade,
        taxa_names: list[str],
        observed_clades: list[int],
        min_branch_lengths: dict[int, float],
    ) -> tuple[int, float]:
        """Recursively collects the observed clades in the tree."""
        if bio_clade.is_terminal():
            if not bio_clade.name:
                raise ValueError("BioClade has no name.")

            node_bitstring = 1
            node_bitstring = node_bitstring << taxa_names.index(bio_clade.name)

            return node_bitstring, 0.0

        left_clade, left_subtree_height = self._process_tree(
            bio_clade.clades[0], taxa_names, observed_clades, min_branch_lengths
        )
        right_clade, right_subtree_height = self._process_tree(
            bio_clade.clades[1], taxa_names, observed_clades, min_branch_lengths
        )

        clade_bitstring = left_clade | right_clade
        observed_clades.append(clade_bitstring)

        subtree_height = max(
            bio_clade.clades[0].branch_length + left_subtree_height,
            bio_clade.clades[1].branch_length + right_subtree_height,
        )

        min_branch_lengths[clade_bitstring] = max(
            bio_clade.branch_length / (bio_clade.branch_length + subtree_height), 1e-6
        )

        return clade_bitstring, subtree_height

    def apply_initial_transform(self):
        taxa_names = self.source_dataset[0]["taxa_names"]

        all_observed_clades: set[int] = set()
        for item in self.source_dataset:
            tree: Tree = item["tree"]

            observed_clades: list[int] = []
            min_branch_lengths: dict[int, float] = {}
            _, tree_height = self._process_tree(
                tree.root, taxa_names, observed_clades, min_branch_lengths
            )

            item["clades"] = sorted(observed_clades)
            item["branch_lengths"] = torch.tensor(
                [min_branch_lengths[clade] for clade in item["clades"]], dtype=torch.float32
            )
            item["tree_height"] = tree_height

            all_observed_clades.update(observed_clades)

        all_observed_clades_list = sorted(list(all_observed_clades))
        for item in self.source_dataset:
            item["all_observed_clades"] = all_observed_clades_list

        self.transformed_data = [item for item in self.source_dataset]

    @classmethod
    def _get_clade_bitstring(cls, bio_clade: Clade, taxa_names: list[str]) -> int:
        """Recursively sets the branch lengths of the tree."""
        if bio_clade.is_terminal():
            if not bio_clade.name:
                raise ValueError("BioClade has no name.")

            node_bitstring = 1
            node_bitstring = node_bitstring << taxa_names.index(bio_clade.name)

            return node_bitstring

        left_clade = cls._get_clade_bitstring(bio_clade.clades[0], taxa_names)
        right_clade = cls._get_clade_bitstring(bio_clade.clades[1], taxa_names)

        clade_bitstring = left_clade | right_clade

        return clade_bitstring

    @classmethod
    def _set_relative_branch_lengths(
        cls,
        bio_clade: Clade,
        taxa_names: list[str],
        clade_to_branch_lengths: dict[int, float],
        subtree_height: float,
    ):
        """Recursively sets the branch lengths of the tree."""
        if bio_clade.is_terminal():
            bio_clade.branch_length = subtree_height
            return

        clade_bitstring = cls._get_clade_bitstring(bio_clade, taxa_names)
        bio_clade.branch_length = (
            subtree_height * clade_to_branch_lengths[clade_bitstring]
        )

        cls._set_relative_branch_lengths(
            bio_clade.clades[0],
            taxa_names,
            clade_to_branch_lengths,
            subtree_height - bio_clade.branch_length,
        )
        cls._set_relative_branch_lengths(
            bio_clade.clades[1],
            taxa_names,
            clade_to_branch_lengths,
            subtree_height - bio_clade.branch_length,
        )

    @classmethod
    def set_branch_lengths(
        cls,
        tree: Tree,
        branch_lengths: list[float],
        clades: list[int],
        tree_height: float,
    ):
        clade_to_branch_lengths = dict(zip(clades, branch_lengths))
        cls._set_relative_branch_lengths(
            tree.root, get_taxa_names(tree), clade_to_branch_lengths, tree_height
        )
