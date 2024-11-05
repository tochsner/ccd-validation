from Bio.Phylo.BaseTree import Tree as BioTree, Clade as BioClade
import pandas as pd
from src.distribution_analysis.clade import (
    ObservedLeaf,
    ObservedCladeSplit,
    ObservedNode,
)
from src.utils.tree_utils import to_pruned_newick


def _process_tree(
    observed_nodes: list[ObservedNode],
    observed_clade_splits: list[ObservedCladeSplit],
    tree_index: int,
    newick_tree: str,
    bio_clade: BioClade,
    taxa_names: list[str],
    height: float = 0,
    distance_to_root: int = 0,
) -> ObservedNode:
    """Recursively collects the observed nodes and observed clade splits in the tree."""
    if bio_clade.is_terminal():
        if not bio_clade.name:
            raise ValueError("BioClade has no name.")

        node_bitstring = 1
        node_bitstring = node_bitstring << taxa_names.index(bio_clade.name)
        node = ObservedLeaf(
            node_bitstring=node_bitstring,
            height=height,
            distance_to_root=distance_to_root,
            distance_to_leaf=0,
            tree_index=tree_index,
            newick_tree=newick_tree,
        )
        observed_nodes.append(node)
        return node

    left_clade = _process_tree(
        observed_nodes,
        observed_clade_splits,
        tree_index,
        newick_tree,
        bio_clade.clades[0],
        taxa_names,
        height + (bio_clade.clades[0].branch_length or 0),
        distance_to_root + 1,
    )
    right_clade = _process_tree(
        observed_nodes,
        observed_clade_splits,
        tree_index,
        newick_tree,
        bio_clade.clades[1],
        taxa_names,
        height + (bio_clade.clades[1].branch_length or 0),
        distance_to_root + 1,
    )

    split_bitstring = left_clade.node_bitstring | right_clade.node_bitstring

    clade_split = ObservedCladeSplit(
        node_bitstring=split_bitstring,
        bitstring=(left_clade.node_bitstring, right_clade.node_bitstring),
        height=height,
        left_clade=left_clade,
        right_clade=right_clade,
        distance_to_root=distance_to_root,
        distance_to_leaf=1
        + max(left_clade.distance_to_leaf, right_clade.distance_to_leaf),
        tree_index=tree_index,
        newick_tree=newick_tree,
    )
    observed_nodes.append(clade_split)
    observed_clade_splits.append(clade_split)

    return clade_split


def get_observed_nodes(
    trees: list[BioTree], taxa_names: list[str]
) -> tuple[list[ObservedNode], list[ObservedCladeSplit]]:
    """Collects all observed nodes and clade splits in the given trees.
    Note that the returned lists contain one element per occurrence,
    so the same node and the same clade split can occur multiple times."""
    observed_nodes: list[ObservedNode] = []
    observed_clade_splits: list[ObservedCladeSplit] = []

    for i, tree in enumerate(list(trees)):
        _process_tree(
            observed_nodes,
            observed_clade_splits,
            i,
            to_pruned_newick(tree),
            tree.root,
            taxa_names,
        )

    return observed_nodes, observed_clade_splits


def get_clade_split_df(clade_splits: list[ObservedCladeSplit]) -> pd.DataFrame:
    """Constructs a dataframe containing one row per observed clade split. It contains
    information about the clade splits and the associated branche lengths."""
    df_dict = {
        "tree_index": [],
        "newick_tree": [],
        "clade_split": [],
        "left_branch": [],
        "right_branch": [],
        "min_branch": [],
        "min_branch_down": [],
        "max_branch": [],
        "distance_to_root": [],
        "distance_to_leaf": [],
        "height": [],
    }

    for clade_split in clade_splits:
        df_dict["tree_index"].append(clade_split.tree_index)
        df_dict["newick_tree"].append(clade_split.newick_tree)
        df_dict["clade_split"].append(str(clade_split))

        left = clade_split.left_clade.height - clade_split.height
        right = clade_split.right_clade.height - clade_split.height

        df_dict["left_branch"].append(left)
        df_dict["right_branch"].append(right)

        df_dict["min_branch"].append(min(left, right))
        df_dict["max_branch"].append(max(left, right))

        if left < right:
            if not isinstance(clade_split.left_clade, ObservedCladeSplit):
                df_dict["min_branch_down"].append(None)
            else:
                left = clade_split.left_clade.left_clade.height - clade_split.left_clade.height
                right = clade_split.left_clade.right_clade.height - clade_split.left_clade.height
                df_dict["min_branch_down"].append(min(left, right))
        else:
            if not isinstance(clade_split.right_clade, ObservedCladeSplit):
                df_dict["min_branch_down"].append(None)
            else:
                left = clade_split.right_clade.left_clade.height - clade_split.right_clade.height
                right = clade_split.right_clade.right_clade.height - clade_split.right_clade.height
                df_dict["min_branch_down"].append(min(left, right))

        df_dict["distance_to_root"].append(clade_split.distance_to_root)
        df_dict["distance_to_leaf"].append(clade_split.distance_to_leaf)
        df_dict["height"].append(clade_split.height)

    df_branches = pd.DataFrame(df_dict)

    df_branches["clade_split_count"] = df_branches.groupby("clade_split")["clade_split"].transform("count")

    return df_branches
