from io import StringIO
from Bio.Phylo._io import write, parse
from Bio.Phylo.BaseTree import Tree


def to_pruned_newick(tree: Tree) -> str:
    with StringIO() as string:
        write(tree, string, format="newick", plain=True)
        return string.getvalue()


def from_pruned_newick(newick: str) -> Tree:
    with StringIO() as string:
        string.write(newick)
        string.seek(0)
        return next(parse(string, format="newick"))


def get_tree_height(tree: Tree) -> float:
    return tree.distance(tree.root, tree.get_terminals()[0])


def get_taxa_names(tree: Tree) -> list[str]:
    return [terminal.name for terminal in tree.get_terminals()]
