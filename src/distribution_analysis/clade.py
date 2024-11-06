from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ObservedNode(ABC):
    tree_index: int
    newick_tree: str
    node_bitstring: int
    parent_height: float | int
    height: float | int
    distance_to_root: int
    distance_to_leaf: int

    def __str__(self) -> str:
        return "{0:010b}".format(self.node_bitstring)

    @property
    @abstractmethod
    def is_leaf(self):
        raise NotImplementedError


@dataclass
class ObservedCladeSplit(ObservedNode):
    bitstring: tuple[int, int]
    left_clade: ObservedNode
    right_clade: ObservedNode

    def __str__(self) -> str:
        return "{0:010b}||{1:010b}".format(*self.bitstring)

    @property
    def min_branch_length(self):
        return min(
            self.left_clade.height - self.height, self.right_clade.height - self.height
        )
    
    @property
    def min_branch_clade(self):
        return self.left_clade if self.left_clade.height < self.right_clade.height else self.right_clade

    @property
    def is_leaf(self):
        return False


@dataclass
class ObservedLeaf(ObservedNode):
    @property
    def is_leaf(self):
        return True
