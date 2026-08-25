import numpy as np

from discretesampling.base import types
from discretesampling.domain.incremental_decision_tree.tree_ops import IncBDTree


class IncrementalTree(IncBDTree, types.DiscreteVariable):
    """
    A particle for the incremental decision tree domain.
    """

    @classmethod
    def getProposalType(cls):
        from discretesampling.domain.incremental_decision_tree.proposals.mh \
            import IncrementalTreeProposal
        return IncrementalTreeProposal

    @classmethod
    def getTargetType(cls):
        from discretesampling.domain.incremental_decision_tree.target \
            import IncrementalTreeTarget
        return IncrementalTreeTarget

    @classmethod
    def getInitialProposalType(cls):
        from discretesampling.domain.incremental_decision_tree.initial_proposal \
            import IncrementalTreeInitialProposal
        return IncrementalTreeInitialProposal

    # -------------------- identity -------------------- #

    def __eq__(self, other):
        if not isinstance(other, IncBDTree):
            return NotImplemented
        if len(self.tree) != len(other.tree):
            return False
        mine = sorted([int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])]
                      for r in self.tree)
        theirs = sorted([int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])]
                        for r in other.tree)
        if mine != theirs:
            return False
        if set(self.leaf_idx) != set(other.leaf_idx):
            return False
        return all(np.array_equal(self.counts[leaf], other.counts[leaf])
                   for leaf in self.counts)

    def __copy__(self):
        return self.deep_copy()

    def __deepcopy__(self, memo):
        return self.deep_copy()

    def __str__(self):
        return (f"IncrementalTree({len(self.tree)} nodes, "
                f"{len(self.leaf_idx)} leaves)")

    __repr__ = __str__

    # -------------------- encoding -------------------- #

    @classmethod
    def encode(cls, x):
        """
        Encode a tree as a 1D array of floats, for use in a particle's state vector.
        The first element is the number of rows in the tree, followed by the
        rows themselves, flattened. Each row is 6 floats:
        [node_id, feat, thr, left_child, right_child, leaf_class]
        """
        rows = np.asarray(x.tree, dtype=np.float64).ravel() if x.tree \
            else np.empty(0, dtype=np.float64)
        return np.hstack((np.array([len(x.tree)], dtype=np.float64), rows))

    @classmethod
    def decode(cls, x, particle):
        n_rows = int(x[0])
        rows = np.asarray(x[1:1 + 6 * n_rows], dtype=np.float64).reshape(n_rows, 6)
        tree = [[int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4]), int(r[5])]
                for r in rows]
        return cls.from_rows(particle.problem, tree)
