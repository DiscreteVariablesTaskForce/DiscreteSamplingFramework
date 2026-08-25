import math

from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.domain.incremental_decision_tree.incremental_tree import IncrementalTree


class IncrementalTreeInitialProposal(DiscreteVariableInitialProposal):
    """
    Initial proposal for the incremental decision tree domain.
    The initial proposal is a stump, i.e. a tree with no splits,
    a single leaf where all training data is routed.
    """

    def __init__(self, problem):
        self.problem = problem

    def sample(self, rng=RNG(), target=None):
        return IncrementalTree.stump(self.problem)

    def eval(self, x, target=None):
        if len(x.tree) == 0:
            return 0.0
        return -math.inf

    def __repr__(self):
        return f"<IncrementalTreeInitialProposal {self.problem}>"
