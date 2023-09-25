from math import log
from sympy.utilities.iterables import multiset_partitions
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.base.random import RNG
from discretesampling.domain.additive_structure.numbers import bell
from discretesampling.domain.additive_structure.additive_structure import AdditiveStructure


class AdditiveStructureInitialProposal(DiscreteVariableInitialProposal):  # noqa
    def __init__(self, elems):
        self.elems = elems
        n = len(self.elems)

        # Bell(n) is the number of possible (unordered) parititions of N
        # elements
        self.bell_n = bell(n)
        # Create AdditiveStructure for each possible partition
        values = [AdditiveStructure(x) for x in multiset_partitions(elems)]
        assert self.bell_n == len(values), "Should be Bell(n) different" +\
                                           "unordered partitions"

        # Assume we sample uniformly from all possible partitions
        probs = [1/self.bell_n for x in values]
        super().__init__(values, probs)

    def eval(self, x: AdditiveStructure, target=None):
        # TODO: check that elems of x match self.elems
        logprob = -log(self.bell_n)
        return logprob

    def sample(self, rng=RNG(), target=None):
        x = super().sample()
        return x
