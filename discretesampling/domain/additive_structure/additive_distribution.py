from math import log, inf
from discretesampling.domain.additive_structure.numbers import binomial, stirling
from discretesampling.base.types import DiscreteVariableProposal
from discretesampling.base.random import RNG


class AdditiveStructureProposal(DiscreteVariableProposal):
    def __init__(self):
        pass

    @classmethod
    def norm(self, x):
        # Number of sets in x
        return len(x.discrete_set)

    @classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    def heuristic(self, x, y):
        # At most ones more or one fewer sets
        return abs(y-x) == 1

    def eval(self, current, proposed):
        # For now assume there is a valid move
        current_num_sets = len(current.discrete_set)
        proposed_num_sets = len(proposed.discrete_set)
        curr = [tuple(t) for t in current.discrete_set]
        propose = [tuple(t) for t in proposed.discrete_set]
        logprob = -inf

        # Figure out whether we do a split or merge to go from current to
        # proposed
        multi_subsets = self.generateSubsets(current)

        frac = 2
        if (current_num_sets > proposed_num_sets) and \
                len(set(curr).difference(propose)) == 2 and \
                len(set(propose).difference(curr)) == 1:

            if len(multi_subsets) == 0:
                frac = 1
            else:
                frac = 2

            logprob = self.probability_merge(frac, len(current.discrete_set))

        elif (proposed_num_sets > current_num_sets) and \
                len(set(propose).difference(curr)) == 2 and \
                len(set(curr).difference(propose)) == 1:

            if len(current.discrete_set) == 1:
                frac = 1
            else:
                frac = 2
            # Figure out which subset needs to be split to go from current to
            # proposed (i.e. the set in current which isn't in proposed)
            subset_to_split = [i for i in current.discrete_set
                               if proposed.discrete_set.count(i) < 1][0]
            logprob = self.probability_split(frac, len(multi_subsets),
                                             len(subset_to_split))
        else:
            # Else proposed_num_sets == current_num_sets which is not allowed
            logprob = -inf

        return logprob

    def sample(self, current, rng=RNG()):
        """
        Given an initial set, if both split and merge can be performed,
        we randomly choose a move and sample a new set based on the initial.
        """
        multi_subsets = self.generateSubsets(current)
        if len(multi_subsets) == 0:
            return current.merge_subset(frac=1)
        elif len(current.discrete_set) == 1:
            return current.split_subset(frac=1)
        elif rng.random() < 0.5:
            return current.merge_subset(frac=2, rng=rng)
        else:
            return current.split_subset(frac=2, rng=rng)

    def generateSubsets(self, startSet):
        multi_subsets = [subset for subset in startSet.discrete_set
                         if len(subset) > 1]
        return multi_subsets

    @staticmethod
    def probability_merge(frac, num_set):
        return -log(frac) - log(binomial(num_set, 2))

    @staticmethod
    def probability_split(frac, num_non_single, elements_subset):
        return -log(frac) - log(num_non_single) -\
            log(stirling(elements_subset, 2))
