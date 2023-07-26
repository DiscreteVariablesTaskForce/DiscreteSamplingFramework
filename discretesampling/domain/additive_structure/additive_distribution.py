from math import log, inf
from .numbers import binomial, stirling
from ...base import types
from ...base.random import RNG


class AdditiveStructureProposal(types.DiscreteVariableProposal):
    def __init__(self, current, rng=RNG()):
        self.current = current
        self.multi_subsets = [subset for subset in self.current.discrete_set
                              if len(subset) > 1]
        self.rng = rng

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

    def eval(self, proposed):
        # For now assume there is a valid move
        current_num_sets = len(self.current.discrete_set)
        proposed_num_sets = len(proposed.discrete_set)
        current = [tuple(t) for t in self.current.discrete_set]
        propose = [tuple(t) for t in proposed.discrete_set]
        logprob = -inf

        # Figure out whether we do a split or merge to go from current to
        # proposed
        frac = 2
        if (current_num_sets > proposed_num_sets) and \
                len(set(current).difference(propose)) == 2 and \
                len(set(propose).difference(current)) == 1:

            if len(self.multi_subsets) == 0:
                frac = 1
            else:
                frac = 2

            logprob = self.probability_merge(frac, len(self.current.discrete_set))

        elif (proposed_num_sets > current_num_sets) and \
                len(set(propose).difference(current)) == 2 and \
                len(set(current).difference(propose)) == 1:

            if len(self.current.discrete_set) == 1:
                frac = 1
            else:
                frac = 2
            # Figure out which subset needs to be split to go from current to
            # proposed (i.e. the set in current which isn't in proposed)
            subset_to_split = [i for i in self.current.discrete_set
                               if proposed.discrete_set.count(i) < 1][0]
            logprob = self.probability_split(frac, len(self.multi_subsets),
                                             len(subset_to_split))
        else:
            # Else proposed_num_sets == current_num_sets which is not allowed
            logprob = -inf

        return logprob

    def sample(self):
        """
        Given an initial set, if both split and merge can be performed,
        we randomly choose a move and sample a new set based on the initial.
        """
        if len(self.multi_subsets) == 0:
            return self.current.merge_subset(frac=1)
        elif len(self.current.discrete_set) == 1:
            return self.current.split_subset(frac=1)
        elif self.rng.random() < 0.5:
            return self.current.merge_subset(frac=2, rng=self.rng)
        else:
            return self.current.split_subset(frac=2, rng=self.rng)

    @staticmethod
    def probability_merge(frac, num_set):
        return -log(frac) - log(binomial(num_set, 2))

    @staticmethod
    def probability_split(frac, num_non_single, elements_subset):
        return -log(frac) - log(num_non_single) -\
            log(stirling(elements_subset, 2))
