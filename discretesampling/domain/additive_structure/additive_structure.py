import numpy as np
import copy
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.random import RNG
from discretesampling.domain.additive_structure.additive_distribution import AdditiveStructureProposal
from discretesampling.domain.additive_structure.additive_target import AdditiveStructureTarget


class AdditiveStructure(DiscreteVariable):
    def __init__(self, discrete_set):
        self.discrete_set = discrete_set

    def __eq__(self, x) -> bool:
        return len(x.discrete_set) == len(self.discrete_set) and\
            all([x.discrete_set[i] == self.discrete_set[i] for i in range(len(self.discrete_set))])

    def __copy__(self):
        return AdditiveStructure(copy.deepcopy(self.discrete_set))

    @classmethod
    def getProposalType(self):
        return AdditiveStructureProposal

    @classmethod
    def getTargetType(self):
        return AdditiveStructureTarget

    def split_subset(self, frac, rng=RNG()):
        """
        :param discrete_set: initial set
        :param frac: 1 or 2 indicating the probability of choosing split
        :return: new set, log probability of moving to new set, reverse log
        probability
        """
        single_subsets = [subset for subset in self.discrete_set
                          if len(subset) == 1]
        multi_subsets = [subset for subset in self.discrete_set
                         if len(subset) > 1]

        if len(multi_subsets) == 0:     # there is no subset that can be split
            return single_subsets

        # Get a random subset from the list
        index = rng.randomInt(0, len(multi_subsets)-1)
        subset_to_split = np.array(multi_subsets[index])

        # assign each element to a subset
        split_vec = rng.randomChoices([True, False], k=len(subset_to_split))
        while np.sum(split_vec) == 0 or np.sum(np.logical_not(split_vec)) == 0:
            split_vec = rng.randomChoices([True, False], k=len(subset_to_split))

        multi_subsets.pop(index)
        multi_subsets.append(list(subset_to_split[split_vec]))
        multi_subsets.append(list(subset_to_split[np.logical_not(split_vec)]))

        new_set = multi_subsets + single_subsets
        new_set.sort()

        new_singleton = [subset for subset in self.discrete_set
                         if len(subset) == 1]
        if len(new_set) == len(new_singleton):
            frac_rev = 1
        else:
            frac_rev = 2  # noqa

        return AdditiveStructure(new_set)

    def merge_subset(self, frac, rng=RNG()):
        """
        :param discrete_set: initial set
        :param frac: 1 or 2 indicating the probability of choosing merge
        :return: new set, log probability of moving to new set, reverse log
                 probability
        """
        if len(self.discrete_set) == 1:
            return copy.copy(self)

        index_1 = rng.randomInt(0, len(self.discrete_set)-1)
        remaining = list(range(len(self.discrete_set)))
        remaining.pop(index_1)
        index_2 = rng.randomChoice(remaining)

        first_subset = self.discrete_set[index_1]
        second_subset = self.discrete_set[index_2]

        new_subset = first_subset + second_subset
        new_subset.sort()

        new_discrete_set = [i for i in self.discrete_set
                            if (i != first_subset and i != second_subset)]
        new_discrete_set.append(new_subset)
        new_discrete_set.sort()

        return AdditiveStructure(new_discrete_set)
