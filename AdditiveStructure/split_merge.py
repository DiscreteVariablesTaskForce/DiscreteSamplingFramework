from math import log
from random import randint, random, choices
import random
import numpy as np
from numbers import binomial, stirling


class ProposedMoves:
    def __init__(self):
        pass


class SplitMergeMove(ProposedMoves):
    def propose(self, discrete_set):
        """
        Given an initial set, if both split and merge can be performed,
        we randomly choose a move and sample a new set based on the initial.
        """
        multi_subsets = [subset for subset in discrete_set if len(subset) > 1]

        if len(multi_subsets) == 0:
            return self.merge_subset(discrete_set, frac=1)
        elif len(discrete_set) == 1:
            return self.split_subset(discrete_set, frac=1)
        elif random() < 0.5:
            return self.merge_subset(discrete_set, frac=2)
        else:
            return self.split_subset(discrete_set, frac=2)

    def split_subset(self, discrete_set, frac):
        """
        :param discrete_set: initial set
        :param frac: 1 or 2 indicating the probability of choosing split
        :return: new set, log probability of moving to new set, reverse log probability
        """
        single_subsets = [subset for subset in discrete_set if len(subset) is 1]
        multi_subsets = [subset for subset in discrete_set if len(subset) > 1]

        if len(multi_subsets) == 0:     # there is no subset that can be split
            return single_subsets

        # Get a random subset from the list
        index = randint(0, len(multi_subsets)-1)
        subset_to_split = np.array(multi_subsets[index])

        # assign each element to a subset
        split_vec = choices([True, False], k=len(subset_to_split))
        while np.sum(split_vec) == 0 or np.sum(np.logical_not(split_vec)) == 0:
            split_vec = choices([True, False], k=len(subset_to_split))

        multi_subsets.pop(index)
        multi_subsets.append(list(subset_to_split[split_vec]))
        multi_subsets.append(list(subset_to_split[np.logical_not(split_vec)]))

        new_set = multi_subsets + single_subsets
        new_set.sort()

        new_singleton = [subset for subset in discrete_set if len(subset) is 1]
        if len(new_set) == len(new_singleton):
            frac_rev = 1
        else:
            frac_rev = 2

        proposal_logprob = self.probability_split(frac, len(multi_subsets), len(subset_to_split))
        reverse_logprob = self.probability_merge(frac_rev, len(new_set))

        return new_set, proposal_logprob, reverse_logprob

    def merge_subset(self, discrete_set, frac):
        """
        :param discrete_set: initial set
        :param frac: 1 or 2 indicating the probability of choosing merge
        :return: new set, log probability of moving to new set, reverse log probability
        """
        if len(discrete_set) == 1:
            return discrete_set

        index_1 = randint(0, len(discrete_set)-1)
        remaining = list(range(len(discrete_set)))
        remaining.pop(index_1)
        index_2 = randint(0, len(remaining)-1)

        first_subset = discrete_set[index_1]
        second_subset = discrete_set[index_2]

        new_subset = first_subset + second_subset
        new_subset.sort()

        discrete_set = [i for i in discrete_set if i != first_subset or i != second_subset]
        discrete_set.append(new_subset)
        discrete_set.sort()

        multi_subsets = [subset for subset in discrete_set if len(subset) > 1]

        if len(discrete_set) == 1:
            frac_rev = 1
        else:
            frac_rev = 2

        proposal_logprob = self.probability_merge(frac, len(discrete_set))
        reverse_logprob = self.probability_split(frac_rev, len(multi_subsets), len(new_subset))

        return discrete_set, proposal_logprob, reverse_logprob

    @staticmethod
    def probability_merge(frac, num_set):
        return -log(frac) - log(binomial(num_set, 2))

    @staticmethod
    def probability_split(frac, num_non_single, elements_subset):
        return -log(frac) - log(num_non_single) - log(stirling(elements_subset, 2))
