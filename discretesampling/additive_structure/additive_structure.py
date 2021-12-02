from math import log, inf
from random import randint, random, choices
import random
import numpy as np
import copy
from .numbers import binomial, stirling, bell
from sympy.utilities.iterables import multiset_partitions
from .. import discrete

class AdditiveStructure(discrete.DiscreteVariable):
    def __init__(self,discrete_set):
        self.discrete_set = discrete_set
        
    @classmethod
    def getProposalType(self):
        return AdditiveStructureProposal
    
    @classmethod
    def getTargetType(self):
        return AdditiveStructureTarget

    def split_subset(self, frac):
        """
        :param discrete_set: initial set
        :param frac: 1 or 2 indicating the probability of choosing split
        :return: new set, log probability of moving to new set, reverse log probability
        """
        single_subsets = [subset for subset in self.discrete_set if len(subset) == 1]
        multi_subsets = [subset for subset in self.discrete_set if len(subset) > 1]
        
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
        
        new_singleton = [subset for subset in self.discrete_set if len(subset) is 1]
        if len(new_set) == len(new_singleton):
            frac_rev = 1
        else:
            frac_rev = 2
        
        return AdditiveStructure(new_set)

    def merge_subset(self, frac):
        """
        :param discrete_set: initial set
        :param frac: 1 or 2 indicating the probability of choosing merge
        :return: new set, log probability of moving to new set, reverse log probability
        """
        if len(self.discrete_set) == 1:
            return copy.deepcopy(self)
        
        index_1 = randint(0, len(self.discrete_set)-1)
        remaining = list(range(len(self.discrete_set)))
        remaining.pop(index_1)
        index_2 = random.choice(remaining)
        
        first_subset = self.discrete_set[index_1]
        second_subset = self.discrete_set[index_2]
        
        new_subset = first_subset + second_subset
        new_subset.sort()
        
        new_discrete_set = [i for i in self.discrete_set if (i != first_subset and i != second_subset)]
        new_discrete_set.append(new_subset)
        new_discrete_set.sort()
        
        return AdditiveStructure(new_discrete_set)


class AdditiveStructureProposal(discrete.DiscreteVariableProposal):
    def __init__(self, current: AdditiveStructure):
        self.current = current
        self.multi_subsets = [subset for subset in self.current.discrete_set if len(subset) > 1]
    
    def eval(self, proposed: AdditiveStructure):
        #For now assume there is a valid move
        #TODO: add check that elements in current and proposed are the same
        current_num_sets = len(self.current.discrete_set)
        proposed_num_sets = len(proposed.discrete_set)
        
        logprob = -inf
        
        #Figure out whether we do a split or merge to go from current to proposed
        frac = 2
        if (current_num_sets > proposed_num_sets):
            if len(self.multi_subsets) == 0:
                frac = 1
            else:
                frac = 2
                
            logprob = self.probability_merge(frac, len(self.current.discrete_set))
        elif (proposed_num_sets > current_num_sets):
            if len(self.current.discrete_set) == 1:
                frac = 1
            else:
                frac = 2
            #Figure out which subset needs to be split to go from current to proposed
            #(i.e. the set in current which isn't in proposed)
            subset_to_split =  [i for i in self.current.discrete_set if proposed.discrete_set.count(i) < 1][0]
            logprob = self.probability_split(frac, len(self.multi_subsets), len(subset_to_split))
        else:
            #Else proposed_num_sets == current_num_sets which is not allowed
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
        elif random.random() < 0.5:
            return self.current.merge_subset(frac=2)
        else:
            return self.current.split_subset(frac=2)
    
    @staticmethod
    def probability_merge(frac, num_set):
        return -log(frac) - log(binomial(num_set, 2))
    
    @staticmethod
    def probability_split(frac, num_non_single, elements_subset):
        return -log(frac) - log(num_non_single) - log(stirling(elements_subset, 2))


class AdditiveStructureTarget(discrete.DiscreteVariableTarget):
    def __init__(self,data):
        self.data = data
    
    
    def eval(self, x: AdditiveStructure):
        #Calculate logposterior at "point" x, an instance of AdditiveStructure
        #presumably some function of x.discrete_set and some data which
        #could be defined in constructor as self.data
        logprob = -inf
        return logprob



class AdditiveStructureInitialProposal(discrete.DiscreteVariableInitialProposal):
    def __init__(self, elems):
        self.elems = elems
        n = len(self.elems)

        #Bell(n) is the number of possible (unordered) parititions of N elements
        self.bell_n = bell(n)
        #Create AdditiveStructure for each possible partition
        values = [AdditiveStructure(x) for x in multiset_partitions(elems)]
        assert self.bell_n == len(values), "Should be Bell(n) different unordered partitions"

        #Assume we sample uniformly from all possible partitions
        probs = [1/self.bell_n for x in values]
        super().__init__(values, probs)
    
    def eval(self, x: AdditiveStructure):
        #TODO: check that elems of x match self.elems
        logprob = -log(self.bell_n)
        return logprob

    def sample(self):
        x = super().sample()
        return x