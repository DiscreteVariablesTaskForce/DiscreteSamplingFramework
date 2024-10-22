import numpy as np
from math import log, inf
import copy
from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariableProposal


class TreeProposal(DiscreteVariableProposal):
    def __init__(self, moves_prob=[0.25, 0.1, 0.45, 0.25]):
        self.moves_prob = moves_prob  # good for Poisson and heart l = 12, and diabetes l = 10

    @classmethod
    def norm(self, tree):
        return len(tree.tree)

    @classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    def heuristic(self, x, y):
        return y < x or abs(x-y) < 2

    def sample(self, start_tree, rng=RNG(), num_nodes=10):
        # self.moves_prob = [0.4, 0.1, 0.1, 0.4] # Good for chipman
        # initialise the probabilities of each move
        moves = ["prune", "swap", "change", "grow"]  # noqa
        moves_prob = self.moves_prob
        if len(start_tree.tree) == 1:
            moves_prob = [0.0, 0.0, 0.5, 0.5]
        elif len(start_tree.tree) >= num_nodes:
            moves_prob = [0.1, 0.1, 0.8, 0.0]
        random_number = rng.random()
        moves_probabilities = np.cumsum(moves_prob)
        newTree = copy.copy(start_tree)
        if random_number < moves_probabilities[0]:
            # prune
            newTree = newTree.prune(rng=rng)

        elif random_number < moves_probabilities[1]:
            # swap
            newTree = newTree.swap(rng=rng)

        elif random_number < moves_probabilities[2]:
            # change
            newTree = newTree.change(rng=rng)

        else:
            # grow
            newTree = newTree.grow(rng=rng)

        return newTree

    
    def prunable_node_indices(self, tree):
        candidates = []
        for i in range(1, len(tree.tree)): # cannot prune the root
            node_to_prune = tree.tree[i]
            if ((node_to_prune[1] in tree.leafs) and (node_to_prune[2] in tree.leafs)):
                candidates.append(i)
        return(candidates)
    
    def eval(self, start_tree, sampledTree):
        initialTree = start_tree
        moves_prob = self.moves_prob
        logprobability = -inf
        if len(initialTree.tree) == 1:
            moves_prob = [0.0, 0.0, 0.5, 0.5]

        nodes_differences = [i for i in sampledTree.tree + initialTree.tree
                             if i not in sampledTree.tree or
                             i not in initialTree.tree]
        # In order to get sampledTree from initialTree we must have:
        # Grow
        if (len(initialTree.tree) == len(sampledTree.tree)-1):
            logprobability = (log(moves_prob[3])
                              - log(len(initialTree.X_train[0]))
                              - log(len(initialTree.X_train[:]))
                              - log(len(initialTree.leafs)))
        # Prune
        elif (len(initialTree.tree) > len(sampledTree.tree)):
            logprobability = (log(moves_prob[0])
                              - log(len(self.prunable_node_indices(initialTree))))
        # Change
        elif (
            len(initialTree.tree) == len(sampledTree.tree)
            and (
                len(nodes_differences) == 2
                or len(nodes_differences) == 0
            )
        ):
            logprobability = (log(moves_prob[2])
                              - log(len(initialTree.tree))
                              - log(len(initialTree.X_train[0]))
                              - log(len(initialTree.X_train[:])))
        # swap
        elif (len(nodes_differences) == 4 and len(initialTree.tree) > 1):
            logprobability = (log(moves_prob[1])
                              - log(len(initialTree.tree))
                              - log(len(initialTree.tree) - 1)
                              + log(2))

        return logprobability


def forward(forward, forward_probability):
    forward.append(forward_probability)
    forward_probability = np.sum(forward)
    return forward_probability


def reverse(forward, reverse_probability):
    reverse_probability = reverse_probability + np.sum(forward)
    return reverse_probability
