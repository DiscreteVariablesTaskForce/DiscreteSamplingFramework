import numpy as np
from math import log, inf
import copy
from ...base.random import Random
from ...base import types


class TreeProposal(types.DiscreteVariableProposal):
    def __init__(self, tree):
        self.X_train = tree.X_train
        self.y_train = tree.y_train
        self.tree = copy.deepcopy(tree)

    @classmethod
    def norm(self, tree):
        return len(tree.tree)

    @classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    def heuristic(self, x, y):
        return y < x or abs(x-y) < 2

    def sample(self):
        # initialise the probabilities of each move
        moves = ["prune", "swap", "change", "grow"]  # noqa
        moves_prob = [0.4, 0.1, 0.1, 0.4]
        if len(self.tree.tree) == 1:
            moves_prob = [0.0, 0.0, 0.5, 0.5]
        moves_probabilities = np.cumsum(moves_prob)
        random_number = Random().eval()
        newTree = copy.deepcopy(self.tree)
        if random_number < moves_probabilities[0]:
            # prune
            newTree = newTree.prune()

        elif random_number < moves_probabilities[1]:
            # swap
            newTree = newTree.swap()

        elif random_number < moves_probabilities[2]:
            # change
            newTree = newTree.change()

        else:
            # grow
            newTree = newTree.grow()

        return newTree

    def eval(self, sampledTree):
        initialTree = self.tree
        moves_prob = [0.4, 0.1, 0.1, 0.4]
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
                              - log(len(initialTree.tree) - 1))
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
