import math
import numpy as np
from ...base import types
from .metrics import calculate_leaf_occurences


class TreeTarget(types.DiscreteVariableTarget):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval(self, x):
        # call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = calculate_leaf_occurences(x)
        # call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities(x)
        # p(T)
        target3 = self.evaluatePrior(x)
        return (target1+target2+target3)

    # (theta|T)
    def features_and_threshold_probabilities(self, x):
        # this need to change
        logprobabilities = []

        '''
            the likelihood of choosing the specific feature and threshold must
            be computed. We need to find out the probabilty of selecting the
            specific feature multiplied by 1/the margins. it should be
            (1/number of features) * (1/(upper bound-lower bound))
        '''
        for node in x.tree:
            logprobabilities.append(
                -math.log(len(x.X_train[0]))
                - math.log(max(x.X_train[:, node[3]]) - min(x.X_train[:, node[3]]))
            )

            # probabilities.append(math.log( 1/len(X_train[0]) *
            # 1/len(X_train[:]) ))

        logprobability = np.sum(logprobabilities)
        return (logprobability)

    def evaluatePrior(self, x):
        i = len(x.tree) - 1
        depth = 0
        while i >= 0:
            node = x.tree[i]
            next_node = x.tree[i-1]
            if node[0] == next_node[1]:
                depth += 1
            if node[0] == next_node[2]:
                depth += 1
            i -= 1
        depth = depth + 1
        logprior = math.log(self.a) - self.b*math.log(1+depth)
        return (logprior)
